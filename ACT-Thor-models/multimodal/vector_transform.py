
import json
import os
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import numpy
import pandas
import seaborn

import torch
import torch.nn as nn

from torch.nn import functional as F
from collections import OrderedDict, defaultdict

from tqdm import tqdm

import sys
sys.path.append('.')  # needed for executing from top directory

import plots

from visual_features.utils import load_model_from_path
from visual_features.data import get_data
from visual_features.data import __default_dataset_path__
from visual_features.utils import setup_argparser, get_optimizer
from visual_features.data import VecTransformDataset  # needed for hold-out procedures

from multimodal.vect_models import *
from multimodal.vect_models import __allowed_activations__ as allowed_activations
from multimodal.vect_models import __allowed_conditioning_methods__ as allowed_conditioning_methods


_ALLOWED_MODELS = ['moca-rn', 'clip-rn']
_ALLOWED_TRANSFORM_MODULES = {
    'Action-Matrix': LinearVectorTransform,
    'Concat-Linear': LinearConcatVecT,
    'Concat-Multi': ConditionalFCNVecT
}

__default_train_config__ = {
    # General settings
    'batch_size': (256, ),
    'learning_rate': (1e-2, ),
    'epochs': (20, ),
    'optimizer': ('adam', ['adamw', 'adam', 'sgd']),
    'scheduler': ('none', ['step', 'none']),

    # VecT settings
    'vect_model': ('Concat-Linear', list(_ALLOWED_TRANSFORM_MODULES.keys())),
    'use_bn': False,
    'dropout_p': (0.0, [0.0, 0.3, 0.5, 0.7, 0.9]),
    'activation': ('none', allowed_activations),
    'hidden_sizes': (None, ),
    'use_contrastive': False,
    'use_infonce': True,
    'conditioning_method': ('embedding', ['concat', 'embedding']),

    # Dataset settings
    'extractor_model': ('moca-rn', _ALLOWED_MODELS),
    'use_regression': False,
    'hold_out_procedure': ('samples', VecTransformDataset.allowed_holdout_procedures),
    'cs_type': ('CS4', ['CS3', 'CS4']),

    # Statistics
    'statistical_iterations': (5, ),

    # Others
    'data_path': (__default_dataset_path__,),
    'save_model': False,
    'save_path': ('new-vect-results', ),
    'load_path': (None, ),

    'device': ('cuda' if torch.cuda.is_available() else 'cpu', ['cuda', 'cpu']),
    'dataparallel': False,
    'use_wandb': False
}

__tosave_hyperparams__ = [
    'batch_size',
    'learning_rate',
    'epochs',
    'optimizer',
    'scheduler',
    'vect_model',
    'use_bn',
    'dropout_p',
    'activation',
    'extractor_model',
    'use_regression',
    'use_contrastive',
    'use_infonce'
]


def compute_embedding_distance(contrasts, pred):
    """
    Computes distance between embeddings of a contrast (in the form [negative1, negative2, ...., positive]) and a prediction.
    :param contrasts: Tensor of size (batch-size, contrast-size, vector size); note that contrast-size is equivalent to number of negatives plus 1 for the positive
    :param pred: Tensor of size (batch-size, vector size)
    """

    use_infonce_distance = True
    if use_infonce_distance:
        positive = contrasts[:, -1]
        negatives = contrasts[:, :-1]
        batch_size, contrast_size, vec_size = negatives.shape
        dv = contrasts.device
        pred = pred.view(batch_size, 1, vec_size).to(dv)
        positive = positive.view(batch_size, 1, vec_size).to(dv)
        negatives = negatives.to(dv)

        # NOTE: we need to transpose prediction to perform MatMul
        logits_negatives = torch.bmm(negatives, pred.view(batch_size, vec_size, 1)).view(batch_size, contrast_size)
        logits_positive = torch.bmm(positive.view(batch_size, 1, vec_size), pred.view(batch_size, vec_size, 1)).view(batch_size, 1)

        logits = torch.cat((logits_negatives, logits_positive), dim=-1)

        # No need to randomize displacements, because just evaluating

        # Instead, need subtraction, because the distance in the parent function is supposed to be minimized
        return 1 - F.softmax(logits, dim=-1)

    else:
        bs = contrasts.shape[0]
        contrast_size = contrasts.shape[1]
        assert len(pred.shape) == 2 and pred.shape[0] == bs, f"Prediction {pred.shape}, Contrast {contrasts.shape}"
        assert len(contrasts.shape) == 3 and (contrasts.shape[-1] == pred.shape[-1]), f"Prediction {pred.shape}, Contrast {contrasts.shape}"

        return torch.cdist(contrasts, pred.unsqueeze(-2), p=2.0).view(bs, contrast_size).to('cpu')


def contrastive_train_step(model: AbstractVectorTransform, batch, alpha=1.0):
    _, pred, positive, negatives = model(
        batch['before'].to(model.device),
        batch['action'].to(model.device),
        batch['positive'].to(model.device),
        batch['negatives'].to(model.device)
    )
    contrasts = torch.cat((negatives, positive), dim=-2).to(model.device)
    distances = compute_embedding_distance(contrasts, pred)
    distances = distances ** 2
    batch_loss = (alpha + (distances[:, -1].unsqueeze(-1) - distances[:, :-1]))
    # batch_loss = batch_loss * batch['mask']  # annihilates padded elemnents in contrast
    loss = (batch_loss * (batch_loss >= 0).int()).sum(dim=-1).mean(dim=0)
    return loss


def infonce_train_step(model: AbstractVectorTransform, batch):
    _, pred, positive, negatives = model(
        batch['before'].to(model.device),
        batch['action'].to(model.device),
        batch['positive'].to(model.device),
        batch['negatives'].to(model.device)
    )

    batch_size, contrast_size, vec_size = negatives.shape
    dv = batch['negatives'].device
    pred = pred.view(batch_size, 1, vec_size).to(dv)
    positive = positive.view(batch_size, 1, vec_size).to(dv)
    negatives = negatives.to(dv)

    # NOTE: we need to transpose prediction to perform MatMul
    logits_negatives = torch.bmm(negatives, pred.view(batch_size, vec_size, 1)).view(batch_size, contrast_size)
    logits_positive = torch.bmm(positive.view(batch_size, 1, vec_size), pred.view(batch_size, vec_size, 1)).view(batch_size, 1)

    logits = torch.cat((logits_negatives, logits_positive), dim=-1)

    # 1. Compute random placement of positive for batch
    displacements = torch.stack([torch.randperm(contrast_size + 1) for _ in range(batch_size)], dim=0)
    pos_indices = torch.where(displacements == contrast_size)[-1]  # GT for this batch now is the position of the positive <--> (ex-) last index in the contrast

    batch_indices = torch.tensor([i for i in range(batch_size) for _ in range(contrast_size + 1)])  # help in randomizing contrast position

    reworked_logits = logits[batch_indices, displacements.view(1, -1)].view(logits.shape)

    # 2. Compute crossentropy with correct positive placement
    loss = F.cross_entropy(reworked_logits, pos_indices)

    return loss


def iterate(model: AbstractVectorTransform, dl, optimizer=None,
            loss_fn=None, mode='eval',
            contrastive=False, infonce=False,
            save_outputs=False):
    """
    Performs one epoch, chosen by the `mode` argument:
        * `train`
        * `eval` (validation using vector similarity as metric)
        * `eval-contrasts` (validation using contrast set task, in this case `dl` argument should be a dataset
        instead of a dataloader)

    Can be used also as method for evaluating models on the test set.

    By using the `save_outputs` parameter, it can be returned a DataFrame containing the whole set of predictions
    (only for the eval-contrasts mode).

    The parameters `contrastive` and `infonce` determine the type of loss used and cannot be true at the same time.
    """

    assert not (contrastive and infonce), "choose only one loss type"
    if mode == 'train':
        model.train()
        loss_history = []
        for batch in tqdm(dl, desc=mode.upper() + "..."):
            optimizer.zero_grad()
            if contrastive:
                loss = contrastive_train_step(model, batch)
            elif infonce:
                loss = infonce_train_step(model, batch)
            else:
                loss = model.process_batch(batch, loss_fn)
            loss.backward()
            optimizer.step()
            loss_history.append(loss.item())
        return torch.tensor(loss_history)
    elif mode == 'eval':
        model.eval()
        similarity_history = []
        with torch.no_grad():
            for batch in tqdm(dl, desc=mode.upper() + "..."):
                preds, gts = model.process_batch(batch)
                similarity_history.extend(torch.cosine_similarity(preds, gts, dim=-1).flatten().tolist())
        return torch.tensor(similarity_history).float().mean(dim=-1).item()
    elif mode == 'eval-contrasts':

        assert isinstance(dl, torch.utils.data.Dataset), "in this branch it is assumed that a Dataset object is passed"

        model.eval()
        accuracy_history = []

        if save_outputs:
            outs_df = []

        with torch.no_grad():
            # Here dl should be the bare dataset, not DataLoader
            pbar = tqdm(total=len(dl), desc=mode.upper() + "...")
            for i in range(len(dl)):
                smp = dl[i]  # dict containing 'before', 'action', 'positive', 'negatives', 'neg_actions'

                before = smp['before'].unsqueeze(0).to(model.device)
                action = torch.tensor(smp['action'], dtype=torch.long).unsqueeze(0).to(model.device)
                positive = smp['positive'].unsqueeze(0).to(model.device)
                negatives = torch.stack(smp['negatives'], dim=0).unsqueeze(0).to(model.device)

                # Performs actual prediction
                before, pred, positive, negatives = model(before, action, positive, negatives)

                before = before.to('cpu')
                pred = pred.to('cpu')
                positive = positive.to('cpu')
                negatives = negatives.to('cpu')

                contrast = torch.cat((negatives, positive.unsqueeze(1)), dim=1)  # Concatenates in the 'contrast' dimension --> final shape (1, contrast+1, vec_size)

                distances = compute_embedding_distance(contrast, pred)

                pred_idx = torch.argmin(distances, dim=-1).item()
                correct = (pred_idx == (distances.shape[-1] - 1))  # correct if the positive (last one) is the nearest
                accuracy_history.append(int(correct))

                if save_outputs:

                    # gets action names
                    action_choices = smp['neg_actions'] + [smp['action']]
                    pred_ac = dl.dataset.ids_to_actions[action_choices[pred_idx]]
                    gt_ac = dl.dataset.ids_to_actions[smp['action']]

                    # computes visual distractor (most similar to before image)
                    before_distances = compute_embedding_distance(contrast, before)
                    vis_distractor_idx = torch.argmin(before_distances, dim=-1).item()
                    vis_distractor_ac = dl.dataset.ids_to_actions[action_choices[vis_distractor_idx]]
                    is_vis_distractor = 'none' if (vis_distractor_idx == -1) else ('yes' if pred_idx == vis_distractor_idx else 'no')   # exclude cases where v.d. is actually correct

                    # gets language distractor (for size 4 contrasts)
                    lang_distractor_idx = 2
                    is_lang_distractor = 'none' if (dl.dataset.cs_type == 'CS3') else ('yes' if pred_idx == lang_distractor_idx else 'no')

                    outs_df.append({
                        'after_path': smp['pos_path'],
                        'pred_id': action_choices[pred_idx],
                        'pred_action': pred_ac,
                        'gt_id': action_choices[-1],
                        'gt_action': gt_ac,
                        'visual_distractor_action': vis_distractor_ac,
                        'object_name': smp['object'],
                        'pred_visual_distractor': is_vis_distractor,
                        'pred_language_distractor': is_lang_distractor
                    })


                pbar.update()
            pbar.close()

        if save_outputs:
            outs_df = pandas.DataFrame(data=outs_df)
            return torch.tensor(accuracy_history).float().mean(dim=-1).item(), outs_df
        else:
            return torch.tensor(accuracy_history).float().mean(dim=-1).item(), None


def train(*args, contrastive=False, infonce=False):
    return round(iterate(*args, mode='train', contrastive=contrastive, infonce=infonce).mean(dim=-1).item(), 4)


def evaluate(model, dl, use_contrasts=False, save_outputs=False):
    sim = iterate(model, dl, mode='eval')
    acc = None
    outs_df = None
    if use_contrasts:
        acc, outs_df = iterate(model, dl.dataset, mode='eval-contrasts', save_outputs=save_outputs)
    return round(sim, 4), round(acc, 4), outs_df


def baselines_evaluation(test_set):
    """
    Performs one epoch of baselines (random, similarity) evaluation. Not used in the final implementation,
    which instead relies on the saved outputs' DataFrame (see the `iterate` method).
    """

    random_acc = []
    similarity_acc = []

    pbar = tqdm(total=len(test_set), desc="Evaluating baselines...")
    for i in range(len(test_set)):
        smp = test_set[i]  # dict containing 'before', 'action', 'positive', 'negatives', 'neg_actions'
        if len(smp['negatives']) > 0:  # remove contrast with only 1 positive
            before_vec = smp['before'].unsqueeze(0)
            # action = torch.tensor(smp['action'], dtype=torch.long).unsqueeze(0)  # not needed
            pred = before_vec

            contrast = torch.stack(smp['negatives'] + [smp['positive']], dim=0).unsqueeze(0)  # add fake batch dimension
            distances = compute_embedding_distance(contrast, pred)
            # distances = torch.cosine_similarity(contrast, pred, dim=-1)

            sim_correct = torch.argmin(distances, dim=-1).item() == (distances.shape[-1] - 1)  # correct if the positive (last one) is the nearest
            similarity_acc.append(int(sim_correct))

            random_acc.append(distances.shape[-1])

        pbar.update()
    pbar.close()

    random_acc = 1 / (sum(random_acc) / len(random_acc))

    return {
        'random': round(random_acc, 4),
        'similarity': round(torch.tensor(similarity_acc).float().mean(dim=-1).item(), 4)
    }


def get_savepath(st_path, args, create_dir=True, create_topdir=True):
    """
    Computes a savepath for the experiment by creating also top and intermediate directories if needed.
    """
    if not create_topdir:
        return None

    if isinstance(st_path, str):
        st_path = Path(st_path)

    os.makedirs(str(st_path), exist_ok=True)

    if create_dir:
        names = [name for name in os.listdir(st_path) if len(name.split("_")) >= 2]
        save_name = f"{max([int(name.split('_')[-2] if name.split('_')[-2].isdigit() else -1) for name in names], default=-1) + 1}_@{args.statistical_iterations}"
        save_path = st_path / save_name
        os.makedirs(save_path, exist_ok=True)
    else:
        save_path = st_path
    return save_path


def get_model(actions, vec_size, args, **kwargs):
    return _ALLOWED_TRANSFORM_MODULES[args.vect_model](actions, vec_size, **vars(args), **kwargs).to(args.device)


def run_training(args, fixed_dataset=None, save_results=True, plot=True, return_test_outs=False):
    """
    Runs training for a single model. Can be executed passing a fixed dataset to control experiments from the outside,
    otherwise initializes it independently.

    In case `return_test_outs` is set to `True`, it returns a DataFrame containing raw predictions for test set;
    see the `iterate` method (evaluate-contrasts mode) for more details on raw outputs.
    """
    pprint(vars(args))

    nr_epochs = int(args.epochs)
    if fixed_dataset is None:
        full_dataset, train_dl, valid_dl, test_dl = get_data(**vars(args), dataset_type='vect')
    else:
        full_dataset, train_dl, valid_dl, test_dl = fixed_dataset

    model = get_model(set(full_dataset.ids_to_actions.keys()), full_dataset.get_vec_size(), args)
    opt, sched = get_optimizer(args, model)
    loss_fn = torch.nn.MSELoss()  # ONLY USED IN CASE OF NOT CONTRASTIVE NOR INFONCE, here for compatibility reasons

    ep_loss_mean = []
    ep_sim_mean = []
    ep_acc_mean = []

    # Prepares path for saving
    if args.use_contrastive:
        args.vect_model = args.vect_model + '-contrastive'
    elif args.use_infonce:
        args.vect_model = args.vect_model + '-infonce'

    if not return_test_outs:
        # Executed independently thus should save in a separate directory
        pth = Path(args.save_path, "models", "+".join([args.vect_model, args.extractor_model]))
        pth = get_savepath(pth, args, create_dir=save_results)
        pth = Path(*pth.parts[:-1], args.hold_out_procedure + "_" + str(pth.parts[-1]))

    best_acc = -1.0
    for ep in range(nr_epochs):
        print("EPOCH", ep + 1, "/", nr_epochs)

        # Training
        ep_loss_mean.append(train(model, train_dl, opt, loss_fn, contrastive=args.use_contrastive, infonce=args.use_infonce))
        print("Train loss: ", ep_loss_mean[-1])

        # Validation
        ev_sim, ev_acc, _ = evaluate(model, valid_dl, use_contrasts=True)
        ep_sim_mean.append(ev_sim)
        print("Eval sim: ", ep_sim_mean[-1])
        ep_acc_mean.append(ev_acc)
        print("Eval acc: ", ep_acc_mean[-1])

        # # Saving best model according to validation accuracy
        # # (not used, since performances happened to be almost always monotonic)
        # if args.save_model:
        #     if ep_acc_mean[-1] > best_acc:
        #         best_acc = ep_acc_mean[-1]
        #         torch.save(model.state_dict(), pth / 'checkpoint.pth')

    # TESTING
    print()
    print("------ TEST ------")

    test_sim, test_acc, test_df = evaluate(model, test_dl, use_contrasts=True, save_outputs=True)

    print("Test sim: ", test_sim)
    print("Test acc: ", test_acc)

    if args.save_model:
        torch.save(model.state_dict(), str(pth) + '.pth')


    if save_results:
        # In case it is executed independently

        os.makedirs(pth, exist_ok=True)

        with open(pth / 'config.json', mode='wt') as f:
            json.dump(vars(args), f, indent=2)

        # Saving metrics
        pandas.DataFrame({
            'Train Loss': ep_loss_mean,
            'Cosine Similarity': ep_sim_mean,
            'Contrast Accuracy': ep_acc_mean
        }).to_csv(str(pth / 'metrics.csv'), index=False)

        if plot:
            # Plotting
            plt.figure(figsize=(15, 6))
            plt.suptitle(args.vect_model + f" ({args.extractor_model})")
            for idx, title, metric in zip([1, 2, 3], ['Train Loss', 'Cosine Similarity', 'Contrast Accuracy'], [ep_loss_mean, ep_sim_mean, ep_acc_mean]):
                plt.subplot(1, 3, idx)
                plt.title(title)
                plt.plot(metric)
                plt.xlabel('Epoch')
            plt.savefig(str(pth / 'metrics.png'))
            plt.close()

        return model
    else:
        if return_test_outs:
            return test_sim, test_acc, test_df

        # needed for statistical evaluation
        return test_sim, test_acc, full_dataset  # to register hold-out items


def exp_hold_out(args, return_test_outs=False):
    """
    Investigates model generalization capabilities by changing holding out procedure and running several tests,
    in order to build a statistical analysis with randomly sampled test set.
    At each iteration, a new dataset is initialized, and with it a new split of seen/unseen items
    (objects or scenes, depending on the procedure); within the 'seen' items are created a training and a validation set,
    while from the 'unseen' ones it is built the test set.
    Supposing contrasts of hard-negatives (i.e. same object, same scene) each contrast set appears wholly either in the
    seen or in the unseen split; instead, for train and validation samples from the same contrast may be put in different splits.
    """

    # Hard-coded params
    args.activation = 'relu'
    args.use_bn = True

    st_path = Path(args.save_path, "statistics", "hold-out")
    save_path = get_savepath(st_path, args, create_topdir=not return_test_outs)

    tested_procedures = ['structural', 'reversible', 'receptacle', 'surface', 'scene', 'action']
    nr_samples_per_it = {k: [] for k in tested_procedures}

    tested_vect_models = _ALLOWED_TRANSFORM_MODULES

    df = []
    for stat_it in range(args.statistical_iterations):
        for extractor_model in ['moca-rn', 'clip-rn']:
            for hold_out_procedure in tested_procedures:

                # needed both the following two to fix the dataset
                args.hold_out_procedure = hold_out_procedure
                args.extractor_model = extractor_model

                fixed_ds = get_data(**vars(args), dataset_type='vect')

                if not return_test_outs:
                    with open(save_path / 'items.txt', mode='at') as f:
                        f.write(str(fixed_ds[0].get_hold_out_items()) + "\n")

                nr_samples_per_it[hold_out_procedure].append(fixed_ds[0].get_nr_hold_out_samples())

                for vect_model in tested_vect_models:
                    args.vect_model = vect_model
                    sim, acc, ds = run_training(args, fixed_dataset=fixed_ds, save_results=False, plot=False, return_test_outs=return_test_outs)

                    # if outputs are not needed, proceed with standard dataframe of accuracy
                    if not return_test_outs:
                        df.append({
                            'hold_out_procedure': hold_out_procedure,
                            'extractor_model': extractor_model,
                            'vect_model': args.vect_model,
                            'similarity': sim,
                            'accuracy': acc
                        })
                    else:
                        # here ds --> dataframe containing all outputs from the current model
                        ds['hold_out_procedure'] = hold_out_procedure
                        ds['extractor_model'] = extractor_model
                        ds['vect_model'] = args.vect_model
                        ds['iteration'] = stat_it
                        df.append(ds)

                if not return_test_outs:
                    # Adds baselines for this dataset
                    bsl_acc = baselines_evaluation(fixed_ds[-1].dataset)
                    for k in bsl_acc:
                        df.append({
                            'hold_out_procedure': hold_out_procedure,
                            'extractor_model': extractor_model,
                            'vect_model': 'baseline-' + k,
                            'similarity': -1.0,
                            'accuracy': bsl_acc[k]
                        })

    if return_test_outs:
        return pandas.concat(df, ignore_index=True)

    # ### Following part executed only if experiment is executed alone ###
    with open(save_path / 'items.txt', mode='at') as f:
        f.write(str({
            'sizes': nr_samples_per_it,
            'averages': {k: ((sum(el) / len(el)) if len(el) > 0 else 0) for k, el in nr_samples_per_it.items()}
        }))

    with open(save_path / 'config.json', mode='at') as f:
        json.dump(vars(args), f, indent=2)

    df = pandas.DataFrame(df, columns=['extractor_model', 'vect_model', 'hold_out_procedure', 'similarity', 'accuracy'])
    df.to_csv(save_path / 'results.csv', index=False)

    seaborn.set(font_scale=1.5)

    # create column plot
    g = seaborn.catplot(x='extractor_model', y='accuracy', hue='vect_model', data=df[~df['vect_model'].isin({'baseline-random', 'baseline-similarity'})], col="hold_out_procedure", kind="bar", height=10, aspect=.75)

    # random baseline
    [ax.axhline(
        df[(df['vect_model'] == 'baseline-random') & (df['hold_out_procedure'] == ax.get_title().split()[-1])]['accuracy'].mean(),
        linestyle='--', color='gray', alpha=0.7)
        for el in g.axes for ax in el]

    # similarity baseline
    [ax.axhline(
        df[(df['vect_model'] == 'baseline-similarity') & (df['hold_out_procedure'] == ax.get_title().split()[-1])]['accuracy'].mean(),
        linestyle='--', color='orange', alpha=0.7)
        for el in g.axes for ax in el]

    for ax, title in zip(g.axes.flatten(), ['samples', 'objects', 'scenes']):
        ax.set_title("Hold-out " + title)

    plt.legend(loc='upper right', facecolor='white')

    g.savefig(save_path / 'stats.png')

    g = seaborn.catplot(hue='extractor_model', x='vect_model', y='accuracy', data=df, kind='swarm', col='hold_out_procedure')
    g.savefig(save_path / 'observations.png')

    return df


def exp_fcn_hyperparams(args):
    """
    Computes all hyperparameters combinations for grid-searching the best Concat-Multi training configuration.
    """

    args.vect_model = 'Concat-Multi'
    args.statistical_iterations = 1

    hyperparam_options = {
        'batch_size': [256],
        'learning_rate': [1e-2, 1e-3, 1e-5],
        'optimizer': ['adam', 'adamw'],
        'use_bn': [True, False],
        'dropout_p': [0.0, 0.7],
        'extractor_model': ['moca-rn'],
        'activation': allowed_activations
    }
    df = []

    from itertools import product
    print("TOTAL CONFIGS:", len(list(product(*list(hyperparam_options.values())))), f"(* {args.statistical_iterations} iterations)")
    for i in range(args.statistical_iterations):
        for cfg in product(*list(hyperparam_options.values())):
            cfg = {k: cfg[i] for i, k in enumerate(hyperparam_options.keys())}
            args = vars(args)
            args.update(cfg)
            args = Namespace(**args)
            args.vect_model = 'Concat-Multi'  # needed for consistency with contrastive setup
            sim, acc, ds = run_training(args, save_results=False, plot=False)
            df.append({**cfg, **{'similarity': sim, 'accuracy': acc}})
    df = pandas.DataFrame(df, columns=list(hyperparam_options.keys()) + ['similarity', 'accuracy'])

    st_path = Path(args.save_path, "statistics", "hyperparams")
    save_name = get_savepath(st_path, args)
    df.to_csv(save_name / "result.csv", index=False)
    with open(save_name / 'best.txt', mode='wt') as f:
        bst = df.iloc[df['accuracy'].idxmax()].to_dict()
        pprint(bst, f)
    return df


def exp_regression(args):
    """
    Tests AM models where matrices are learned by least squares regression.
    """

    def run_train_regression(args, fixed_dataset=None, save_results=False):

        # Prepares path for saving
        pth = Path(args.save_path, "regression", "+".join([args.vect_model, args.extractor_model]))
        pth = get_savepath(pth, args)

        if fixed_dataset is not None:
            full_dataset, reg_matrices, test_dl = fixed_dataset
        else:
            full_dataset, reg_matrices, test_dl = get_data(**vars(args), dataset_type='vect', use_regression=True)

        model = get_model(set(full_dataset.ids_to_actions.keys()), full_dataset.get_vec_size(), args)
        model.regression_init(reg_matrices, regression_type='torch')

        print(f"------ TEST ------")
        test_sim, test_acc, test_df = evaluate(model, test_dl, use_contrasts=True)
        print("Test sim: ", test_sim)
        print("Test acc: ", test_acc)
        print()

        if save_results:
            os.makedirs(pth, exist_ok=True)

            with open(pth / 'config.json', mode='wt') as f:
                json.dump(vars(args), f, indent=2)

            pandas.DataFrame({
                'similarity': test_sim,
                'accuracy': test_acc,
            }).to_csv(str(pth / 'metrics.csv'), index=False)

        return test_sim, test_acc, full_dataset  # to register hold-out items

    tested_procedures = ['object_name', 'scene']

    args.vect_model = 'Action-Matrix'
    args.use_regression = True

    st_path = Path(args.save_path, "regression")

    save_path = get_savepath(st_path, args)

    nr_samples_per_it = {k: [] for k in tested_procedures}
    df = []
    for stat_it in range(args.statistical_iterations):
        for extractor_model in ['moca-rn', 'clip-rn']:
            for hold_out_procedure in tested_procedures:
                args.hold_out_procedure = hold_out_procedure
                args.extractor_model = extractor_model

                pprint({
                    'extractor_model': args.extractor_model,
                    'hold-out': args.hold_out_procedure
                })

                # needed both to fix the dataset
                fixed_ds = get_data(**vars(args), dataset_type='vect')

                with open(save_path / 'items.txt', mode='at') as f:
                    f.write(str(fixed_ds[0].get_hold_out_items()) + "\n")

                nr_samples_per_it[hold_out_procedure].append(fixed_ds[0].get_nr_hold_out_samples())

                sim, acc, ds = run_train_regression(args, fixed_dataset=fixed_ds, save_results=False)
                df.append({
                    'hold_out_procedure': hold_out_procedure,
                    'extractor_model': extractor_model,
                    'similarity': sim,
                    'accuracy': acc,
                    'vect_model': args.vect_model
                })

    with open(save_path / 'items.txt', mode='at') as f:
        f.write(str({
            'sizes': nr_samples_per_it,
            'averages': {k: ((sum(el) / len(el)) if len(el) > 0 else 0) for k, el in nr_samples_per_it.items()}
        }))

    df = pandas.DataFrame(df, columns=['extractor_model', 'vect_model', 'hold_out_procedure', 'similarity', 'accuracy'])
    df.to_csv(save_path / 'results.csv', index=False)

    seaborn.set(font_scale=2)
    g = seaborn.catplot(x='vect_model', y='accuracy', hue='extractor_model', data=df, col="hold_out_procedure", kind="bar", height=10, aspect=.75)
    g.savefig(save_path / 'stats.png')

    # # Code for showing individual datapoints
    # g = seaborn.catplot(hue='extractor_model', x='vect_model', y='accuracy', data=df, kind='swarm', col='hold_out_procedure')
    # g.savefig(save_path / 'observations.png')


def exp_cs_size(args):
    """Initially planned to run experiments with different contrast-set-sizes (3, 4). Now only used
    to run hold-out experiments easily and with a better plot."""

    save_path = get_savepath(args.save_path, args)
    outs = []
    for cs_type in ['CS4']:
        args.cs_type = cs_type
        res_df = exp_hold_out(args, return_test_outs=True)
        res_df['contrast_type'] = cs_type
        outs.append(res_df)

    with open(save_path / 'config.json', mode='wt') as fp:
        json.dump(vars(args), fp, indent=2)

    full_df = pandas.concat(outs, ignore_index=True)

    full_df.to_csv(save_path / 'outputs.csv', index=False)

    g = plots.training(save_path / 'outputs.csv', show=False)

    g.savefig(save_path / 'results.png')


def exp_nearest_neighbors(args, k=None):
    """Computes ranking of all test images (positives and negatives) for each predicted vector
    (= transformed before vector), by using the cosine similarity as a measure of distance.
    Executes for all models that can be found in subfolders, thus it should be executed carefully when
    there are more than 1 statistical iteration."""

    def run_nearest_neighbors(args, k=None):
        """
        Given a path, uses the (first) model present in the path to save nearest-neighbor images.
        """

        pairwise_distance_fn = lambda prd, other: (1 - torch.cosine_similarity(prd, other)) / 2

        neighbors_path = Path(Path(args.load_path).parent, Path(args.load_path).parts[-1].split("@")[0] + "neighbors")

        if not os.path.exists(neighbors_path):
            os.mkdir(neighbors_path)

            args.extractor_model = Path(args.load_path).parent.parts[-1].split("+")[-1]
            ho_proc = Path(args.load_path).parts[-1].split("@")[0]
            ho_proc = "_".join(ho_proc.split("_")[:-2])  # needed to merge object_name
            args.hold_out_procedure = ho_proc
            full_dataset, _, _, test_dl = get_data(**vars(args), dataset_type='vect')

            model = load_model(args.load_path, actions=set(full_dataset.ids_to_actions.keys()))
            model.eval()

            neighbors_pool = {}
            predictions = {}

            ds = test_dl.dataset

            with torch.no_grad():
                for i in tqdm(range(len(ds)), desc='Extracting vectors...'):
                    smp = ds[i]
                    before = smp['before'].unsqueeze(0).to(model.device)
                    action = torch.tensor(smp['action'], dtype=torch.long).unsqueeze(0).to(model.device)
                    positive = smp['positive'].unsqueeze(0).to(model.device)
                    negatives = torch.stack(smp['negatives'], dim=0).unsqueeze(0).to(model.device)

                    _, pred, positive, negatives = model(before, action, positive, negatives)

                    pred = pred.to('cpu')
                    positive = positive.to('cpu')
                    negatives = negatives.to('cpu').squeeze()

                    if neighbors_pool.get(smp['pos_path'], None) is None:
                        neighbors_pool[smp['pos_path']] = positive

                    for i, npath in enumerate(smp['neg_paths']):
                        if neighbors_pool.get(npath, None) is None:
                            neighbors_pool[npath] = negatives[i]

                    predictions[smp['pos_path']] = {
                        'vec': pred,
                        'before': smp['before_path']
                    }

            with open(Path(neighbors_path, 'pred.pkl'), mode='wb') as fp:
                pickle.dump(predictions, fp)

            with open(Path(neighbors_path, 'neighbors.pkl'), mode='wb') as fp:
                pickle.dump(neighbors_pool, fp)
        else:
            with open(Path(neighbors_path, 'pred.pkl'), mode='rb') as fp:
                predictions = pickle.load(fp)

            with open(Path(neighbors_path, 'neighbors.pkl'), mode='rb') as fp:
                neighbors_pool = pickle.load(fp)

        if not os.path.exists(Path(neighbors_path, 'ranking.csv')):
            scores_df = []
            for pred_pth in tqdm(predictions, desc='Computing distances...'):
                # # Excluding ground truth path
                # scores = [(pth, pairwise_distance_fn(predictions[pred_pth]['vec'], neighbors_pool[pth])) for pth in
                #           neighbors_pool if pth != pred_pth]
                scores = [(pth, pairwise_distance_fn(predictions[pred_pth]['vec'], neighbors_pool[pth])) for pth in
                          neighbors_pool]
                ranked_neighbors = [tp[0] for tp in sorted(scores, key=lambda el: el[1], reverse=False)]
                if k is not None and k > 0:
                    ranked_neighbors = ranked_neighbors[:k]

                scores_df.append({'pth': pred_pth, 'neighbors': ranked_neighbors})

            scores_df = pandas.DataFrame(scores_df)
            scores_df.to_csv(Path(neighbors_path, 'ranking.csv'), index=False)

            return scores_df
        else:
            return pandas.read_csv(Path(neighbors_path, 'ranking.csv'))

    # ############## Terminates wrapped function ###############

    if not os.path.exists(Path(args.load_path, 'stats_results.csv')):
        for r, d, fnames in os.walk(args.load_path):
            for fname in fnames:
                if fname.endswith('.pth'):
                    tmpargs = Namespace(**vars(args))
                    tmpargs.load_path = os.path.join(r, fname)
                    print(f"------------ NN Experiment for {str(Path(tmpargs.load_path).parent.parts[-1])} ------------")
                    run_nearest_neighbors(tmpargs, k)

    if not os.path.exists(Path(Path(args.load_path).parent, 'images-with-before')):
        plots.nearest_samples(args.load_path)

    metrics = plots.stats_nearest_samples(args.load_path, savefig=True)

    return metrics


def exp_neighbor_mds(args):
    """
    Run an MDS analysis of after-vectors and transformed-vectors by projecting them in a 2D space for visualization.
    Works only on models saved and pretrained in the specified folder, and only on those folders containing '_neighbors'
    in their name, i.e. those where the method `exp_nearest_neighbors` has already been executed.
    """
    import shutil
    from sklearn.manifold import MDS
    from visual_features.data import load_and_rework_csv
    from visual_features.data import __default_dataset_fname__ as dataset_fname

    ann_df = load_and_rework_csv(args.data_path, dataset_fname)

    path2action = dict([
        *list(zip(ann_df.after_image_path, ann_df.action_name)),
        *[tp for c in [0,1,2] for tp in zip(ann_df[f"distractor{c}_path"], ann_df[f"distractor{c}_action_name"])]
    ])  # should NOT be the case that 2 same paths have different actions when in different columns...

    def get_action(pth):
        return path2action[pth]

    def run_mds(args):
        """
        Runs MDS for a single model. Stores results in the same directory as the model `.pth` file.
        """

        tosave_pth = args.load_path

        with open(Path(args.load_path, 'neighbors.pkl'), mode='rb') as fp:
            vecs = pickle.load(fp)
        with open(Path(args.load_path, 'pred.pkl'), mode='rb') as fp:
            preds = pickle.load(fp)

        reduction = MDS(
            n_components=2,
            metric=True,
            dissimilarity='euclidean'  # TODO change this to precomputed?
        )

        # Needed to separate vectors from predictions after projection
        separator = len(vecs)

        if not os.path.exists(Path(args.load_path, 'mds.pkl')):
            v = numpy.vstack([
                *[el.squeeze().numpy() for el in vecs.values()],
                *[el['vec'].squeeze().numpy() for el in preds.values()]
            ]).astype(numpy.float64)

            print('Transforming...')
            overall_vec = reduction.fit_transform(v)
            print("Done!")

            with open(Path(args.load_path, 'mds.pkl'), mode='wb') as fp:
                pickle.dump(overall_vec, fp)
        else:
            with open(Path(args.load_path, 'mds.pkl'), mode='rb') as fp:
                overall_vec = pickle.load(fp)
            print(f"Loaded MDS from {str(Path(args.load_path, 'mds.pkl'))}")

        projected_vecs = overall_vec[:separator]
        projected_preds = overall_vec[separator:]

        projected_vecs = {
            'vec': projected_vecs,
            'category': [get_action(pth) for pth in vecs]
        }
        projected_preds = {
            'vec': projected_preds,
            'category': [get_action(pth) for pth in preds]
        }

        # Plot mds points with colors associated
        fig = plots.mds(projected_vecs, projected_preds, plots.shorten_model_name(str(Path(*Path(args.load_path).parts[-2:]))), show=False)
        fig.savefig(Path(tosave_pth) / 'mds-actions.png')
    # #################### End wrap ####################

    # Performs MDS for each saved model in folder
    for r, dnames, fs in os.walk(args.load_path):
        for dname in dnames:
            if "_neighbors" in dname:
                tmpargs = Namespace(**vars(args))
                tmpargs.load_path = os.path.join(r, dname)
                print(f"------------ MDS Experiment for {str(Path(*Path(tmpargs.load_path).parts[-2:]))} ------------")
                run_mds(tmpargs)

    # Copies all images in the same directory for prettier visualization
    os.mkdir(Path(args.load_path, 'mds-images'))
    for r, dnames, fs in os.walk(args.load_path):
        if r.endswith("_neighbors"):
            new_fname = plots.shorten_model_name(str(Path(*Path(r.replace("_neighbors", "")).parts[-2:])))
            new_fpath = Path(args.load_path, 'mds-images', new_fname + '.png')
            shutil.copyfile(str(Path(r, 'mds-actions.png')), str(new_fpath))


def exp_action_crossval(args):
    """
    Computes action cross-validation by holding out one different object for several actions and testing models
    on each of them. Reports performance
    """

    def generate_combinations(annotation_dataframe, max_combs_per_action=5):
        """Generates combinations of each action with object-action pairings containing all possible objects for that action.
        In order to maintain consistency, for each action, every combination sets as fixed all the object-action
        associations except for the current action that iterates several objects.
        Action-object pairings are not shared across actions, meaning that to the same non-current action
        can correspond different objects when switching across current-actions."""
        from random import choice

        # Retrieves all combinations
        existent_combinations = defaultdict(set)
        for a, o in zip(annotation_dataframe.action_name, annotation_dataframe.object_name):
            existent_combinations[a].update((o,))

        # sort actions by name so that coherence is preserved
        existent_combinations = {k: list(existent_combinations[k]) for k in sorted(existent_combinations)}

        res = {}

        for ac in tqdm(existent_combinations, desc='generating combinations...'):
            comb = None
            interrupt = False
            ac_cfgs = []
            while not interrupt:
                # assigns action to object, then checks length if there has been repeated assignment
                comb = {choice(existent_combinations[cmb_ac]): cmb_ac
                        for cmb_ac in existent_combinations if cmb_ac != ac}

                # If first, should generate avoiding superpositions
                # Otherwise, in case:
                #  - curr action A
                #  - other action B
                #  - current object o1 belongs to A but is assigned to B
                # if exists an object shared by the two actions (which will be used in some configuration for A)
                # then chooses it for B, thus "swapping" them, otherwise chooses a random object for B

                if len(comb) != (len(existent_combinations) - 1):
                    pass  # completely invalid configuration, multiple objects for the same action
                else:
                    for obj in existent_combinations[ac]:
                        tmp = comb.copy()
                        if comb.get(obj, None) is not None:
                            other_ac = comb[obj]
                            possible_objs = set(existent_combinations[other_ac]).difference({k for k in comb})
                            if len(possible_objs) == 0:
                                # no alternative objects for this concurrent action --> should rebuild the combination
                                ac_cfgs = []
                                break

                            shared_objs = possible_objs.intersection(set(existent_combinations[ac]))
                            if len(shared_objs) > 0:
                                # chooses another object from the shared set, excluding the current one
                                possible_objs = shared_objs.difference({obj})
                            tmp[choice(list(possible_objs))] = other_ac

                        tmp[obj] = ac
                        tmp = {v: k for k, v in tmp.items()}
                        tmp = {k: tmp[k] for k in sorted(tmp)}
                        ac_cfgs.append(list(tmp.items()))

                if len(ac_cfgs) == len(existent_combinations[ac]):
                    # generated all necessary configs
                    interrupt = True

            res[ac] = ac_cfgs

        res = {ac: res[ac][:max_combs_per_action] for ac in res}
        return res

    # Hard-coded params
    args.activation = 'relu'
    args.use_bn = True

    save_path = get_savepath(args.save_path, args, create_topdir=True, create_dir=False)

    nr_samples_per_it = []  # saved for later computations

    tested_vect_models = _ALLOWED_TRANSFORM_MODULES

    df = []

    tosave_combinations = {'moca-rn': None,
                           'clip-rn': None}  # lists of combinations used for each model saved in .json

    for extractor_model in ['moca-rn', 'clip-rn']:

        # needed both the following two to fix the dataset
        args.hold_out_procedure = 'action'
        args.extractor_model = extractor_model

        fixed_ds = get_data(**vars(args), dataset_type='vect')

        annotation_df = fixed_ds[0].get_annotation()  # seen/unseen split does not matter, only needed for action-objects associations
        test_combinations = generate_combinations(annotation_df)
        tosave_combinations[extractor_model] = test_combinations

        for i_ac, curr_action in enumerate(test_combinations):  # select action to cross-validate with its objects configurations
            for i_comb, current_test_combination in enumerate(test_combinations[curr_action]):
                print()
                print(f"~~~~~~~~~~~~~~~~~ {extractor_model.upper()}, Action {i_ac + 1}/{len(test_combinations)}, Comb {i_comb + 1}/{len(test_combinations[curr_action])} ~~~~~~~~~~~~~~~~~")
                print()

                for stat_it in range(args.statistical_iterations):

                    # Change action split with the current action-object combination
                    # (place initialization here so that each time training & validation are randomized)
                    fixed_ds = fixed_ds[0].change_action_split(current_test_combination, args.batch_size)

                    nr_samples_per_it.append(fixed_ds[0].get_nr_hold_out_samples())

                    for vect_model in tested_vect_models:
                        args.vect_model = vect_model
                        _, _, ds = run_training(args, fixed_dataset=fixed_ds, save_results=False, plot=False,
                                                return_test_outs=True)

                        # ds = dataframe containing all outputs from the current model
                        ds['hold_out_procedure'] = 'action'
                        ds['extractor_model'] = extractor_model
                        ds['vect_model'] = args.vect_model
                        ds['iteration'] = stat_it
                        ds['tested_object'] = [_obj for _ac, _obj in current_test_combination if _ac == curr_action][0]
                        ds['tested_action'] = curr_action
                        df.append(ds)

    with open(save_path / 'action-object-combinations.json', mode='wt') as fp:
        json.dump(tosave_combinations, fp, indent=2)

    outs = pandas.concat(df, ignore_index=True)
    outspth = save_path / 'outputs.csv'
    outs.to_csv(outspth, index=False)

    g = plots.action_crossval(outspth, show=False, savefig=True)
    # g.savefig(save_path / 'results.png')

    return outs


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--exp_hold_out', action='store_true')
    parser.add_argument('--exp_fcn_hyperparams', action='store_true')
    parser.add_argument('--exp_regression', action='store_true')
    parser.add_argument('--exp_cs_size', action='store_true')
    parser.add_argument('--exp_nearest_neighbors', action='store_true')
    parser.add_argument('--exp_action_crossval', action='store_true')
    parser.add_argument('--exp_neighbor_mds', action='store_true')

    parser = setup_argparser(__default_train_config__, parser)
    args = parser.parse_args()

    if args.conditioning_method == 'none':
        tmp = args.conditioning_method
        while tmp == 'none':
            tmp = input(f"Please select an allowed conditioning method within {allowed_conditioning_methods}: ")
            tmp = tmp if tmp in allowed_conditioning_methods else 'none'
        args.conditioning_method = tmp

    if args.train:
        res = run_training(args)
    elif args.exp_fcn_hyperparams:
        res = exp_fcn_hyperparams(args)
    elif args.exp_hold_out:
        res = exp_hold_out(args)
    elif args.exp_regression:
        res = exp_regression(args)
    elif args.exp_cs_size:
        res = exp_cs_size(args)
    elif args.exp_nearest_neighbors:
        res = exp_nearest_neighbors(args)
    elif args.exp_action_crossval:
        res = exp_action_crossval(args)
    elif args.exp_neighbor_mds:
        res = exp_neighbor_mds(args)
    else:
        pass
        # # Place for debug scripts
        # for r, d, fnames in os.walk('./experiment-nearest-neighbors/models'):
        #     if r.endswith("_neighbors"):
        #         print(r)
        #         with open(os.path.join(r, 'neighbors.pkl'), mode='rb') as fp:
        #             n = pickle.load(fp)
        #         with open(os.path.join(r, 'pred.pkl'), mode='rb') as fp:
        #             p = pickle.load(fp)
        #         with open(os.path.join(r, 'mds.pkl'), mode='rb') as fp:
        #             mds = pickle.load(fp)
        #         print(len(n), " | ", len(p), " | ", len(mds), f"({len(n) + len(p)})", sep='     ')
        #         del n
        #         del p
        #         del mds
