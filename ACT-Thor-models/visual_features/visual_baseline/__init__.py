import os
import torch
import torch.nn as nn
import torchvision
import clip
import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from pprint import pprint
from argparse import ArgumentParser
from collections import OrderedDict
from PIL import Image

import sys
sys.path.extend(['.', '..'])

from visual_features.data import ActionDataset, \
    convert_action, \
    NEGATIVE_SAMPLING_METHODS as negsamp_choices

# CONSTANTS

SAVE_ROOT_DIR = "extracted"
tmp = [os.path.join(root, d) for root, dirs, _ in os.walk('.') for d in dirs if d == 'clip_models']
_clip_download_root = 'clip_models' if len(tmp) == 0 else tmp[0]
tmp = [os.path.join(root, d) for root, dirs, _ in os.walk('.') for d in dirs if d == 'moca_models']
_moca_download_root = 'moca_models' if len(tmp) == 0 else tmp[0]

MODEL_NAMES_MAP = {
    'clip-rn': "RN50",
    # 'clip-vit': "ViT-B/32",
    'moca-rn': 'moca-rn',
    'imagenet-rn': 'imagenet-rn',
    'untrained-rn': 'untrained-rn',
    'untrained-rn-zero-init': 'untrained-rn-zero-init'
}

ALLOWED_MODELS = list(MODEL_NAMES_MAP.keys()) + ['all']

DISTANCE_NAMES_MAP = {
    'cosine': lambda a, b: (1 - torch.cosine_similarity(a.float(), b.float(), dim=-1)) / 2,
    'euclidean': lambda a, b: torch.cdist(a.unsqueeze(0).float(), b.unsqueeze(0).float(), p=2.0)
}


# UTILITIES

class MOCAVisEnc(nn.Module):
    """
    visual encoder from MOCA from https://github.com/gistvision/moca/blob/2c2f8fdfe6521be1c731fdd915b1e0787b8db7cd/models/nn/vnn.py#L36
    """

    def __init__(self, dframe=147):
        super(MOCAVisEnc, self).__init__()
        self.dframe = dframe

        self.flattened_size = 64*7*7

        self.conv1 = nn.Sequential(*[
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU()
        ])
        self.conv2 = nn.Sequential(*[
            nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU()
        ])
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattened_size, self.dframe)
        )

    def forward(self, x):
        x = self.conv1(x)

        x = self.conv2(x)

        # x = x.view(-1, self.flattened_size)
        x = self.fc(x)

        return x


def load_model_and_transform(args, keep_pooling=False, add_flatten=True):
    model = None
    transform = None
    to_add = []
    if args.model_name == 'clip-rn':
        model, transform = clip.load(MODEL_NAMES_MAP[args.model_name], device=args.device, download_root=_clip_download_root)
        model = model.visual
        to_add = list(model.children())

        # NOTE: when using unfrozen CLIP, usually pooling is excluded because adds 4 fully-connected layers for Attentional Pooling
        # (differently from ImageNet, where the pooling is a simple AveragePool module)
        if not keep_pooling:
            to_add = to_add[:-1]
    elif args.model_name == 'moca-rn':
        rn = torchvision.models.resnet18(pretrained=True)
        conv_head = MOCAVisEnc()

        # Rework dict to match the current structure
        sd = {k.replace('dec.vis_encoder.', ''): p for k, p in torch.load(os.path.join(_moca_download_root, 'pretrained.pth'))['model'].items() if k.startswith('dec.vis_encoder')}
        sd = {k.replace('conv1.', 'conv1.0.'): p for k, p in sd.items()}
        sd = {k.replace('bn1.', 'conv1.1.'): p for k, p in sd.items()}
        sd = {k.replace('conv2.', 'conv2.0.'): p for k, p in sd.items()}
        sd = {k.replace('bn2.', 'conv2.1.'): p for k, p in sd.items()}
        sd = {k.replace('fc.', 'fc.1.'): p for k, p in sd.items()}  # NOTE: MOCA is the only one that actually has a final fully-connected, therefore theoretically does not need flattening
        conv_head.load_state_dict(OrderedDict(sd))

        if keep_pooling:
            # Do nothing, otherwise incompatible with convolutional head
            print("MOCA initialization: skipping pooling preservation for compatibility with convolutional encoder...")
        to_add = list(rn.children())[:-2] + list(conv_head.children())
    elif args.model_name == 'imagenet-rn':
        model = torchvision.models.resnet50(pretrained=True).to(args.device)

        to_add = list(model.children())[:-1]  # excluding fc layer
        if not keep_pooling:
            to_add = to_add[:-1]

    elif args.model_name.startswith('untrained-rn'):
        model = torchvision.models.resnet50(pretrained=False).to(args.device)

        to_add = list(model.children())[:-1]  # excluding fc layer
        if not keep_pooling:
            to_add = to_add[:-1]

        if args.model_name.endswith('zero-init'):
            for name, param in model.named_parameters():
                param.requires_grad_(False)
                param.zero_()
                param.requires_grad_(True)
    else:
        raise ValueError(f'unsupported model ({args.model_name})')

    if transform is None:
        # default transformation for torchvision models
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(224),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    if add_flatten:
        to_add.append(nn.Flatten())

    model = nn.Sequential(*to_add).to(args.device)
    model.device = args.device
    return model.float(), transform


def extract_features(args):
    model, preprocess = clip.load(MODEL_NAMES_MAP[args.model_name], args.device)

    with torch.no_grad():
        for root, d, files in os.walk(args.image_dir):
            to_parse = []
            for f in files:
                if f.endswith('.png'):
                    to_parse.append(f)
            for b in to_parse[::args.batch_size]:
                imgb = torch.tensor([preprocess(Image.open(os.path.join(root, d, img))) for img in b]).to(args.device)
                feats = model.encode_image(imgb)
                for i, name in enumerate(b):
                    torch.save(feats[i], os.path.join(root, d, name) + ".pth")


def visualize(args, res):
    c_acc = res['matrices']
    cmap = sns.color_palette('YlGnBu')
    fig = plt.figure(figsize=(12, 11))
    axs = fig.subplots(len(c_acc) // 2, 2)
    plt.subplots_adjust(left=0.074, bottom=0.06, right=0.94, top=0.914, wspace=0.207, hspace=0.867)
    plt.suptitle(f"Confusions by action set for {args.model_name}")
    for i, k in enumerate(c_acc):
        action_set = list(c_acc[k]['action_idx'].keys())
        action_set_names = [convert_action(actionid) for actionid in action_set]

        # Generates heatmap
        m = torch.tensor(c_acc[k]['matrix'])
        m = m / m.sum(dim=1).unsqueeze(1)
        sns.heatmap(data=m, ax=axs[i // 2, i % 2],
                    vmin=0, vmax=1,
                    xticklabels=action_set_names, yticklabels=action_set_names,
                    cmap=cmap, annot=True, cbar=False)
        axs[i // 2, i % 2].set_title(", ".join(action_set_names))
    plt.savefig(os.path.join(args.eval_save_path, args.model_name))
    plt.show()


# EVALUATION METHODS

def evaluate_distance(model,
                      dataset,
                      distance_fn=DISTANCE_NAMES_MAP['cosine']):
    """
    Runs the evaluation of the model in finding the correct 'after' through minimization of embedding distances.

    :param model: the model to evaluate (should support an `encode_image` method)
    :param dataset: the dataset on which to iterate; should return a dict containing preprocessed tensors
    for `before`, `positive` and `negatives` (this one as list)
    :param distance_fn: a callable (before-embedding, after-embeddings) computing distances between embeddings
    (default: `torch.cosine_similarity`)
    :return: accuracy measure over the whole dataset (in range [0,1]), measured as number of cases where the model could guess the
    correct followup image.
    """
    # TODO: understand what happens when there is only the positive and why accuracy is 0

    model.eval()

    acc_counter = 0
    matrices = {}

    with torch.no_grad():
        for sample in tqdm.tqdm(dataset, total=len(dataset), desc="Evaluating dataset..."):
            b = torch.stack([sample['before']] + sample['negatives'] + [sample['positive']], dim=0).to(model.device)
            embs = model(b)
            distances = distance_fn(embs[0], embs[1:])
            predicted = torch.argmin(distances, dim=-1).item()

            # Adds if prediction is positive
            if predicted == distances.shape[-1] - 1:
                acc_counter += 1

            # Conditional Evaluation
            if args.conditional_eval:
                current_actions = sample['neg_actions'] + [sample['action']]  # defines set of actions in current scope
                actions_set = sorted(list(set(current_actions)), key=int)
                conditioned_str = ",".join(actions_set)  # identifies uniquely current set of actions
                if matrices.get(conditioned_str, None) is None:
                    matrices[conditioned_str] = {
                        'action_idx': dict([(el, i) for i, el in enumerate(actions_set)]),
                        # set up an index conversion table for consistency
                        'matrix': torch.zeros(len(actions_set), len(actions_set), dtype=int)  # create confusion matrix
                    }

                # Gets action indices and updates confusion matrix
                pred_idx = matrices[conditioned_str]['action_idx'][current_actions[predicted]]
                gt_idx = matrices[conditioned_str]['action_idx'][sample['action']]
                matrices[conditioned_str]['matrix'][gt_idx, pred_idx] += 1

        __inverted = True  # check if it gets predicted index 0 by default
        # if not __inverted:
        #     for sample in tqdm.tqdm(dataset, total=len(dataset), desc="Evaluating dataset..."):
        #         b = torch.stack([sample['before'], sample['positive']] + sample['negatives']).to(model.device)
        #         embs = model(b)
        #         distances = distance_fn(embs[0], embs[1:])
        #         predicted = torch.argmin(distances, dim=-1).item()
        #
        #         # Adds if prediction is positive
        #         if predicted == 0:
        #             acc_counter += 1
        #
        #         # Conditional Evaluation
        #         if args.conditional_eval:
        #             current_actions = [sample['action']] + sample['neg_actions']  # defines set of actions in current scope
        #             actions_set = sorted(list(set(current_actions)), key=int)
        #             conditioned_str = ",".join(actions_set)  # identifies uniquely current set of actions
        #             if matrices.get(conditioned_str, None) is None:
        #                 matrices[conditioned_str] = {
        #                     'action_idx': dict([(el, i) for i, el in enumerate(actions_set)]),  # setup an index conversion table for consistency
        #                     'matrix': torch.zeros(len(actions_set), len(actions_set), dtype=int)  # create confusion matrix
        #                 }
        #
        #             # Gets action indices and updates confusion matrix
        #             pred_idx = matrices[conditioned_str]['action_idx'][current_actions[predicted]]
        #             gt_idx = matrices[conditioned_str]['action_idx'][sample['action']]
        #             matrices[conditioned_str]['matrix'][gt_idx, pred_idx] += 1


    s = None
    if args.conditional_eval:
        # checks if accuracy is correct
        s = sum([ac_set['matrix'].diag().sum().item() for ac_set in matrices.values()])
        s = s / len(dataset)

        # converts matrices for readability
        matrices = {k: {'action_idx': matrices[k]['action_idx'], 'matrix': matrices[k]['matrix'].tolist()} for k in matrices}

    return {'accuracy': acc_counter / len(dataset), 'conditional acc': s, 'matrices': matrices}


def evaluate(args):
    model, transform = load_model_and_transform(args)
    dataset = ActionDataset(
        path=args.image_dir,
        transform=None,
        negative_sampling_method=args.negative_sampling_method, soft_negatives_nr=args.negative_nr
    )
    res = evaluate_distance(model, dataset, distance_fn=DISTANCE_NAMES_MAP[args.distance])

    pprint(res)

    if args.save_results:
        if not os.path.exists(args.eval_save_path):
            os.makedirs(args.eval_save_path, exist_ok=True)
        pprint(res, open(os.path.join(args.eval_save_path, args.model_name + ".txt"), mode='wt'))

    if args.plot_results and args.conditional_eval:
        visualize(args, res)

    return res


def eval_untrained_resnet(args):
    results = []
    for seed in range(args.seed_nr):
        print("--------------------------------------------")
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        args.model_name = f'untrained-rn-{seed}'
        tmp = evaluate(args)
        results.append(tmp['accuracy'])
    res = {"results": results, "mean": sum(results) / len(results)}
    pprint(res)
    if args.save_results:
        with open(os.path.join(args.eval_save_path, 'summary.txt'), mode='wt') as fstream:
            pprint(res, fstream)
    return res


def eval_moca(args):
    args.model_name = 'moca-rn'
    args.save_results = True
    args.negative_sampling_method='fixed_onlyhard'
    args.eval_save_path = os.path.join(args.eval_save_path, 'moca_hard_results')
    hard_acc = evaluate(args)['accuracy']

    args.negative_sampling_method = 'random_onlysoft'
    args.eval_save_path = os.path.join(args.eval_save_path, 'moca_soft_results')
    soft_acc = evaluate(args)['accuracy']
    with open(os.path.join(args.eval_save_path, 'summary.txt'), mode='wt') as fstream:
        pprint({'hard': hard_acc, 'soft': soft_acc}, fstream)

    return None


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", default=None, choices=ALLOWED_MODELS + ['all'])
    parser.add_argument("--device", default='cuda' if torch.cuda.is_available() else 'cpu')

    parser.add_argument("--distance", type=str, choices=list(DISTANCE_NAMES_MAP.keys()), default='cosine')

    parser.add_argument("--image_dir", default='dataset/data-bbxs')

    parser.add_argument('--extract_features', dest='extract_features', action='store_true', default=False)

    parser.add_argument("--batch_size", default=32)

    # Evaluation utilities
    parser.add_argument("--evaluate", dest='evaluate', action='store_true', default=False)

    parser.add_argument("--conditional_eval", action='store_true', default=False)
    parser.add_argument("--plot_results", action='store_true', default=False)
    parser.add_argument("--save_results", action='store_true', default=False)
    parser.add_argument("--eval_save_path", type=str, default='tmp_eval_results')

    # Evaluating randomly initialized untrained ResNets
    parser.add_argument("--eval_untrained_rn", action='store_true', default=False)
    parser.add_argument("--seed_nr", type=int, default=5)

    # Evaluating MOCA model
    parser.add_argument("--eval_moca", action='store_true', default=False)

    # Utilities for negative sampling
    parser.add_argument("--negative_sampling_method", type=str, choices=negsamp_choices, default='fixed_onlyhard')
    parser.add_argument("--negative_nr", type=int, default=5)

    args = parser.parse_args()

    if args.eval_untrained_rn:
        res = eval_untrained_resnet(args)
    elif args.eval_moca:
        res = eval_moca(args)
    else:
        if args.model_name is None:
            while args.model_name is None:
                args.model_name = input(f"Please provide a model name in {ALLOWED_MODELS}: ") \
                    if args.model_name not in ALLOWED_MODELS else None

        if args.model_name == 'all':
            to_run = set(ALLOWED_MODELS) - {'all'}
        else:
            to_run = [args.model_name]

        # Runs operations for chosen models
        for mname in to_run:
            args.model_name = mname
            print(f"Running for model {args.model_name}...")

            if args.extract_features:
                extract_features(args)

            if args.evaluate:
                res = evaluate(args)
