
"""

"""

import os
import sys
sys.path.append('..')

import torch
import torch.nn as nn
import wandb
import pandas

from argparse import Namespace, ArgumentParser
from tqdm import tqdm

from visual_features.visual_baseline import load_model_and_transform, ALLOWED_MODELS
from visual_features.utils import compute_last_layer_size, setup_argparser
from visual_features.data import get_data
from visual_features.bounding_box_utils import compute_iou


ALLOWED_MODELS = set(ALLOWED_MODELS) - {'all', 'untrained-rn-zero-init'}


class BBLoss(nn.Module):
    def __init__(self, alphas=torch.tensor([1.0, 1.0]), use_ce=True):
        super(BBLoss, self).__init__()
        self.alphas = alphas
        self.use_ce = use_ce
        self.ce = torch.nn.CrossEntropyLoss()
        self.mse = torch.nn.MSELoss()

    def forward(self, yp, yt):
        if self.use_ce:
            l1 = self.ce(yp[0].float(), yt[:, 0].long())
            l2 = self.mse(yp[1].float(), yt[:, 1:].float())
            # l2 = l2 * ((yp[:, 0].argmax(dim=-1).view(-1, 1) == yt[:, 0]).float())  # only compute regression loss for correctly-classified samples
            return self.alphas[0] * l1 + self.alphas[1] * l2
        else:
            tmp = yt
            # tmp = torch.cat((yt[:, :2], yt[:, 2:] - yt[:, 0:2]), dim=-1)  # converting to format (x,y,w,h)
            res = self.mse(yp.float(), tmp.float())
            return res


class ConditionalRegressor(torch.nn.Module):
    def __init__(self, nr_target_categories, backbone_name='imagenet-rn', device='cuda', dropout_p=0.5):
        super(ConditionalRegressor, self).__init__()

        self.__dropout_p = dropout_p
        self.device = device

        self.backbone, self.transform = load_model_and_transform(Namespace(device=device, model_name=backbone_name), keep_pooling=True)

        backbone_out_features = compute_last_layer_size(self.backbone)
        self.conv_finale = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(self.__dropout_p)
        ).to(self.device)

        self.head = nn.Sequential(
            nn.Linear(backbone_out_features, backbone_out_features),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(backbone_out_features, 4 * nr_target_categories),
            nn.Sigmoid()
        ).to(self.device)

        self._target_bbox_index = torch.arange(0, 4 * nr_target_categories, 1).resize_(nr_target_categories, 4).to(self.device)

    def forward(self, x):
        # x = self.transform(x)
        tmp = self.backbone(x)
        embs = tmp.float()
        embs = self.conv_finale(embs)
        return self.head(embs)

    def process_batch(self, batch, loss_fn):
        x, y = batch  # here y should be a dict {'target': torch.LongTensor, 'bbox': torch.FloatTensor of range [0, width] and [0,height}
        x = x.to(self.device)
        yt = y['bbox'].to(self.device)
        out = self(x)
        yp = self.retrieve_bbox_prediction(out, y['target'].to(self.device)) * x.shape[-1]
        loss = loss_fn(yp, yt)
        # yp = torch.cat((yp[:, 0:2], yp[:, 0:2] + yp[:, 2:4]), dim=-1)  # reshape prediction from (x,y,w,h) to (x,y,x1,y1)
        yp_with_confidence = torch.cat([torch.ones((x.shape[0], 1), dtype=yp.dtype).to(self.device), yp], dim=-1)  # adding fake 1.0 confidence for evaluation
        return loss, yp_with_confidence, torch.cat([y['target'].view(-1, 1), y['bbox']], dim=-1)

    def retrieve_bbox_prediction(self, output, target):
        """
        Retrieves values for the bounding box for GT targets from the output vector.
        :param output: vector of predicted BBoxes (shape: batch_size x (4 * nr_target_categories))
        :param target: vector of GT target indices for the current batch (shape: batch_size x 1)
        :return: a tensor of shape batch_size x 4 containing BBox locations for the batch
        """
        return output.gather(-1, self._target_bbox_index.index_select(0, target))


def get_optimizer(args, model):
    if args.optimizer == 'adamw':
        return torch.optim.AdamW(model.parameters(), args.learning_rate)
    elif args.optimizer == 'adam':
        return torch.optim.Adam(model.parameters(), args.learning_rate)


def get_loss(args):
    return BBLoss(use_ce=(args.regressor == 'detector'))


def get_model(args, nr_target_categories):
    model = None
    if args.regressor == 'conditional':
        model = ConditionalRegressor(nr_target_categories, args.model_name, args.device)
    else:
        raise ValueError('unrecognized regressor type: {}'.format(args.regressor))

    # Freezing convolutional backbone
    if args.model_name != 'untrained-rn-zero-init' and not args.unfreeze:
        for name, param in model.backbone.named_parameters():
            param.requires_grad_(False)

    return model.to(args.device), model.transform


def epoch_it(epoch_nr, model, dataloader, optimizer, loss_fn, mode='eval'):
    if mode == 'train':
        loss_history = []
        model.train()
        for batch in tqdm(dataloader, total=len(dataloader), desc='TRAIN for epoch {}...'.format(epoch_nr + 1)):
            optimizer.zero_grad()
            loss, _, _ = model.process_batch(batch, loss_fn)
            loss.backward()
            optimizer.step()
            loss_history.append(loss.squeeze().item())
        return torch.tensor(loss_history).mean(dim=0).item()
    else:
        pred_history = []
        gt_history = []
        model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, total=len(dataloader), desc='EVAL for epoch {}...'.format(epoch_nr + 1)):
                _, predictions, gts = model.process_batch(batch, loss_fn)
                pred_history.append(predictions)
                gt_history.append(gts.int().cpu())

        preds, gts = [el for batch in pred_history for el in batch], [el for batch in gt_history for el in batch]
        # mAP = compute_map(preds, gts)
        # return mAP
        iou = compute_iou(preds, gts)
        return iou


def train(*args):
    return epoch_it(*args, mode='train')


def evaluate(*args):
    return epoch_it(*args, mode='eval')


def iterate(args, model, train_dl, valid_dl, optimizer, loss_fn, greedy_save=False):
    best_mAP = 0.0
    if greedy_save:
        model_save_path = os.path.join(args.save_path, 'checkpoint.pth')
    with open(os.path.join(args.save_path, 'train.txt'), mode='wt') as resfile:
        for ep in range(args.epochs):
            print("---------- EPOCH {} / {} -----------".format(ep + 1, args.epochs))
            resfile.write("---------- EPOCH {} / {} -----------\n".format(ep + 1, args.epochs))
            train_loss = train(ep, model, train_dl, optimizer, loss_fn)
            s_train = "TRAIN --> loss: " + str(train_loss)
            print(s_train)

            valid_mAP = evaluate(ep, model, valid_dl, optimizer, loss_fn)
            # s_valid = "VALID --> mAP: " + str(valid_mAP)
            s_valid = "VALID --> iou: " + str(valid_mAP)
            print(s_valid)

            resfile.write(s_train + '\n' + s_valid + '\n')

            if args.use_wandb:
                wandb.log({
                    'train-loss': train_loss,
                    # 'valid-mAP': valid_mAP,
                    'valid-iou': valid_mAP
                })

            if greedy_save:
                if valid_mAP > best_mAP:
                    best_mAP = valid_mAP
                    torch.save(model.state_dict(), model_save_path)
    if greedy_save:
        model.load_state_dict(model_save_path)
    return model


def run_training(args):
    obj_dict = {obj: i for i, obj in enumerate(list({
        el.split("|")[0] for el in pandas.read_csv(os.path.join(args.data_path, 'labels.csv'))['object_id']
    }))}
    model, transform = get_model(args, len(obj_dict))
    full_dataset, train_dl, valid_dl, _ = get_data(args.data_path, batch_size=args.batch_size, dataset_type='bboxes', model_transform=transform)
    loss = get_loss(args)
    optimizer = get_optimizer(args, model)
    os.makedirs(args.save_path, exist_ok=True)
    model = iterate(args, model, train_dl, valid_dl, optimizer, loss, greedy_save=args.save_model)
    return model


__default_train_config__ = {
    'batch_size': (128, ),
    'learning_rate': (1e-3, ),
    'epochs': (40, ),
    'optimizer': ('adam', ['adamw', 'adam']),

    'model_name': ('imagenet-rn', ALLOWED_MODELS),
    'deep_classifier': False,
    'unfreeze': False,

    'data_path': ('dataset/bboxes',),
    'save_model': False,
    'save_path': ('bbox_results', ),

    'device': ('cuda' if torch.cuda.is_available() else 'cpu', ['cuda', 'cpu']),
    'dataparallel': False,
    'use_wandb': False
}

__tosave_hyperparams__ = [
    'batch_size',
    'learning_rate',
    'epochs',
    'optimizer',
    'model_name',
    'unfreeze'
]


def freeze_exp(args):
    models = set(ALLOWED_MODELS) - {'all', 'untrained-rn-zero-init'}

    for v in [True, False]:
        args.unfreeze = v
        for mname in models:
            args.project = 'frozen' if not args.unfreeze else 'unfrozen'
            args.model_name = mname
            if args.use_wandb:
                with wandb.init(project=args.project, name=mname + "({})".format(args.project), config={k: vars(args)[k] for k in __tosave_hyperparams__}):
                    _ = run_training(args)
            else:
                _ = run_training(args)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--freeze_unfreeze_exp', action='store_true', default=False)

    parser.add_argument('--wandb_project_name', dest='project', default='Internship-BBoxDetection')

    parser = setup_argparser(__default_train_config__, parser)

    args = parser.parse_args()

    # needed for run_script.sh
    args.save_path = os.path.join(args.save_path, args.regressor, args.model_name)
    args.batch_size = int(args.batch_size)
    args.learning_rate = float(args.learning_rate)
    args.epochs = int(args.epochs)

    if args.freeze_unfreeze_exp:
        freeze_exp(args)
    elif args.train:
        if args.use_wandb:
            with wandb.init(project=args.project, name=args.regressor + "(" + args.model_name + ")", config=vars(args)):
                model = run_training(args)
        else:
            model = run_training(args)
    else:
        model_name = 'clip-rn'
        unf_args = Namespace(model_name=model_name, regressor='conditional', unfreeze=True, device='cpu')
        f_args = Namespace(model_name=model_name, regressor='conditional', unfreeze=False, device='cpu')
        clip_net = get_model(f_args, 100)[0]
        print('Frozen:')
        print("Backbone:")
        print(sum([t.numel() for t in clip_net.backbone.parameters()]))
        print(sum([t.numel() for t in clip_net.backbone.parameters() if t.requires_grad]))
        print("Detector:")
        print(sum([t.numel() for t in clip_net.parameters()]))
        print(sum([t.numel() for t in clip_net.parameters() if t.requires_grad]))

        clip_net = get_model(unf_args, 100)[0]
        print('Unfrozen:')
        print("Backbone:")
        print(sum([t.numel() for t in clip_net.backbone.parameters()]))
        print(sum([t.numel() for t in clip_net.backbone.parameters() if t.requires_grad]))
        print("Detector:")
        print(sum([t.numel() for t in clip_net.parameters()]))
        print(sum([t.numel() for t in clip_net.parameters() if t.requires_grad]))

