
import json
import os
from pathlib import Path
from pprint import pprint

import torch
import torch.nn as nn
import torchvision
import wandb
from tqdm import tqdm

from argparse import Namespace, ArgumentParser

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator


import sys
sys.path.append('.')  # needed for executing from top directory

from visual_features.visual_baseline import load_model_and_transform, ALLOWED_MODELS
from visual_features.utils import compute_last_layer_channels, setup_argparser, get_optimizer
from visual_features.data import get_data

from visual_features.vision_helper.engine import train_one_epoch, evaluate
from visual_features.bounding_box_utils import compute_iou

_ALLOWED_MODELS = list((set(ALLOWED_MODELS).union({'torchvision-rcnn'})) - {'all', 'untrained-rn-zero-init'})

__default_train_config__ = {
    'batch_size': (32, ),
    'learning_rate': (1e-4, ),
    'epochs': (30, ),
    'optimizer': ('adam', ['adamw', 'adam', 'sgd']),
    'scheduler': ('none', ['step', 'none']),

    'model_name': ('imagenet-rn', _ALLOWED_MODELS),
    'unfreeze': False,
    'add_fpn': False,

    'data_path': ('dataset/data-bbxs',),
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
    'scheduler',
    'model_name',
    'unfreeze',
    'add_fpn'
]


class FPNBackbone(nn.Module):
    """
    Implements a convolutional network mapping from high to low dimensionality.
    Still a WiP, in future versions should be a full-fledge Feature Pyramid Network.
    """

    def __init__(self, backbone, channel_list=None):
        super(FPNBackbone, self).__init__()
        self.base_model = backbone
        self.device = self.base_model.device

        activation = nn.ReLU

        if channel_list is None:
            channel_list = [compute_last_layer_channels(backbone), 256]
        self.fpn = nn.ModuleList(
            [nn.Sequential(
                nn.Conv2d(channel_list[i], channel_list[i+1], kernel_size=1),
                nn.BatchNorm2d(channel_list[i+1]),
                activation()
            ) for i in range(len(channel_list) - 1)]
        ).to(self.device)

    def forward(self, x):
        x = self.base_model(x)
        for i in range(len(self.fpn)):
            x = self.fpn[i](x)
        return x


def init_detector(nr_target_categories,
                  backbone_name='imagenet-rn',
                  device='cuda',
                  img_mean=None, img_std=None,
                  use_fpn=False):

    bb, transform = load_model_and_transform(
            Namespace(device=device, model_name=backbone_name),
            keep_pooling=backbone_name != 'clip-rn',
            add_flatten=False
    )

    tmean, tstd = transform.transforms[-1].mean, transform.transforms[-1].std  # supposing Normalize is the last one applied
    if img_mean is not None:
        tmean = img_mean
    if img_std is not None:
        tstd = img_std

    if backbone_name == 'moca-rn':
        # excluding FC of pretrained moca convolutional head
        bb = nn.Sequential(*(list(bb.children())[:-1])).to(device)
        bb.device = device

    if use_fpn:

        pre_channels = compute_last_layer_channels(bb)
        post_channels = 256
        intermediate_channels = int((pre_channels + post_channels) / 2)

        # TODO: understand if it is worth implementing a FPN,
        #  but this way to do it is completely wrong!!
        """
       # Actual code for implementing FPN
       fpn = torchvision.ops.feature_pyramid_network.FeaturePyramidNetwork(pre_channels, post_channels)
       roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2)
       """

        # bb = FPNBackbone(bb, channel_list=[pre_channels, intermediate_channels, post_channels]).to(bb.device)
        bb = FPNBackbone(bb)
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=["0"], output_size=7, sampling_ratio=2)

    else:
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)

    bb.out_channels = compute_last_layer_channels(bb)

    anchor_generator = AnchorGenerator(sizes=((16, 32, 64, 128),),
                                       aspect_ratios=((0.5, 1.0, 1.5),))

    net = FasterRCNN(
            bb,
            rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler,
            num_classes=nr_target_categories,
            image_mean=tmean, image_std=tstd,
            min_size=300, max_size=300
        ).to(device)
    net.device = device
    return net


def get_model(args, nr_target_categories, dataset=None):
    if dataset is not None:
        img_mean, img_std = list(dataset.get_stats().values())
    else:
        img_mean = img_std = None

    if args.model_name not in _ALLOWED_MODELS:
        raise ValueError(f"model named '{args.model_name}' is currently not supported")

    if args.model_name == 'torchvision-rcnn':
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, image_mean=img_mean, image_std=img_std)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, nr_target_categories)
    else:
        model = init_detector(nr_target_categories, args.model_name, args.device, use_fpn=args.add_fpn, img_mean=img_mean, img_std=img_std)

    # Be sure that all parameters are trainable
    if args.model_name != 'torchvision-rcnn':
        for p in model.parameters():
            p.requires_grad_(True)
    else:
        print("Warning: torchvision-rcnn is too large to be trained in unfrozen setup; skipping unfreezing.")

    # Freezing convolutional backbone
    if args.model_name != 'untrained-rn-zero-init' and not args.unfreeze:
        for name, param in model.backbone.named_parameters():
            param.requires_grad_(False)

        # Activate gradient descent for FPN network
        if args.add_fpn:
            for param in model.backbone.fpn.parameters():
                param.requires_grad_(True)

    return model.to(args.device)


def train(*args):
    return {k: v.median for k,v in dict(train_one_epoch(*args, print_freq=1000).meters).items() if k.startswith('loss')}


def eval(*args):
    metrics = torch.from_numpy(evaluate(*args).coco_eval['bbox'].stats)
    res = {
        'mAP': metrics[:6].mean().item(),
        'mAR': metrics[6:].mean().item(),
        # 'large-mAP': metrics[5].item(),
        # 'large-mAR': metrics[-1].item()
    }
    return res


def custom_evaluation(model, valid_dl, device):
    model.eval()

    iou_thresh_for_class = [.0, .5, .75, .9]
    conf_mat = [torch.zeros(62, 62, dtype=int) for thr in iou_thresh_for_class]  # one matrix per threshold

    m_iou_history = []
    predicted_boxes_per_image = []

    first_box_miou = []
    first_box_acc = []

    with torch.no_grad():
        for data in tqdm(valid_dl, desc="Evaluating with custom function..."):
            x,y = data
            images = list(img.to(device) for img in x)

            yp = model(images)
            predicted_boxes_per_image.extend([yp[i]['boxes'].shape[0] for i in range(len(yp))])  # iterate over batch to find nr of boxes per image

            mean_ious_per_image = []  # For each image in the batch, compute mean IoU over all the predicted BBoxes

            for i in range(len(yp)):  # for images in batch

                # Compute iou per each bbox
                ious = compute_iou([t for t in yp[i]['boxes']], [y[i]['boxes'][0] for t in yp[i]['boxes']], return_list=True)

                gt_class_label = y[i]['labels'].int().squeeze().item()

                # Update 1st box metrics
                if len(ious) > 0:
                    first_box_miou.append(ious[0])
                    first_box_acc.append(1 if gt_class_label == yp[i]['labels'][0].int().squeeze().item() else 0)

                # For each threshold update confusion matrix
                for i_thr, thr in enumerate(iou_thresh_for_class):
                    toeval_indices = [idx for idx, _ in enumerate(yp[i]['boxes']) if ious[idx] >= thr]
                    if len(toeval_indices) > 0:
                        for idx in toeval_indices:
                            conf_mat[i_thr][gt_class_label, yp[i]['labels'][idx].int().squeeze().item()] += 1

                # Compute mean IoU for current image over all bboxes
                if len(ious) != 0:
                    mean_ious_per_image.append(sum(ious) / len(ious))
                else:
                    mean_ious_per_image.append(0.0)

            # Append mean IoU for current batch (most likely of size 1) to epoch history
            m_iou_history.append(sum(mean_ious_per_image) / len(mean_ious_per_image))

    # Normalizes conf matrices by row
    for i, thr in enumerate(iou_thresh_for_class):
        conf_mat[i] = conf_mat[i] / conf_mat[i].sum(dim=-1).view(-1, 1)
        conf_mat[i][conf_mat[i].isnan()] = 0

        # If completely empty fills diagonal with -1
        # (needed bc. if threshold is too strict matrix gets never updated)
        if torch.all(conf_mat[i] == 0):
            conf_mat[i][tuple(range(conf_mat[i].shape[0])), tuple(range(conf_mat[i].shape[0]))] -= 0.001

    return {**{f'class-accuracy@{thr}': conf_mat[i_thr].diag().mean() for i_thr, thr in enumerate(iou_thresh_for_class)},
            'mIoU': sum(m_iou_history) / len(m_iou_history),
            'mean-boxes-nr': sum(predicted_boxes_per_image) / len(predicted_boxes_per_image),
            '1st-box-mIoU': (sum(first_box_miou) / len(first_box_miou)) if len(first_box_miou) > 0 else 0.0,
            '1st-box-acc': (sum(first_box_acc) / len(first_box_acc)) if len(first_box_acc) > 0 else 0.0}


def iterate(args, model, train_dl, valid_dl, optimizer, scheduler=None, greedy_save=False):
    best_eval_metric = -1.0
    if greedy_save:
        model_save_path = os.path.join(args.save_path, 'checkpoint.pth')

    for ep in range(args.epochs):
        print("---------- EPOCH {} / {} -----------".format(ep + 1, args.epochs))

        loss_train = train(model, optimizer, train_dl, args.device, ep + 1)
        loss_train = {'train-' + name: val for name, val in loss_train.items()}
        s_train = [f'{k}: {v}' for k,v in loss_train.items()]
        print("--- TRAIN ---", s_train, sep='\n')

        valid_metrics = eval(model, valid_dl, args.device)
        custom_metrics = custom_evaluation(model, valid_dl, args.device)

        valid_metrics = {**valid_metrics, **custom_metrics}
        valid_metrics = {'eval-' + name: val for name, val in valid_metrics.items()}

        s_eval = [f'{k}: {v}' for k,v in valid_metrics.items()]
        print("--- EVAL ---", s_eval, sep='\n')

        if scheduler is not None:
            scheduler.step()

        if args.use_wandb:
            wandb.log({
                **loss_train,
                **valid_metrics
            })

        if greedy_save:
            new_candidate = torch.tensor(list(valid_metrics.values())).mean()
            if new_candidate > best_eval_metric:
                best_eval_metric = new_candidate
                torch.save(model.state_dict(), model_save_path)

    if greedy_save:
        model.load_state_dict(torch.load(model_save_path))
        new_name = '{}.pth'.format(len([el for el in os.listdir(Path(model_save_path).parent) if (el.endswith('.pth') and el != 'checkpoint.pth')]))
        os.rename(model_save_path, os.path.join(args.save_path, new_name))
        with open(os.path.join(args.save_path, new_name.replace('.pth', '.json')), mode='wt') as fp:
            json.dump(vars(args), fp)

    return model


def run_training(args):
    pprint(vars(args))

    # Substitute transform for compatibility issues with FasterRCNN (which has its own)
    full_dataset, train_dl, valid_dl, _ = get_data(args.data_path, batch_size=args.batch_size, dataset_type='bboxes',
                                                transform=torchvision.transforms.ToTensor())

    model = get_model(args, len(full_dataset.get_object_set()), dataset=full_dataset)
    optimizer, scheduler = get_optimizer(args, model)
    os.makedirs(args.save_path, exist_ok=True)
    model = iterate(args, model, train_dl, valid_dl, optimizer, scheduler=scheduler, greedy_save=args.save_model)
    return model


def wrap_run_training(args):
    args.save_path = os.path.join(args.save_path, args.model_name)

    restore_fpn = args.fpn
    if args.model_name == 'clip-rn' and args.unfreeze and not args.use_fpn:
        print("CLIP: unfrozen configuration requires FPN for memory limitations; adding it automatically")
        args.use_fpn = True

    if args.use_wandb:
        with wandb.init(project=args.project, name=args.model_name, config={k: v for k,v in vars(args).items() if k in __tosave_hyperparams__}):
            model = run_training(args)
    else:
        model = run_training(args)

    args.fpn = restore_fpn

    return model


def train_all(args):
    parent_save_path = args.save_path
    for mname in _ALLOWED_MODELS:
        args.save_path = parent_save_path
        args.model_name = mname

        if (args.model_name == 'torchvision-rcnn') and args.unfreeze:
            # skip this configuration since it is invalid
            pass
        else:
            _ = wrap_run_training(args)


def debug_evaluation(args):
    # Substitute transform for compatibility issues with FasterRCNN (which has its own)
    full_dataset, train_dl, valid_dl, _ = get_data(args, transform=torchvision.transforms.ToTensor())

    model = get_model(args, len(full_dataset.get_object_set()), dataset=full_dataset)

    valid_metrics = eval(model, valid_dl, args.device)

    valid_metrics = {**valid_metrics, **custom_evaluation(model, valid_dl, args.device)}

    pprint(valid_metrics)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--check_dataset', action='store_true', default=False)
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--train_all', action='store_true', default=False)
    parser.add_argument('--debug_evaluation', action='store_true', default=False)

    parser.add_argument('--wandb_project_name', dest='project', default='Internship-BBoxDetection')

    parser = setup_argparser(__default_train_config__, parser)
    args = parser.parse_args()

    # needed for script usage
    args.batch_size = int(args.batch_size)
    args.learning_rate = float(args.learning_rate)
    args.epochs = int(args.epochs)

    if args.train:
        model = wrap_run_training(args)
    elif args.train_all:
        train_all(args)
    elif args.debug_evaluation:
        debug_evaluation(args)
    else:
        from itertools import product
        from utils import write_model_summary

        n_classes = 50
        device = 'cpu'
        models = ['clip-rn', 'moca-rn', 'imagenet-rn']
        for mname, freeze_val in product(models, [False, True]):
            args = Namespace(device=device, model_name=mname, unfreeze=freeze_val, add_fpn=False)
            model = get_model(args, 50)
            write_model_summary(model, args)

        # clip_args = Namespace(model_name='clip-rn', device='cpu', unfreeze=False)
        # clip_net = get_model(clip_args, 100)
        #
        # imgnet_args = Namespace(model_name='imagenet-rn', device='cpu', unfreeze=False)
        # imgnet_net = get_model(imgnet_args, 100)


        # fmts = "{:^50} | {:>20} | {:^50} | {:>20}"
        # clip_npars = list(clip_net.named_parameters())
        # inet_npars = list(imgnet_net.named_parameters())
        # depth = max(len(clip_npars), len(inet_npars))
        # longer, s_longer, other, s_other = (clip_npars, 'clip', inet_npars, 'imagenet') if len(clip_npars) == depth else (inet_npars, 'imagenet', clip_npars, 'clip')
        # print(fmts.format(s_longer, 'params', s_other, 'params'))
        # for i in range(depth):
        #     longer[i]
        #     if longer[i][0] != other[i][0]:
        #         pass
        #
        # for cpar, ipar in zip(clip_net.named_parameter, imgnet_net.named_parameters()):
        #     print(fmts.format(*cpar, *ipar))
