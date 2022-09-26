
import os

import pandas
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import PIL
import seaborn as sns
import torchvision.transforms.functional as F

from tqdm import tqdm
from pathlib import Path
from pprint import pprint

import sys
sys.path.append('.')  # needed to execute from top directory

from visual_features.bounding_box_utils import get_iou
from visual_features.utils import load_model_from_path, setup_argparser
from visual_features.data import BBoxDataset

# EVALUATION


def mean(l):
    if len(l) == 0:
        return -1.0
    return sum(l) / len(l)



def eval_separated_tasks(model, valid_ds,
                         iou_threshold=.5, confidence_threshold=.5):
    nr_classes = len(valid_ds.get_object_set())
    conf_matrix = torch.zeros(nr_classes, nr_classes)  # confusion matrix
    scored_conf_matrix = torch.zeros(nr_classes, nr_classes)
    regression_metric = []  # iou

    model.eval()

    with torch.no_grad():
        for data in tqdm(valid_ds, desc='Evaluating...'):
            x, y = data
            x = x.to(model.device).unsqueeze(0)
            yp = model(x)  # list of size [1] of dict(boxes, labels, scores)
            yp = yp[0]  # only 1 image processed
            regression_metric.append([])
            for i in range(yp['boxes'].shape[0]):
                conf_matrix[y['labels'][0], yp['labels'][i]] += 1
                scored_conf_matrix[y['labels'][0], yp['labels'][i]] += yp['scores'][i].cpu()
                regression_metric[-1].append(
                    get_iou(yp['boxes'][i].int().cpu().tolist(), y['boxes'][0].int().cpu().tolist())
                )


    conf_matrix = conf_matrix / conf_matrix.sum(dim=-1).view(-1, 1)
    scored_conf_matrix = scored_conf_matrix / scored_conf_matrix.sum(dim=-1).view(-1, 1)

    regression_metric = [mean(el) for el in regression_metric]

    return conf_matrix, scored_conf_matrix, regression_metric


def run_evaluation(args):
    model_path = os.path.join(args.models_path, args.model_name)
    model = load_model_from_path(model_path, nr=args.file_nr, name=args.file_name, device=args.device)
    ds = BBoxDataset(args.data_path, transform=torchvision.transforms.ToTensor())  # supposing test set
    conf_matrix, scored_conf_matrix, regression_metric = eval_separated_tasks(model, ds)
    names = sorted(list(ds.get_object_set()), key=lambda name: ds.convert_object_to_ids(name))

    plt.figure()
    cmap = sns.color_palette('bone', as_cmap=True)
    sns.heatmap(data=conf_matrix,
                vmin=0, vmax=1,
                xticklabels=names, yticklabels=names,
                cmap=cmap)
    plt.title('confusion matrix')
    plt.ylabel('ground truth')
    plt.figure()
    sns.heatmap(data=scored_conf_matrix, vmin=0, vmax=scored_conf_matrix.max(),
                xticklabels=names, yticklabels=names,
                cmap=cmap)
    plt.ylabel('ground truth')
    plt.title('score-based conf. matrix')
    plt.figure()
    sns.scatterplot(data=pandas.DataFrame({'image id': list(range(len(ds))), 'mean IoU': regression_metric}), x='image id', y='mean IoU')
    plt.title('mean IoU by image')
    plt.show()


def wrap_run_evaluation(args):
    if args.model_name is None:
        raise ValueError('please specify a model')

    if args.model_name == 'all':
        from visual_features.detection import _ALLOWED_MODELS
        for model_name in _ALLOWED_MODELS:
            args.model_name = model_name
            run_evaluation(args)
    else:
        run_evaluation(args)


# PLOTTING

__save_plot_path__ = Path('bbox_results/plots')


def show_tensors(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


def plot_bboxes(model, img, gt, draw_separate_gt=False):
    gt_color = (0,255,0)
    pred_color = torch.tensor((255,0,0))
    width = 1

    if model.training:
        model.eval()

    # assuming 1 image only
    yp = model(img.to(model.device).unsqueeze(0))[0]

    for k in yp:
        yp[k] = yp[k].to('cpu')

    pil_img = F.pil_to_tensor(F.to_pil_image(img))  # need to convert from range (0,1) to (0,255), but also to arrange channels

    pil_img = torchvision.utils.draw_bounding_boxes(pil_img, gt['boxes'], colors=[gt_color], width=width)

    gt_img = F.to_pil_image(pil_img.clone()) if draw_separate_gt else None

    correct_preds = [(yp['boxes'][i], yp['scores'][i]) for i in range(len(yp['boxes']))
                     if yp['labels'][i].squeeze().item() == gt['labels'].squeeze().item()]
    correct_preds = ([correct_preds[i][0] for i in range(len(correct_preds))], [correct_preds[i][1] for i in range(len(correct_preds))])

    if len(correct_preds[0]) > 0:
        # Drawing in reverse order so that the most visible is the one with highest confidence
        correct_img = torchvision.utils.draw_bounding_boxes(
            pil_img, torch.stack(list(reversed(correct_preds[0])), dim=0).int(),
            colors=list(reversed([tuple((pred_color * correct_preds[1][i]).int().tolist()) for i in range(len(correct_preds[1]))])),
            width=width
        )
    else:
        correct_img = pil_img.clone()

    if yp['boxes'].shape[0] > 0:
        wrong_img = torchvision.utils.draw_bounding_boxes(
            pil_img, yp['boxes'].int(),
            colors=[tuple((pred_color * yp['scores'][i]).int().tolist()) for i in range(len(yp['scores']))],
            width=width
        )
    else:
        wrong_img = pil_img.clone()

    new_df_row = {
        'gt_label': gt['labels'].squeeze().item(),
        'preds_label': yp['labels'].flatten().int().tolist(),
        'preds_score': yp['scores'].flatten().tolist(),
        'gt_box': gt['boxes'].int().squeeze().tolist(),
        'preds_box': yp['boxes'].int().squeeze().tolist(),
        'ious': [float(np.round(get_iou(yp['boxes'][i].int().cpu().tolist(), gt['boxes'][0].int().cpu().tolist()), 3)) for i in range(len(yp['boxes']))]
    }

    return F.to_pil_image(correct_img), F.to_pil_image(wrong_img), gt_img, new_df_row


def plot_on_dataset(model_name='moca-rn', models_path='bbox_results',
                    file_nr=0, file_name=None,
                    data_path='dataset/bboxes/test',
                    device='cuda', nr_images_processed=None, **kwargs):

    model = load_model_from_path(os.path.join(models_path, model_name), file_nr, file_name, device)
    ds = BBoxDataset(data_path, transform=torchvision.transforms.ToTensor())

    data_path = Path(data_path)

    parent_path = data_path.parent / 'boxed_dataset'
    os.makedirs(parent_path, exist_ok=True)

    gt_path = parent_path / data_path.parts[-1] / 'ground_truth'
    draw_on_ds = False
    if not os.path.exists(gt_path):
        draw_on_ds = True
        os.makedirs(gt_path)

    loc = None
    if file_nr is not None:
        loc = str(file_nr)
    elif file_name is not None:
        loc = str(file_name)

    pred_path = Path(models_path) / model_name / loc
    os.makedirs(pred_path / 'correct', exist_ok=True)
    os.makedirs(pred_path / 'wrong', exist_ok=True)


    if nr_images_processed is None:
        nr_images_processed = len(ds)

    eval_dicts = []
    for i in tqdm(range(nr_images_processed), desc='Plotting bounding boxes...'):
        x, y = ds[i]
        original_name = ds.image_name_from_idx(y['image_id'].int().squeeze().item())

        correct, wrong, gt, new_row = plot_bboxes(model, x, y, draw_separate_gt=draw_on_ds)

        new_row['filename'] = original_name
        eval_dicts.append(new_row)

        original_name = original_name.split('.')[0] + '.png'  # change image extension
        if draw_on_ds:
            gt.save(gt_path / original_name)

        correct.save(pred_path / 'correct' / original_name)
        wrong.save(pred_path / 'wrong' / original_name)

    if kwargs.get('save_excel', False):
        # rework labels to be readable
        for i in range(len(eval_dicts)):
            eval_dicts[i]['gt_label'] = ds.convert_id_to_obj(eval_dicts[i]['gt_label'])
            try:
                eval_dicts[i]['preds_label'] = [ds.convert_id_to_obj(obj_id) for obj_id in eval_dicts[i]['preds_label']]
            except:
                print(eval_dicts[i]['preds_label'])
                exit(1)
        df = pandas.DataFrame(eval_dicts)
        df = df[['filename'] + [col for col in df.columns.tolist() if col != 'filename']]
        df.to_excel(str(pred_path / 'eval.xlsx'), index=False)


def wrap_plot_on_dataset(args):
    if args.model_name is None:
        raise ValueError('please specify a model')

    if args.model_name == 'all':
        from visual_features.detection import _ALLOWED_MODELS
        for model_name in _ALLOWED_MODELS:
            args.model_name = model_name
            pprint(vars(args))
            plot_on_dataset(**vars(args))
    else:
        plot_on_dataset(**vars(args))


if __name__ == '__main__':
    parser = setup_argparser({
        'models_path': 'bbox_results',
        'model_name': None,
        'file_nr': 0,
        'file_name': None,
        'data_path': 'dataset/bboxes/test',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_excel': False
    })
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--plot_bboxes', action='store_true')
    args = parser.parse_args()

    if args.evaluate:
        wrap_run_evaluation(args)
    elif args.plot_bboxes:
        wrap_plot_on_dataset(args)
    else:
        print(vars(args))

