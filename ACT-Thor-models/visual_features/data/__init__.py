import json
import os
import random
import pickle

import pandas
import torch

from pathlib import Path

import torchvision.transforms
from matplotlib import use
from torchvision import transforms
from collections import defaultdict
from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm

from torch.nn.utils.rnn import pad_sequence

import itertools

import sys

sys.path.extend(['..', '.'])

from visual_features.vision_helper.utils import collate_fn as default_collate_fn

__default_dataset_fname__ = 'dataset_with_new_splits.csv'
__default_dataset_path__ = 'new-dataset/data-improved-descriptions'

NEGATIVE_SAMPLING_METHODS = [
    'fixed_onlyhard',
    'random_onlysoft',
    'default3',
    'default4'
    # 'fixed_soft+hard', 'random_soft+hard'
]

# TODO: make the following two dynamic and loaded from an external file

__other_scene_action_id__ = 100
ACTION_CONVERSION_TABLE = {
    1: 'drop', 2: 'throw', 3: 'put', 4: 'push', 5: 'pull', 6: 'open', 7: 'close',
    8: 'slice', 9: 'dirty', 10: 'fill', 11: 'empty',
    12: 'toggle', 13: 'useUp', 14: 'break', 15: 'cook',
    __other_scene_action_id__: 'other'  # needed for soft negatives
}

# TODO update this set
# here not to be used dynamically but only as a reference
__bbox_target_categories__ = {
    'CellPhone', 'Pen', 'Towel', 'Candle', 'SoapBar', 'Footstool', 'BaseballBat', 'WateringCan',
    'SoapBottle', 'Egg', 'DishSponge', 'Book', 'HandTowel', 'Ladle', 'Pencil', 'Plunger', 'Kettle',
    'Lettuce', 'TeddyBear', 'TableTopDecor', 'Box', 'Bowl', 'AluminumFoil', 'Plate', 'Pillow',
    'Vase', 'Mug', 'Pan', 'Pot', 'RemoteControl', 'KeyChain', 'SaltShaker', 'SprayBottle', 'Cup',
    'TennisRacket', 'Boots', 'Bread', 'Bottle', 'Knife', 'CD', 'Potato', 'Tomato', 'Newspaper',
    'Watch', 'CreditCard', 'Dumbbell', 'ButterKnife', 'TissueBox', 'Statue', 'AlarmClock',
    'Spatula', 'ToiletPaper', 'Cloth', 'ScrubBrush', 'Fork', 'Laptop', 'BasketBall', 'WineBottle',
    'PepperShaker', 'Spoon', 'Apple', 'PaperTowelRoll'}

__nr_bbox_target_categories__ = len(__bbox_target_categories__)

__default_dataset_stats__ = {
    'mean': torch.tensor([0.4689, 0.4682, 0.4712]),
    'std': torch.tensor([0.2060, 0.2079, 0.2052])
}  # to rework with full dataset


def convert_action(name: str):
    """
    Returns action extended description from action id.
    :param name: name (in numbers) representing the action
    :return: a string describing the action
    """
    # TODO: implement action conversion table
    return ACTION_CONVERSION_TABLE[int(name)]


def load_and_rework_csv(basepath, dfname, drop_cook=True):
    """Loads the annotation DataFrame from the specified csv file and reworks paths to coincide
     with the specified basepath. Also drops contrasts (rows) where
    the action is 'cook' is present either as positive or negative."""
    from pathlib import Path
    basepath = Path(basepath)
    df = pandas.read_csv(basepath / dfname, index_col=0)

    def rework_annotations_path(_df, _basepath):
        # TODO: automatic handling of different path specifications
        for col in _df.columns:
            if "path" in col:
                _df[col] = _df[col].map(lambda pth: str(Path(*_basepath.parts[:-1], pth)))
        return _df

    df = rework_annotations_path(df, basepath)

    if drop_cook:
        for c in ['action_name', 'distractor0_action_name', 'distractor1_action_name', 'distractor2_action_name']:
            df.drop(index=df[df[c] == 'cook'].index, inplace=True)

    df.index = pandas.Series(range(len(df)))

    return df


class ActionDataset(Dataset):
    def __init__(self,
                 path=__default_dataset_path__,
                 cs_type='CS3',
                 image_extension='png',
                 transform=None,
                 **kwargs):

        self.path = Path(path)

        self._stats = __default_dataset_stats__

        # annotations dataframe
        self.annotation = load_and_rework_csv(self.path, __default_dataset_fname__)

        # self.annotation = pandas.read_csv(self.path / 'dataset_cs_scene_object_nopt_augmented_recep.csv', index_col=0)

        # # ALTERNATIVE OBJECT SPLIT
        # self.annotation = load_and_rework_csv(self.path, 'alternative_obj_split_dataset.csv')
        # print("USING ALTERNATIVE OBJECT SPLIT")

        # self.annotation = rework_annotations_path(self.annotation, self.path)  # rework paths for mismatch at top directory
        #
        # # # remove 'cook' action
        # for c in ['action_name', 'distractor0_action_name', 'distractor1_action_name', 'distractor2_action_name']:
        #     self.annotation.drop(index=self.annotation[self.annotation[c] == 'cook'].index, inplace=True)
        #
        # self.annotation.index = pandas.Series(range(len(self.annotation)))  # rework index for consistency

        assert cs_type in {'CS3', 'CS4'}
        self.cs_type = cs_type

        self.image_extension = image_extension

        # TODO: some action indices are missing (8, 13), so I had to workaround by using a new enumeration
        self.actions_map = {ac: i for i, ac in enumerate(sorted(list(
            {el for name in ['distractor0_action_name', 'distractor1_action_name', 'distractor2_action_name'] for el in
             self.annotation[name]}
        )))}

        if not os.path.exists(self.path / 'actions.json'):
            with open(self.path / 'actions.json', mode='wt') as fp:
                json.dump(self.actions_map, fp)

        self.objects = dict(set(zip(
            self.annotation['object_name'].to_list(),
            [int(self.annotation.loc[i, 'image_name'].split("_")[0]) for i in range(len(self.annotation))]
        )))

        # defines default transformation of torchvision models (if not provided) + normalize with its stats
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(self._stats['mean'], self._stats['std'])
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.annotation)

    def _read_image_from_annotation_path(self, path: str):
        return Image.open(Path(path))

    def __getitem__(self, i) -> dict:
        smp = self.annotation.iloc[i]

        if self.cs_type == 'CS3':
            contrast_paths = [smp[col + "_path"] for col in ['distractor0', 'distractor1']]
            contrast_actions = [smp[col + "_action_name"] for col in ['distractor0', 'distractor1']]
        else:
            contrast_paths = [smp[col + "_path"] for col in ['distractor0', 'distractor1', 'distractor2']]
            contrast_actions = [smp[col + "_action_name"] for col in ['distractor0', 'distractor1', 'distractor2']]
        res = {
            'positive': self.transform(self._read_image_from_annotation_path(smp['after_image_path'])),
            'pos_path': smp['after_image_path'],
            'before': self.transform(self._read_image_from_annotation_path(smp['before_image_path'])),
            'before_path': smp['before_image_path'],
            'action': self.actions_map[smp['action_name']],
            'neg_actions': [self.actions_map[ac] for ac in contrast_actions],
            'negatives': [self.transform(self._read_image_from_annotation_path(pth)) for pth in contrast_paths],
            'neg_paths': contrast_paths,
            'object': smp['object_name']
        }
        return res

    def getitem_no_images(self, i):
        smp = self.annotation.iloc[i]

        if self.cs_type == 'CS3':
            contrast_paths = [smp[col + "_path"] for col in ['distractor0', 'distractor1']]
            contrast_actions = [smp[col + "_action_name"] for col in ['distractor0', 'distractor1']]
        else:
            contrast_paths = [smp[col + "_path"] for col in ['distractor0', 'distractor1', 'distractor2']]
            contrast_actions = [smp[col + "_action_name"] for col in ['distractor0', 'distractor1', 'distractor2']]

        return {
            'pos_path': smp['after_image_path'],
            'before_path': smp['before_image_path'],
            'action': self.actions_map[smp['action_name']],
            'neg_actions': [self.actions_map[ac] for ac in contrast_actions],
            'neg_paths': contrast_paths,
            'object': smp['object_name']
        }

    @staticmethod
    def _scene_from_path(p) -> str:
        raise RuntimeError(
            "this was used before migrating to scene-as-directory file structure, it is not intended to be used.")

    @staticmethod
    def obj_from_path(p) -> str:
        return str(Path(p).parts[-1])[:-4].split("_")[0]

    @staticmethod
    def action_from_path(p) -> str:
        return str(Path(p).parts[-1])[:-4].split("_")[-1]

    def get_stats(self):
        if self._stats is None:
            # tmp = self.annotation.copy()
            # tmp[lambda df: df['after_image_path'].isnull()]['after_image_path'] = tmp[lambda df: df['after_image_path'].isnull()]['before_image_path']
            acc = torch.stack([
                to_tensor(Image.open(pth))
                for pth in
                set(self.annotation['after_image_path'].to_list() + self.annotation['before_image_path'].to_list())
            ]).view(3, -1)
            self._stats = {'mean': acc.mean(dim=-1), 'std': acc.std(dim=-1)}

        return self._stats


class VecTransformDataset(Dataset):
    allowed_holdout_procedures = ['object_name',
                                  'scene',
                                  'samples',

                                  'action',

                                  'structural',
                                  'reversible',
                                  'receptacle',
                                  'surface']

    def __init__(self,
                 extractor_model='moca-rn',
                 hold_out_procedure='objects',
                 override_transform=None,
                 **kwargs):
        self._action_dataset = ActionDataset(**kwargs, transform=transforms.ToTensor())
        self.cs_type = self._action_dataset.cs_type

        self.extractor_model = extractor_model

        self.path = (self._action_dataset.path / 'visual-vectors' / self.extractor_model)

        self.actions_to_ids = self._action_dataset.actions_map
        self.ids_to_actions = {v: k for k, v in self.actions_to_ids.items()}
        self.hold_out_procedure = hold_out_procedure

        if isinstance(override_transform, str):
            if override_transform == 'to_tensor':
                self.override_transform = transforms.ToTensor()
        else:
            self.override_transform = override_transform

        if not os.path.exists(self.path):
            os.makedirs(self.path, exist_ok=True)
            self.vectors = self.preprocess()
            with open(self.path / 'vectors.pkl', mode='wb') as vpth:
                pickle.dump(self.vectors, vpth)
        else:
            with open(self.path / 'vectors.pkl', mode='rb') as vpth:
                self.vectors = pickle.load(vpth)

        # Please modify this to add new hold out procedures
        self._internal_split_name_map = {
            'object_name': 'object_split',
            'scene': 'scene_split',
            'samples': 'sample_split',

            'action': 'action_split',

            'structural': 'structural_split',
            'reversible': 'reversible_split',
            'receptacle': 'receptacle_split',
            'surface': 'surface_split'
        }

        self.hold_out_rows = self._action_dataset.annotation[self._internal_split_name_map[self.hold_out_procedure]][
            lambda el: el == 'test'].index
        self.seen_rows = self._action_dataset.annotation[self._internal_split_name_map[self.hold_out_procedure]][
            lambda el: el != 'test'].index

    def __len__(self):
        return len(self._action_dataset)

    def __getitem__(self, item):
        res = self._action_dataset.getitem_no_images(item)
        res.update({
            'before': self.vectors[res['before_path']],
            'positive': self.vectors[res['pos_path']],
            'negatives': [self.vectors[pth] for pth in res['neg_paths']],
        })
        return res

    def preprocess(self):
        """
        Preprocesses the whole dataset by creating & saving vectors of visual features corresponding to the chosen
        feature extraction architecture.
        """
        from argparse import Namespace
        from visual_features.visual_baseline import load_model_and_transform

        extractor, transform = load_model_and_transform(
            Namespace(model_name=self.extractor_model, device='cuda' if torch.cuda.is_available() else 'cpu'),
            keep_pooling=True,  # self.extractor_model != 'clip-rn',
            add_flatten=True
        )
        extractor.eval()

        # adjusts transform with dataset statistics
        if isinstance(transform, transforms.Compose):
            for t in transform.transforms:
                if isinstance(t, transforms.Normalize):
                    t.mean = self._action_dataset.get_stats()['mean']
                    t.std = self._action_dataset.get_stats()['std']

        self.vectors = None

        # gathers all paths of before, after and distractors
        paths = {el for s in [self._action_dataset.annotation[col] for col in self._action_dataset.annotation.columns if
                              "path" in col] for el in s}

        def get_vec(pth):
            return extractor(transform(Image.open(pth)).unsqueeze(0).to(extractor.device)).squeeze().cpu()

        with torch.no_grad():
            self.vectors = {
                pth: get_vec(pth) for pth in
                tqdm(paths, desc=f"Extracting visual vectors with {self.extractor_model}...")
            }

        return self.vectors

    def split(self, valid_ratio=-1.0, for_regression=False):
        """Splits dataset with the current hold-out settings (defined in initialization). An additional parameter
        allows controlling whether data should be prepared for regression or not.
        Returns two torch Subset objects that contain samples with the training and the held-out sets."""

        hold_out = Subset(self, self.hold_out_rows.to_list())
        if for_regression:
            train_after_df = self._action_dataset.annotation.iloc[self.seen_rows]
            train_after = {action: [] for action in set(self.actions_to_ids.values())}
            train_before = {action: [] for action in set(self.actions_to_ids.values())}
            for i, after_row in train_after_df.iterrows():
                train_after[self.actions_to_ids[after_row['action_name']]].append(
                    self.vectors[after_row['after_image_path']])
                train_before[self.actions_to_ids[after_row['action_name']]].append(
                    self.vectors[after_row['before_image_path']])

            vec_dtype = list(self.vectors.values())[0].dtype
            vec_device = list(self.vectors.values())[0].device
            for action in train_after:
                if len(train_after[action]) > 0:
                    train_after[action] = torch.stack(train_after[action], dim=0)
                    train_before[action] = torch.stack(train_before[action], dim=0)
                else:
                    print(
                        f"Not found samples for action {self.ids_to_actions[action]}, hold out set is {self.get_hold_out_items()}")
                    train_after[action] = torch.tensor([[]], dtype=vec_dtype, device=vec_device)
                    train_before[action] = torch.tensor([[]], dtype=vec_dtype, device=vec_device)

            return (train_before, train_after), hold_out
        else:
            assert (0 < valid_ratio < 1) or valid_ratio == -1.0, "choose for validation ratio a value in (0, 1)"
            train = Subset(self, self.seen_rows.to_list())
            valid_indices = []
            if valid_ratio > 0:
                sep = int(len(train) * (1 - valid_ratio))
                train_indices = list(range(len(train)))[:sep]
                valid_indices = list(range(len(train)))[sep:]

                train_set = Subset(train, train_indices)
                valid_set = Subset(train, valid_indices)

                return train_set, valid_set, hold_out
            else:
                valid = Subset(train, [])
                return train, valid, hold_out

    def change_action_split(self, combination, batch_size, valid_ratio=0.2):
        """Changes the test set by associating new unique objects for each action.
        Needs parameters to `get_data` because actually instantiates DataLoaders too."""

        def local(row):
            return 'test' if any([(row.action_name == ac) and (row.object_name == obj)
                                  for ac, obj in combination]) else 'train'

        self._action_dataset.annotation['action_split'] = self._action_dataset.annotation.apply(lambda row: local(row),
                                                                                                axis=1)

        indices = self._action_dataset.annotation[lambda df: df.action_split == 'train'].index.tolist()
        random.shuffle(indices)
        valid_indices = indices[:int(0.2 * len(indices))]

        self._action_dataset.annotation.at[pandas.Series(valid_indices), "action_split"] = 'valid'

        self.hold_out_rows = self._action_dataset.annotation[lambda df: df.action_split == 'test'].index
        self.seen_rows = self._action_dataset.annotation[lambda df: df.action_split != 'test'].index

        # DataLoaders
        train_set, valid_set, test_set = self.split(for_regression=False, valid_ratio=valid_ratio)

        touse_collate = contrastive_collate  # NOTE: assuming usage of InfoNCE for simplicity
        train_dl = torch.utils.data.DataLoader(train_set, batch_size=batch_size, collate_fn=touse_collate)
        valid_dl = torch.utils.data.DataLoader(valid_set, batch_size=1, collate_fn=touse_collate)
        test_dl = torch.utils.data.DataLoader(test_set, batch_size=1, collate_fn=touse_collate)

        return self, train_dl, valid_dl, test_dl

    def get_vec_size(self):
        return list(self.vectors.values())[0].shape[-1]

    def get_hold_out_items(self):
        ho_items = {
            'object_name': set(self._action_dataset.annotation.loc[self.hold_out_rows, 'object_name'].to_list()),
            'scene': set(self._action_dataset.annotation.loc[self.hold_out_rows, 'scene'].to_list()),
            'samples': set(self.hold_out_rows.to_list()),
            'action': set(list(zip(self._action_dataset.annotation.loc[self.hold_out_rows, 'object_name'],
                                   self._action_dataset.annotation.loc[self.hold_out_rows, 'action_name'])))
        }
        return ho_items[self.hold_out_procedure]

    def get_nr_hold_out_samples(self):
        return len(self.hold_out_rows)

    def get_annotation(self):
        return self._action_dataset.annotation


class BBoxDataset(torch.utils.data.Dataset):

    def __init__(self,
                 path='dataset/data-bbxs',
                 transform=None,
                 image_extension='png',
                 object_conversion_table=None):

        # TODO: add split information and separate train/valid/test splits
        # TODO: add info about pickupable vs non-pickupable (up to now it is supposed to be either one of the two)
        self.path = Path(path)
        self.image_extension = image_extension

        # TODO: update these values with newer ones
        self._stats = __default_dataset_stats__

        self.transform = transforms.ToTensor() if transform is None else transform

        self._annotations = load_and_rework_csv(self._annotations, self.path, drop_cook=False)

        added_before_samples = {}
        for idx, row in self._annotations.iterrows():
            if added_before_samples.get(row['before_image_path'], None) is None:
                tmp = row.copy()
                tmp['after_image_path'] = None
                tmp['after_image_bb'] = None
                tmp['image_name'] = tmp['before_image_name']  # needed for image_name_from_idx
                added_before_samples[row['before_image_path']] = tmp

        self._annotations = self._annotations.append(list(added_before_samples.values()), ignore_index=True)
        self._annotations = self._annotations.sample(frac=1.0,
                                                     random_state=42)  # just needed to shuffle before and after rows, more stochasticity may come in DataLoaders

        self.objs_to_ids = {name: i for i, name in enumerate(list(
            set(self._annotations['object_name'])))} if object_conversion_table is None else object_conversion_table

        self.ids_to_objs = {v: k for k, v in self.objs_to_ids.items()}

        self.excluded_files = []
        for i, row in self._annotations.iterrows():
            name = row['after_image_path'] if row['after_image_path'] is not None else row['before_image_path']
            boxes = row['after_image_bb'] if row['after_image_bb'] is not None else row['before_image_bb']
            boxes = self.tensorize_bbox_from_str(boxes)
            if (not torch.logical_and(boxes[0] < boxes[2], boxes[1] < boxes[3])) or \
                    (not row['visible']):
                self.excluded_files.append(name)
                self._annotations.drop(labels=i, axis=0, inplace=True)

        self._annotations.index = pandas.Series(range(len(self._annotations)))  # rework for better consistency

    def __len__(self):
        return len(self._annotations)

    def __getitem__(self, item):
        smp = self._annotations.iloc[item]
        if smp['after_image_path'] is None:
            # is a before image
            imgpath = smp['before_image_path']
            obj_id = self.convert_object_to_ids(self.obj_name_from_df_row(smp))
            boxes = self.tensorize_bbox_from_str(smp['before_image_bb']).unsqueeze(0)
        else:
            imgpath = smp['after_image_path']
            obj_id = self.convert_object_to_ids(self.obj_name_from_df_row(smp))
            boxes = self.tensorize_bbox_from_str(smp['after_image_bb']).unsqueeze(0)

        assert torch.logical_and(boxes[:, 0] < boxes[:, 2], boxes[:, 1] < boxes[:, 3]), \
            f"wrong box for image {imgpath} ({boxes.int().squeeze().tolist()})"

        img = self.transform(Image.open(imgpath))

        # convert to COCO annotation
        ann = {
            'image_id': torch.tensor([item]),
            'boxes': boxes,
            'labels': torch.tensor([obj_id]),
            'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            'iscrowd': torch.zeros(1, dtype=torch.int)
        }

        return img, ann

    def convert_object_to_ids(self, obj_name):
        return self.objs_to_ids[obj_name]

    def convert_id_to_obj(self, _id):
        return self.ids_to_objs[_id]

    def get_object_set(self):
        return set(self.objs_to_ids.keys())

    @classmethod
    def obj_name_from_df_row(cls, row):
        return row['object_name']

    @classmethod
    def tensorize_bbox_from_str(cls, s):
        return torch.tensor([int(el) for el in s.strip('()').split(',')], dtype=torch.float16)

    def get_stats(self):
        if self._stats is None:
            tmp = self._annotations.copy()
            tmp[lambda df: df['after_image_path'].isnull()]['after_image_path'] = \
            tmp[lambda df: df['after_image_path'].isnull()]['before_image_path']
            acc = torch.stack([to_tensor(Image.open(pth)) for pth in tmp['after_image_path']]).view(3, -1)
            self._stats = {'mean': acc.mean(dim=-1), 'std': acc.std(dim=-1)}

        return self._stats

    def image_name_from_idx(self, i):
        return self._annotations.loc[i, 'image_name']


# Other utilities
def vect_collate(batch):
    """
    :param batch: list of dicts containing 'before' tensor, 'action' id, 'positive' after tensor
    :return: a tuple (stacked inputs, stacked actions, stacked outputs)
    """
    before = torch.stack([batch[i]['before'] for i in range(len(batch))], dim=0)
    actions = torch.tensor([batch[i]['action'] for i in range(len(batch))], dtype=torch.long)
    after = torch.stack([batch[i]['positive'] for i in range(len(batch))], dim=0)
    return before, actions, after


def contrastive_collate(batch):
    """
    :param batch: list of dicts containing 'before' --> tensor, 'action' --> id, 'positive' --> after tensor, 'negatives' --> list of tensors, 'neg_actions' --> list of action ids for negatives
    :return: a tuple (stacked befores, stacked positives, stacked postive actions, batch of stacked negatives, batch of stacked neg-actions)
    """
    # TODO produce mask and perform padding
    res = {
        'before': torch.stack([batch[i]['before'] for i in range(len(batch))], dim=0),
        'action': torch.tensor([batch[i]['action'] for i in range(len(batch))], dtype=torch.long),
        'positive': torch.stack([batch[i]['positive'] for i in range(len(batch))], dim=0),
        'negatives': torch.stack([torch.stack(batch[i]['negatives'], dim=0) for i in range(len(batch))], dim=0)
        # 'neg_actions': pad_sequence([torch.tensor(batch[i]['neg_actions'], dtype=torch.long) for i in range(len(batch))], batch_first=True, padding_value=-1),
        # 'negatives': pad_sequence([torch.stack(batch[i]['negatives'], dim=0) for i in range(len(batch))], batch_first=True, padding_value=0)
    }
    # res['mask'] = (res['neg_actions'] >= 0).float()
    return res


def get_data(data_path, batch_size=32, dataset_type=None, obj_dict=None, transform=None, valid_ratio=0.2, **kwargs):
    """
    Returns dataset and 3 dataloaders (train, validation, test) for the specified `dataset_type` parameter:
    `bboxes` for bounding boxes dataset and `vect` for vector-transform dataset.
    Within `kwargs` there are 2 important parameters:
        * `use_regression`: decides whether to have a simpler dataset for least-squares regression
        * `use_contrastive` / `use_infonce`: determine if the collate function should be simple
        (with only before, action, after) or compatible with contrastive losses as InfoNCE (containing also negatives)

    The seen-unseen split is performed by the dataset according to the `holdout_procedure` argument.

    """

    if dataset_type == 'bboxes':
        # Transforms should be not none because FastRCNN require PIL images
        dataset = BBoxDataset(data_path, transform=transform, object_conversion_table=obj_dict)
        indices = list(range(len(dataset)))
        sep = int(len(dataset) * (1 - valid_ratio))
        train_set = Subset(dataset, indices[:sep])
        valid_set = Subset(dataset, indices[sep:])
        train_dl = torch.utils.data.DataLoader(train_set, batch_size=batch_size, collate_fn=default_collate_fn)
        valid_dl = torch.utils.data.DataLoader(valid_set, batch_size=1, collate_fn=default_collate_fn)
        test_dl = valid_dl
    elif dataset_type == 'actions':
        raise NotImplementedError
    elif dataset_type == 'vect':
        dataset = VecTransformDataset(path=data_path, override_transform=transform, **kwargs)
        if kwargs.get('use_regression', False):
            reg_mat, test_set = dataset.split(for_regression=True)
            test_dl = torch.utils.data.DataLoader(test_set, batch_size=1, collate_fn=vect_collate)
            return dataset, reg_mat, test_dl

        train_set, valid_set, test_set = dataset.split(valid_ratio=valid_ratio, for_regression=False)

        touse_collate = contrastive_collate if (
                    kwargs.get('use_contrastive', False) or kwargs.get('use_infonce', False)) else vect_collate
        train_dl = torch.utils.data.DataLoader(train_set, batch_size=batch_size, collate_fn=touse_collate)
        valid_dl = torch.utils.data.DataLoader(valid_set, batch_size=1, collate_fn=touse_collate)
        test_dl = torch.utils.data.DataLoader(test_set, batch_size=1,
                                              collate_fn=touse_collate)  # here for compatibility, but dataset will be accessed with test_dl.dataset
    else:
        raise ValueError(f"unsupported dataset type '{dataset_type}'")
    return dataset, train_dl, valid_dl, test_dl


if __name__ == "__main__":
    from pprint import pprint
    from visual_features.data import *

    ds = VecTransformDataset()
    pprint(ds[0])
