from argparse import ArgumentParser
from pathlib import Path
from collections import defaultdict
import random

from tqdm import tqdm
import pandas as pd
import numpy as np

random.seed(42)
parser = ArgumentParser()
parser.add_argument('-p', '--path', type=str, default='data')

args = parser.parse_args()
path = Path(args.path)

# enforce put / throw constraint
opposite_action = {'put':'throw', 'throw':'put', 'push':'pull', 'pull':'push'}


def collate_datasets(path):
    path = Path(path)
    dfs = []
    for possible_dir in path.iterdir():
        if not possible_dir.is_dir():
            continue
        df_path = possible_dir / 'metadata.csv'
        df = pd.read_csv(df_path)
        dfs.append(df)
    full_df = pd.concat(dfs)
    full_df.to_csv(path / 'metadata.csv')
    return full_df


def generate_contrast_set(path='new-data', n_imgs=None, share_held=True, share_scene=True, share_object=True,
                          prohibit_put_throw=True, share_code=True, force_remake=False):
    path = Path(path)
    filename = 'metadata_filtered'
    suffix = ''
    if share_held:
        suffix += '_hs'
    if share_scene:
        suffix += '_scene'
    if share_object:
        suffix += '_object'
    if prohibit_put_throw:
        suffix += '_nopt'
    filename = filename + suffix + '.csv'
    filtered_metadata_path = path / filename

    df = pd.read_csv(path / 'metadata.csv')

    # filtering the dataset to eliminate examples where the action failed or left no visible trace
    # yes, == True is weird, but in this case can't be omitted / changed to is True
    successful = df['success'] == True
    # since breaking and slicing technically destroy the object, they always show visible = False
    visible = (df['visible'] == True) | (df['action_name'] == 'break') | (df['action_name'] == 'slice')
    visible_diff = df['visible_difference'] > 0
    print('invisible', len(df) - np.sum(visible), '/', len(df))
    print('no diff', np.sum(~visible_diff), '/', len(df))
    df = df[successful & visible & visible_diff]
    cleaned_df = df

    assert len(df) > 0

    # in order to create a hard contrast set, we need at least 3 examples from one starting point
    # 4 examples if we bar examples with put / throw
    b_image_paths_set = set(df['before_image_path'])
    min_exs = 4 if prohibit_put_throw else 3
    for b_image_path in b_image_paths_set:
        mask = df.before_image_path.values == b_image_path
        n_exs = mask.sum()
        if share_held and share_scene and share_object and n_exs < min_exs:
            df = df[~mask]
    df.to_csv(filtered_metadata_path)

    b_image_paths = df['before_image_path']
    after_image_paths = df['after_image_path']

    combined_paths = list(zip(*[df.index.values, b_image_paths, after_image_paths]))
    random.shuffle(combined_paths)

    row_index = []
    first_distractor = []
    second_distractor = []

    # now, we're just going to iterate over all of the dataset and create contrast sets
    if n_imgs is None:
        n_imgs = len(df)
    for i, (after_image_index, before_image_path, after_image_path) in enumerate(tqdm(combined_paths)):
        if i >= n_imgs:
            break

        after_image_info = df.loc[after_image_index]

        # take samples of contrasting images from a dataset that excludes the target after image
        sample_df = df.drop([after_image_index], axis=0)

        # ... and meets whatever conditions we've specified
        if share_held:
            sample_df = sample_df[sample_df['held'] == after_image_info['held']]
        if share_scene:
            sample_df = sample_df[sample_df['scene'] == after_image_info['scene']]
        if share_object:
            if share_scene:
                sample_df = sample_df[sample_df['object_id'] == after_image_info['object_id']]
            else:
                sample_df = sample_df[sample_df['object_name'] == after_image_info['object_name']]
        if share_code:
            sample_df = sample_df[sample_df['code'] == after_image_info['code']]

        # we've chosen an after image; now we should remove any counterparts
        # that, like push / pull, sometimes look similar
        action = after_image_info['action_name']
        if prohibit_put_throw and action in opposite_action:
            cond = opposite_condition(sample_df, after_image_info)

            cs = cond.sum()
            if cs:
                if cs != 1:
                    print(f'Something went wrong')
                    print(sample_df[cond])
                    raise RuntimeError('opp df too big')
                sample_df = sample_df[~cond]

        # give up if there aren't enough images to take 2 samples
        if len(sample_df) < 2:
            continue

        # actually take the samples
        samples = []
        for _ in range(2):
            sample_id = random.sample(range(len(sample_df)), 1)
            [sample] = sample_df.index.values[sample_id]
            samples.append(sample)

            sample_image_info = sample_df.loc[sample]
            action = sample_image_info['action_name']
            sample_df = sample_df.drop([sample], axis=0)

            # we've chosen an after image; now we should remove any counterparts
            # that, like push / pull, sometimes look similar
            if prohibit_put_throw and action in opposite_action:
                cond = opposite_condition(sample_df, sample_image_info)

                cs = cond.sum()
                if cs:
                    if cs != 1:
                        print(f'Something went wrong')
                        print(sample_df[cond])
                        raise RuntimeError('opp df too big')
                    sample_df = sample_df[~cond]


        distractor1, distractor2 = samples
        row_index.append(after_image_index)
        first_distractor.append(distractor1)
        second_distractor.append(distractor2)

    dataset_df = df.loc[row_index].copy()
    for i, dist in enumerate([first_distractor, second_distractor]):
        distractor_df = df.loc[dist]
        dataset_df[f'distractor{i}_path'] = distractor_df['after_image_path'].values
        dataset_df[f'distractor{i}_object_name'] = distractor_df['object_name'].values
        dataset_df[f'distractor{i}_action_name'] = distractor_df['action_name'].values
        dataset_df[f'distractor{i}_held'] = distractor_df['held'].values
        dataset_df[f'distractor{i}_code'] = distractor_df['code'].values
        dataset_df[f'distractor{i}_scene'] = distractor_df['scene'].values

    # determine what order these images should be presented in
    orders = []
    correct_indices = []
    order = list(range(3))
    for _ in range(len(dataset_df)):
        random.shuffle(order)
        orders.append([x for x in order])
        correct_indices.append(order.index(0))

    dataset_df['order'] = orders
    dataset_df['correct_index'] = correct_indices

    # name and produce the 3-image contrast set csv
    filename = 'dataset' + suffix + '.csv'
    filtered_dataset_path = path / filename

    dataset_df = dataset_df.loc[:, ~dataset_df.columns.str.contains('^Unnamed')]
    for column_name in dataset_df.columns:
        if 'path' in column_name:
            dataset_df[column_name] = [Path(x).as_posix() for x in dataset_df[column_name]]
    dataset_df.to_csv(filtered_dataset_path, index=False)

    # now for adding a 4th image to the contrast set.
    augment_contrast_set(filtered_dataset_path, cleaned_df, dataset_df)

def opposite_condition(df, action_info):
    return  (df['action_name'] == opposite_action[action_info['action_name']]) & \
            (df['held'] == action_info['held']) & \
            (df['scene'] == action_info['scene']) & \
            (df['object_id'] == action_info['object_id']) & \
            (df['code'] == action_info['code'])

def augment_contrast_set(path, original_df, contrast_df, same_recep=False):
    path = Path(path)
    to_drop = []

    dist_attrs = defaultdict(list)
    for index, hs, code, scene, object, action, receptacle in tqdm(list(zip(contrast_df.index, contrast_df['held'], contrast_df['code'], contrast_df['scene'],
                                                contrast_df['object_name'], contrast_df['action_name'], contrast_df['receptacle_name']))):
        dist_df = original_df[(original_df['held'] == hs) & (original_df['scene'] != scene) &
                              (original_df['object_name'] == object) & (original_df['action_name'] == action)]

        if same_recep:
            dist_df = dist_df[dist_df['receptacle_name'] == receptacle]

        if not len(dist_df):
            to_drop.append(index)
            continue

        row_index = random.randrange(len(dist_df))
        row = dist_df.loc[dist_df.index[row_index]]
        for attr in ['after_image_path', 'object_name', 'action_name', 'held', 'code', 'scene']:
            dist_attrs[attr].append(row[attr])

    print(f'Dropped {len(to_drop)} out of {len(contrast_df)} examples.')

    filtered_df = contrast_df.drop(to_drop)
    for k, v in dist_attrs.items():
        if k == 'after_image_path':
            filtered_df['distractor2_path'] = v
        else:
            filtered_df['distractor2_' + k] = v

    did_recep = '_recep' if same_recep else ''
    four = list(range(4))
    def get_four():
        random.shuffle(four)
        return [x for x in four], four.index(0)
    order, indices = zip(*[get_four() for _ in range(len(filtered_df))])
    filtered_df['order_4'] = order
    filtered_df['correct_index_4'] = indices
    filtered_df_filename = path.parent / (path.stem + '_augmented' + did_recep + '.csv')
    print(filtered_df_filename)
    filtered_df.to_csv(filtered_df_filename)


if __name__ == '__main__':
    collate_datasets(path)
    generate_contrast_set(path=path, n_imgs=None, share_held=True, share_scene=True, share_object=True,
                          prohibit_put_throw=True, share_code=True)