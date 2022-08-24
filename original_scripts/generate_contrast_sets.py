from pathlib import Path
from collections import defaultdict
import random

from tqdm import tqdm
import pandas as pd
import numpy as np

random.seed(42)

# enforce put / throw constraint

opposite_action = {'put': 'throw', 'throw': 'put', 'push': 'pull', 'pull': 'push'}

kitchens = [f"FloorPlan{i}_physics" for i in range(1, 31)]
living_rooms = [f"FloorPlan{200 + i}_physics" for i in range(1, 31)]
bedrooms = [f"FloorPlan{300 + i}_physics" for i in range(1, 31)]
bathrooms = [f"FloorPlan{400 + i}_physics" for i in range(1, 31)]
scene_lists = [kitchens, living_rooms, bedrooms, bathrooms]
train_scenes, valid_scenes, test_scenes = set(), set(), set()
for scene_list in scene_lists:
    train_scenes.update(scene_list[:20])
    valid_scenes.update(scene_list[20:25])
    test_scenes.update(scene_list[25:])
scenes = kitchens + living_rooms + bedrooms + bathrooms
scene_map = {scene: 'train' if scene in train_scenes else ('valid' if scene in valid_scenes else 'test') for scene in scenes}


def generate_contrast_set(path='new-data', n_imgs=None, share_cs=True, share_scene=True, share_object=True, prohibit_put_throw=True, force_remake=False):
    path = Path(path)
    filename = 'metadata_filtered'
    if share_cs:
        filename += '_cs'
    if share_scene:
        filename += '_scene'
    if share_object:
        filename += '_object'
    if prohibit_put_throw:
        filename += '_nopt'
    filename += '.csv'
    filtered_metadata_path = path / filename
    if filtered_metadata_path.exists() and not force_remake:
        df = pd.read_csv(filtered_metadata_path)
        b_image_paths = [path / cs / scene / bi for cs, scene, bi in
                         zip(df['contrast_set'], df['scene'], df['before_image_name'])]
        df['before_image_path'] = b_image_paths

    else:
        df = pd.read_csv(path / 'metadata.csv')
        df = df[df['success'] == True]

        visible = (df['visible'] == True) | (df['action_name'] == 'break')
        visible_diff = df['visible_difference'] != 0
        print('invisible', len(df) - np.sum(visible), '/', len(df))
        print('no diff', np.sum(~visible_diff), '/', len(df))
        df = df[visible & visible_diff]

        assert len(df) > 0

        b_image_paths = [path / cs / scene / bi for cs, scene, bi in zip(df['contrast_set'], df['scene'],
                                                                         df['before_image_name'])]
        df['before_image_path'] = b_image_paths

        # filtering
        b_image_paths_set = set(b_image_paths)
        min_exs = 4 if prohibit_put_throw else 3
        for b_image_path in b_image_paths_set:
            mask = df.before_image_path.values == b_image_path
            n_exs = mask.sum()
            if share_cs and share_scene and share_object and n_exs < min_exs:
                df = df[~mask]

        df.to_csv(filtered_metadata_path)

    b_image_paths = [path / cs / scene / bi for cs, scene, bi in zip(df['contrast_set'], df['scene'],
                                                                     df['before_image_name'])]
    after_image_paths = [path / cs / scene / bi for cs, scene, bi in
                         zip(df['contrast_set'], df['scene'], df['image_name'])]
    df['after_image_path'] = after_image_paths

    combined_paths = list(zip(*[df.index.values, b_image_paths, after_image_paths]))
    random.shuffle(combined_paths)

    row_index = []
    first_distractor = []
    second_distractor = []
    # sampling
    if n_imgs is None:
        n_imgs = len(df)
    for i, (after_image_index, before_image_path, after_image_path) in enumerate(tqdm(combined_paths)):
        if i >= n_imgs:
            break

        after_image_info = df.loc[after_image_index]
        sample_df = df.drop([after_image_index], axis=0)

        if share_cs:
            sample_df = sample_df[sample_df['contrast_set'] == after_image_info['contrast_set']]
        if share_scene:
            sample_df = sample_df[sample_df['scene'] == after_image_info['scene']]
        if share_object:
            if share_scene:
                sample_df = sample_df[sample_df['object_id'] == after_image_info['object_id']]
            else:
                # changed from object_id to object_name; let's see if this causes problems
                sample_df = sample_df[sample_df['object_name'] == after_image_info['object_name']]


        action = after_image_info['action_name']
        if prohibit_put_throw and action in opposite_action:
            opp = opposite_action[action]
            opp_df = sample_df[(sample_df['action_name'] == opp) &
                               (sample_df['contrast_set'] == after_image_info['contrast_set']) &
                               (sample_df['scene'] == after_image_info['scene']) &
                               (sample_df['object_id'] == after_image_info['object_id'])]
            if len(opp_df):
                if len(opp_df) != 1:
                    print(f'Something went wrong')
                    print(opp_df['after_image_path'])
                    print(opp_df['object_name'])
                    raise RuntimeError('opp df too big')
                opp_df_index = opp_df.index
                sample_df = sample_df.drop(opp_df_index, axis=0)
        samples = []

        if len(sample_df) < 2:
            continue
        for _ in range(2):
            sample_id = random.sample(range(len(sample_df)), 1)
            [sample] = sample_df.index.values[sample_id]
            samples.append(sample)
            #print(sample_df.index.values)
            #print(sample_id, sample)

            sample_image_info = sample_df.loc[sample]
            action = sample_image_info['action_name']
            sample_df = sample_df.drop([sample], axis=0)
            if prohibit_put_throw and action in opposite_action:
                opp = opposite_action[action]
                #print(sample_df['object_name'])
                #print(sample_image_info['object_name'])
                #print(sample_df['object_name'] == sample_image_info['object_name'])
                opp_df = sample_df[(sample_df['action_name'] == opp) &
                                   (sample_df['contrast_set'] == sample_image_info['contrast_set']) &
                                   (sample_df['scene'] == sample_image_info['scene']) &
                                   (sample_df['object_id'] == sample_image_info['object_id'])]
                if len(opp_df):
                    if len(opp_df) != 1:
                        print(f'Something went wrong')
                        print(opp_df['after_image_path'])
                        raise RuntimeError('opp df too big')
                    opp_df_index = opp_df.index
                    sample_df = sample_df.drop(opp_df_index, axis=0)


        distractor1, distractor2 = samples
        row_index.append(after_image_index)
        first_distractor.append(distractor1)
        second_distractor.append(distractor2)

    dataset_df = df.loc[row_index].copy()
    after_image_paths_unshuffled = df['after_image_path']
    for i, dist in enumerate([first_distractor, second_distractor]):
        distractor_df = df.loc[dist]
        dataset_df[f'distractor{i}_path'] = distractor_df['after_image_path'].values
        dataset_df[f'distractor{i}_object_name'] = distractor_df['object_name'].values
        dataset_df[f'distractor{i}_action_name'] = distractor_df['action_name'].values
        dataset_df[f'distractor{i}_contrast_set'] = distractor_df['contrast_set'].values
        dataset_df[f'distractor{i}_scene'] = distractor_df['scene'].values

    orders = []
    correct_indices = []
    order = list(range(3))
    for _ in range(len(dataset_df)):
        random.shuffle(order)
        orders.append([x for x in order])
        correct_indices.append(order.index(0))

    dataset_df['order_3'] = orders
    dataset_df['correct_index_3'] = correct_indices

    filename = 'dataset'
    if share_cs:
        filename += '_cs'
    if share_scene:
        filename += '_scene'
    if share_object:
        filename += '_object'
    if prohibit_put_throw:
        filename += '_nopt'
    filename += '.csv'
    filtered_dataset_path = path / filename

    dataset_df = dataset_df.loc[:, ~dataset_df.columns.str.contains('^Unnamed')]

    for column_name in dataset_df.columns:
        if 'path' in column_name:
            dataset_df[column_name] = [Path(x).as_posix() for x in dataset_df[column_name]]

    dataset_df.to_csv(filtered_dataset_path, index=False)


def augment_contrast_set(path, cs_df, same_recep=False):
    path = Path(path)
    df_path = path / cs_df
    contrast_df = pd.read_csv(df_path)
    original_df = pd.read_csv(path / 'metadata.csv')
    original_df = original_df[original_df['success'] == True]
    original_df = original_df[original_df['action_name'] != 'cook']
    to_drop = []

    print(path / (df_path.name + '_augmented' + '.csv'))

    dist_attrs = defaultdict(list)

    for index, cs, scene, object, action, receptacle in tqdm(list(zip(contrast_df.index, contrast_df['contrast_set'], contrast_df['scene'],
                                                contrast_df['object_name'], contrast_df['action_name'], contrast_df['receptacle_name']))):
        dist_df = original_df[(original_df['contrast_set'] == cs) & (original_df['scene'] != scene) &
                              (original_df['object_name'] == object) & (original_df['action_name'] == action)]

        if same_recep:
            dist_df = dist_df[dist_df['receptacle_name'] == receptacle]

        if not len(dist_df):
            to_drop.append(index)
            continue

        row_index = random.randrange(len(dist_df))
        row = dist_df.loc[dist_df.index[row_index]]
        for attr in ['after_image_path', 'object_name', 'action_name', 'contrast_set', 'scene']:
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

    train_count = round(len(filtered_df) * 0.64)
    valid_count = round(len(filtered_df) * 0.16)
    test_count = len(filtered_df) - train_count - valid_count
    split_labels = ['train'] * train_count + ['valid'] * valid_count + ['test'] * test_count
    random.shuffle(split_labels)
    filtered_df['sample_split'] = split_labels
    scene_split = []
    for sc in filtered_df['scene']:
        scene_split.append(scene_map[sc])
    filtered_df['scene_split'] = scene_split

    filtered_df.to_csv(path / (df_path.stem + '_augmented' + did_recep + '.csv'))


if __name__ == '__main__':
    generate_contrast_set(path='data-improved-descriptions', n_imgs=None, share_cs=True, share_scene=True, share_object=True, prohibit_put_throw=True)
    augment_contrast_set('data-improved-descriptions', 'dataset_cs_scene_object_nopt.csv', same_recep=True)
