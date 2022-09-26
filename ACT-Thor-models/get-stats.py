import pandas
from pprint import pprint


def get_nocook_dataset(pth):
    df = pandas.read_csv(pth, index_col=0)
    df.drop(index=df[(df['action_name'] == 'cook') |
                     (df['distractor0_action_name'] == 'cook') |
                     (df['distractor1_action_name'] == 'cook') |
                     (df['distractor2_action_name'] == 'cook')].index, inplace=True)
    return df


def rework_split_cols(df):
    new_df = df.copy()
    new_df.loc[lambda d: d['object_split'] != 'test', 'object_split'] = 'Seen'
    new_df.loc[lambda d: d['object_split'] == 'test', 'object_split'] = 'Unseen'
    new_df.loc[lambda d: d['scene_split'] != 'test', 'scene_split'] = 'Seen'
    new_df.loc[lambda d: d['scene_split'] == 'test', 'scene_split'] = 'Unseen'
    return new_df


env_dict = {
    **{str(k): 'kitchen' for k in range(1,31)},
    **{str(k): 'livingroom' for k in range(200, 231)},
    **{str(k): 'bedroom' for k in range(300, 331)},
    **{str(k): 'bathroom' for k in range(400, 431)},
}


def get_before_stats(pth):
    res = {}

    df = get_nocook_dataset(pth)
    res['Unique before images'] = len(set(df['before_image_path']))

    obj_indexer = ['object_split', 'object_name', 'action_name', 'before_image_path']
    obj_split_df = df[obj_indexer]
    obj_split_df.loc[lambda d: d['object_split'] != 'test', 'object_split'] = 'Seen'
    obj_split_df.loc[lambda d: d['object_split'] == 'test', 'object_split'] = 'Unseen'
    res['Unique before per split (object)'] = obj_split_df.groupby(obj_indexer[0]).nunique()['before_image_path']
    print(res['Unique before per split (object)'].sum())
    res['Unique before per item (object)'] = obj_split_df.groupby(obj_indexer[:-2]).nunique()['before_image_path']
    print(res['Unique before per item (object)'].sum())

    scene_indexer = ['scene_split', 'scene', 'action_name', 'before_image_path']
    scene_split_df = df[scene_indexer]
    scene_split_df.loc[lambda d: d['scene_split'] != 'test', 'scene_split'] = 'Seen'
    scene_split_df.loc[lambda d: d['scene_split'] == 'test', 'scene_split'] = 'Unseen'

    res['Unique before per split (scene)'] = scene_split_df.groupby(scene_indexer[0]).nunique()['before_image_path']
    print(res['Unique before per split (scene)'].sum())

    scene_split_df['environment'] = scene_split_df['scene'].map(lambda el: env_dict[el.split("_")[0].replace("FloorPlan", "")])
    res['Unique before per environment (scene)'] = scene_split_df.groupby(['scene_split', 'environment']).nunique()['before_image_path']
    print(res['Unique before per environment (scene)'].sum())

    return res


def get_contrast_stats(pth):
    res = {}

    df = get_nocook_dataset(pth)
    df = rework_split_cols(df)

    df['contrast_set_3'] = pandas.Series([
        tuple(sorted([t1, t2, t3])) for t1, t2, t3 in zip(df['after_image_path'], df['distractor0_path'], df['distractor1_path'])
    ], index=df.index)
    df['contrast_set_4'] = pandas.Series([
        tuple(sorted([t1, t2, t3, t4])) for t1, t2, t3, t4 in zip(df['after_image_path'], df['distractor0_path'], df['distractor1_path'], df['distractor2_path'])
    ], index=df.index)
    df['environment'] = df['scene'].map(lambda el: env_dict[el.split("_")[0].replace("FloorPlan", "")])

    obj_indexer = ['object_split', 'object_name', 'contrast_set_3', 'contrast_set_4']
    scene_indexer = ['scene_split', 'environment', 'contrast_set_3', 'contrast_set_4']

    res['Unique contrasts (3) by object'] = df[obj_indexer].groupby(obj_indexer[:-2]).nunique()['contrast_set_3']
    res['Unique contrasts (3) by object split'] = df[obj_indexer].groupby(obj_indexer[0]).nunique()['contrast_set_3']
    res['Unique contrasts (4) by object'] = df[obj_indexer].groupby(obj_indexer[:-2]).nunique()['contrast_set_4']
    res['Unique contrasts (4) by object split'] = df[obj_indexer].groupby(obj_indexer[0]).nunique()['contrast_set_4']
    res['Total (objs)'] = df[obj_indexer].groupby(obj_indexer[0]).nunique().sum()

    res['Unique contrasts (3) by environment'] = df[scene_indexer].groupby(scene_indexer[:-2]).nunique()['contrast_set_3']
    res['Unique contrasts (3) by scene split'] = df[scene_indexer].groupby(scene_indexer[0]).nunique()['contrast_set_3']
    res['Unique contrasts (4) by environment'] = df[scene_indexer].groupby(scene_indexer[:-2]).nunique()['contrast_set_4']
    res['Unique contrasts (4) by scene split'] = df[scene_indexer].groupby(scene_indexer[0]).nunique()['contrast_set_4']
    res['Total (scenes)'] = df[scene_indexer].groupby(scene_indexer[0]).nunique().sum()

    res['Unique distractors'] = df['distractor2_path'].nunique()

    res['Total'] = df[['contrast_set_3', 'contrast_set_4']].nunique()

    return res


def get_objects_per_action(pth):
    # Results:
    #       stdconfig: Laptop, Book, Cup
    #       alternate: Kettle, Box, Candle, Bowl
    df = get_nocook_dataset(pth)
    print(*list({k: set(df[df['action_name'] == k].object_name) for k in set(df['action_name'])}.items()), sep='\n')


if __name__ == '__main__':
    from pathlib import Path
    from visual_features.data import __default_dataset_path__
    from visual_features.data import __default_dataset_fname__
    pth = str(Path(__default_dataset_path__, __default_dataset_fname__))

    # res = get_before_stats(pth)
    res = get_contrast_stats(pth)
    pprint(res)


