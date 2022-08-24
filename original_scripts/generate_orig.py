from argparse import ArgumentParser
from pathlib import Path
import random
from collections import defaultdict

import jsonlines
import numpy as np
import pandas as pd
from ai2thor.controller import Controller

from controller_wrapper import ControllerWrapper
from text_descriptions import generate_sentence

np.random.seed(42)

controller = ControllerWrapper(Controller())
scenes = controller.controller.ithor_scenes()

parser = ArgumentParser()

parser.add_argument('-c', '--contrast_set', required=True)
parser.add_argument('-s', '--scene', default=None)
parser.add_argument('-si', '--start_index', default=None, type=int)
parser.add_argument('-ei', '--end_index', default=None, type=int)
parser.add_argument('-p', '--path', type=str, default='data')
parser.add_argument('-m', '--max_images', default=None, type=int)


def filter_objects(attrs):
    all_objects = controller.get_metadata()['objects']
    pickupable_objects = [obj for obj in all_objects if all(obj[attr] for attr in attrs)]
    return pickupable_objects

def get_bounding_boxes(max_bboxes):
    random.seed(42)
    path = Path('data/bboxes')

    bbox_scenes = [s for s in scenes]
    random.shuffle(bbox_scenes)

    get_obj_fn = pickupable_initialize
    prep_fn = held_setup
    actions_to_try = ['drop', 'throw', 'put', 'open', 'dirty', 'fill', 'toggle', 'useUp', 'break', 'cook']

    ins, oids, bbxs = [], [], []
    for scene in bbox_scenes:
        image_names, object_ids, bounding_boxes = get_bounding_boxes_scene(get_obj_fn, prep_fn, actions_to_try, path, scene)
        ins += image_names
        oids += object_ids
        bbxs += bounding_boxes

        if len(bbxs) >= max_bboxes:
            break

    indices = list(range(len(bbxs)))
    random.shuffle(indices)
    val_index = int(0.8 * len(indices))
    test_index = int(0.9 * len(indices))

    train_indices = indices[:val_index]
    val_indices = indices[val_index:test_index]
    test_indices = indices[test_index:]

    split = ['' for _ in range(len(bbxs))]

    d = {'image_name': ins, 'object_id': oids, 'bbox': bbxs, 'split': split}
    df = pd.DataFrame.from_dict(d)
    df.loc[train_indices,'split'] = 'train'
    df.loc[val_indices,'split'] = 'valid'
    df.loc[test_indices,'split'] = 'test'

    df.to_csv(path/'labels.csv')

    for split_name, idxs in zip(['train', 'valid', 'test'], [train_indices, val_indices, test_indices]):
        split_path = path / split_name
        split_path.mkdir(exist_ok=True)
        split_df = df.iloc[idxs]
        for fname in split_df['image_name']:
            fpath = path / fname
            fpath.rename(split_path / fname)

        split_df.to_csv(split_path / 'labels.csv')



def get_bounding_boxes_scene(get_obj_fn, prep_fn, actions_to_try, path, scene):
    controller.reset(scene=scene)
    objects, warp_location, rots, receptacle = get_obj_fn()
    print('chose receptacle', receptacle)
    controller.receptacle_id = receptacle

    objids = [obj['objectId'] for obj in objects]
    objid_dict = {obj['objectId']:obj for obj in objects}

    image_names, object_ids, bounding_boxes = [], [], []

    for i, objid in enumerate(objids):
        controller.objid = objid
        for action_name in actions_to_try:
            required_attr = name_to_reqd_attr[action_name]
            has_attr = objid_dict[objid][required_attr]
            if not has_attr:
                continue

            controller.reset(scene=scene)
            prep_fn(objid, warp_location, rots, receptacle)
            success = name_to_action_fn[action_name](objid)

            if not success:
                continue

            bounding_box_dict = controller.controller.last_event.instance_detections2D
            if objid not in bounding_box_dict:
                continue
            bounding_box = bounding_box_dict[objid]
            bounding_boxes.append(bounding_box)
            after_image = controller.get_image()
            action_idx = name_to_idx[action_name]
            after_image_name = f'{scene}_{i}_{action_idx}.png'
            image_names.append(after_image_name)
            after_image.save(path / after_image_name)

            object_ids.append(objid)
    return image_names, object_ids, bounding_boxes



def outer_loop(get_obj_fn, prep_fn, actions_to_try, path, max_images=None, scene=None, is_placed=False):
    controller.reset(scene=scene)
    objects, warp_location, rots, receptacle = get_obj_fn()
    print('chose receptacle', receptacle)
    controller.receptacle_id = receptacle
    recep_name = controller.get_object_by_id(receptacle)['objectType']

    with jsonlines.open(path / 'objects.jsonl', mode='w') as writer:
        for obj in objects:
            writer.write(obj)

    objids = [obj['objectId'] for obj in objects]
    objids = objids[:max_images] if max_images else objids
    objid_dict = {obj['objectId']:obj for obj in objects}

    d = defaultdict(list)

    for i, objid in enumerate(objids):
        controller.objid = objid
        save_before = True
        for action_name in actions_to_try:
            required_fn = name_to_reqd_fn[action_name]
            has_attr = required_fn(objid_dict[objid])
            # print(objid, action_name, has_attr, objid_dict[objid]['pickupable'])
            if not has_attr:
                continue

            controller.reset(scene=scene)

            prep_fn(objid, warp_location, rots, receptacle)
            before_frame = controller.get_frame()
            before_image = controller.get_image()

            bounding_box_dict = controller.controller.last_event.instance_detections2D
            if objid in bounding_box_dict:
                bounding_box = bounding_box_dict[objid]
            else:
                bounding_box = (0, 0, 0, 0)
            d['before_image_bb'].append(bounding_box)

            success = name_to_action_fn[action_name](objid)
            after_frame = controller.get_frame()
            after_image = controller.get_image()

            bounding_box_dict = controller.controller.last_event.instance_detections2D
            if objid in bounding_box_dict:
                bounding_box = bounding_box_dict[objid]
            else:
                bounding_box = (0, 0, 0, 0)
            d['after_image_bb'].append(bounding_box)

            if save_before:
                before_image.save(path / f'{i}_0.png')
                save_before = False

            d['success'].append(success)
            object_metadata = controller.get_object_by_id(objid)
            d['visible'].append(object_metadata['visible'])
            visible_difference = np.sum(np.abs(after_frame - before_frame))
            d['visible_difference'].append(visible_difference)

            object_name = objid_dict[objid]['objectType']
            d['object_name'].append(object_name)
            d['object_id'].append(objid)

            parent_receps = object_metadata['parentReceptacles']
            if not parent_receps or receptacle in parent_receps:
                end_recep = ''
            else:
                end_recep = controller.get_object_by_id(parent_receps[0])['objectType']

            if is_placed or action_name in 'put':
                sent_recep = recep_name
            elif action_name == 'throw' or action_name == 'drop':
                sent_recep = end_recep
            else:
                sent_recep = ''

            if action_name == 'fill' or action_name == 'empty':
                other = 'wine'
            elif action_name == 'toggle':
                other = 'on'
            elif action_name == 'push' or action_name == 'pull':
                other = end_recep
            else:
                other = ''
            sentence = generate_sentence(object_name, action_name, receptacle=sent_recep, other=other)
            d['sentence'].append(sentence)
            d['sent_recep'].append(sent_recep)
            d['sent_other'].append(other)

            action_idx = name_to_idx[action_name]
            d['action_name'].append(action_name)
            d['action_id'].append(action_idx)

            after_image_name = f'{i}_{action_idx}.png'
            d['image_name'].append(after_image_name)
            d['before_image_name'].append(f'{i}_0.png')

            after_image.save(path / after_image_name)

    recep_names = [recep_name for _ in d['image_name']]
    d['receptacle_name'] = recep_names

    df = pd.DataFrame.from_dict(d)
    df.to_csv(path / 'metadata.csv')

name_to_action_fn = {
    'pickUp': controller.force_pick_up,
    'drop': controller.drop,
    'throw': controller.throw,
    'put': controller.put_careful,
    'push': controller.push,
    'pull': controller.pull,
    'open': controller.open,
    'close': controller.close,
    'slice': controller.slice,
    'dirty': controller.dirty,
    'fill': controller.fill,
    'empty': controller.empty,
    'toggle': controller.toggle,
    'useUp': controller.use_up,
    'break': controller.destroy,
    'cook': controller.cook
}

name_to_reqd_attr = {
    'pickUp': 'pickupable',
    'drop': 'pickupable',
    'throw': 'pickupable',
    'put': 'pickupable',
    'push': 'pickupable', # actually moveable, but let's worry about that later
    'pull':'pickupable', # ditto
    'open':'openable',
    'close':'openable',
    'slice':'sliceable',
    'dirty':'dirtyable',
    'fill':'canFillWithLiquid',
    'empty':'canFillWithLiquid',
    'toggle': 'toggleable',
    'useUp': 'canBeUsedUp',
    'break': 'breakable',
    'cook': 'cookable'
}

def requires_attribute(attr):
    def req(d):
        return d[attr]
    return req

name_to_reqd_fn = {name: requires_attribute(attr) for name, attr in name_to_reqd_attr.items()}

name_to_reqd_fn['open'] = lambda d: d['openable'] and not d['isOpen']
name_to_reqd_fn['close'] = lambda d: d['openable'] and d['isOpen']

action_names = ['drop','throw','put','push','pull','open','close','slice',
                'dirty','fill','empty','toggle','useUp','break','cook','pickUp']
name_to_idx = {name: i+1 for i, name in enumerate(action_names)}
# {'drop': 1, 'throw': 2, 'put': 3, 'push': 4, 'pull': 5,
# 'open': 6, 'close': 7, 'slice': 8, 'dirty': 9, 'fill': 10,
# 'empty': 11, 'toggle': 12, 'useUp': 13, 'break': 14, 'cook': 15}
idx_to_name = {idx: name for name, idx in name_to_idx.items()}
# {1: 'drop', 2: 'throw', 3: 'put', 4: 'push', 5: 'pull', 6:
# 'open', 7: 'close', 8: 'slice', 9: 'dirty', 10: 'fill', 11:
# 'empty', 12: 'toggle', 13: 'useUp', 14: 'break', 15: 'cook'}


def pickupable_initialize():
    objects = filter_objects(['pickupable'])
    warp_location, rots, larec_id = controller.warp_near_largest_receptacle()
    return objects, warp_location, rots, larec_id


def held_setup(objid, warp_location, rots, larec_id):
    controller.force_pick_up(objid)
    controller.teleport(warp_location)
    controller.clear(larec_id)
    if larec_id is not None:
        larec = controller.get_object_by_id(larec_id)
        if larec['position']['y'] < 0.25:
            controller.controller.step("LookDown", degrees=30)
    for _ in range(rots):
        controller.rotate()


def not_held_setup(objid, warp_location, rots, larec_id):
    controller.force_pick_up(objid)
    controller.teleport(warp_location)
    controller.clear(larec_id)
    if larec_id is not None:
        larec = controller.get_object_by_id(larec_id)
        if larec['position']['y'] < 0.25:
            controller.controller.step("LookDown", degrees=30)
    for _ in range(rots):
        controller.rotate()
    controller.put_careful()


def do_one_scene(scene, max_images, contrast_set, path):
    if contrast_set == 'pickupable-held':
        get_obj_fn = pickupable_initialize
        prep_fn = held_setup
        actions_to_try = ['drop','throw','put','open','close', 'dirty','fill','toggle','useUp','break','cook']
        is_placed = False

    elif contrast_set == 'pickupable-not-held':
        get_obj_fn = pickupable_initialize
        prep_fn = not_held_setup
        actions_to_try = ['pickUp', 'push', 'pull', 'open', 'close', 'dirty', 'fill', 'toggle', 'useUp', 'break', 'cook', 'slice']
        is_placed = True
    elif contrast_set == 'movable':
        raise ValueError('not yet implemented')
    else:
        raise ValueError('invalid contrast set')

    full_path = path / contrast_set / scene
    if not full_path.exists():
        full_path.mkdir(parents=True)
    outer_loop(get_obj_fn, prep_fn, actions_to_try, full_path, max_images, scene, is_placed=is_placed)

if __name__ == '__main__':
    # default scene is 'FloorPlan10_physics'
    args = parser.parse_args()
    scene = args.scene
    max_images = args.max_images
    contrast_set = args.contrast_set
    path = Path(args.path)

    if scene == 'all':
        start_index = 0 if args.start_index is None else args.start_index
        end_index = len(scenes) if args.end_index is None else args.end_index
        for s in scenes[start_index:end_index]:
            print('doing scene', s)
            do_one_scene(s, max_images, contrast_set, path)

    else:
        if scene.isdigit():
            scene = scenes[int(scene)]
        if scene is None:
            scene = scenes[0]
        assert isinstance(scene, str), f'got bad scene ({scene}) of type {type(scene)}'
        do_one_scene(scene, max_images, contrast_set, path)