import random
from collections import defaultdict

import jsonlines
import numpy as np
import pandas as pd
from ai2thor.controller import Controller

from controller_wrapper import ControllerWrapper
from text_descriptions import generate_sentence, generate_detailed_sentence
from utils import make_name_to_action_fn, name_to_reqd_attr, action_names, name_to_idx, idx_to_name

np.random.seed(42)

controller = ControllerWrapper(Controller())
scenes = controller.controller.ithor_scenes()
name_to_action_fn = make_name_to_action_fn(controller)

def filter_objects_on_attributes(objects, attrs):
    #all_objects = controller.get_metadata()['objects']
    objects = [obj for obj in objects if all(obj[attr] for attr in attrs)]
    return objects

def one_scene_loop(filter_objects, generate_actions_list_for_object, path, scene):
    """
    
    :param filter_objects: a function from List[obj]->List[obj], returning the list of objects that you want to perform
            actions on.
    :param generate_actions_list_for_object: a function that takes in an object, and returns a list of 5-tuples containing:
        - prepare_env: a function saying how to prepare the environment. This includes picking up the object / setting it
            down, but also setting its state to open / closed, filled / empty
        - action_fn: a function that actually performs the action in question
        - action_name: the name of the action
        - filename: the name by which to identify the before image of this action
        - attributes: a dictionary containing useful information 
    :param path: 
    :param scene: 
    :return: 
    """
    path = path / scene
    if not path.exists():
        path.mkdir()
    controller.reset(scene=scene)
    warp_location, rots, receptacle_id = controller.warp_near_largest_receptacle()
    objects = controller.get_metadata()['objects']
    objects = filter_objects(objects)

    with jsonlines.open(path / 'objects.jsonl', mode='w') as writer:
        for obj in objects:
            writer.write(obj)

    d = defaultdict(list)

    for i, obj in enumerate(objects):
        objid = obj['objectId']
        controller.objid = objid
        actions_to_try = generate_actions_list_for_object(obj)

        save_before = True
        for prepare_env, action_fn, action_name, filename, attributes in actions_to_try:
            # record some basic info
            d['scene'].append(scene)
            object_name = obj['objectType']
            d['object_name'].append(object_name)
            d['object_id'].append(objid)
            d['code'].append(filename)
            d['held'].append(attributes['held'])

            action_name = attributes['action_name']
            action_idx = name_to_idx[action_name]
            d['action_name'].append(action_name)
            d['action_id'].append(action_idx)

            # prepare the environment and record frame, image, bbox
            controller.reset(scene=scene)
            prepare_env(controller, objid, warp_location, rots, receptacle_id)
            controller.receptacle_id = receptacle_id
            recep_name = controller.get_object_by_id(receptacle_id)['objectType']
            d['receptacle_name'].append(recep_name)

            before_frame = controller.get_frame()

            before_image = controller.get_image()
            before_image_name = f'{i}_{filename}_0.png'
            before_image_path = path / before_image_name
            d['before_image_path'].append(str(before_image_path))
            if save_before:
                before_image.save(before_image_path)
            d['before_image_name'].append(before_image_name)

            bounding_box = controller.get_bounding_box(objid)
            d['before_image_bb'].append(bounding_box)


            # take the action, and record effects
            success = action_fn(objid)
            d['success'].append(success)

            after_frame = controller.get_frame()
            visible_difference = np.sum(np.abs(after_frame - before_frame))
            d['visible_difference'].append(visible_difference)

            after_image = controller.get_image()
            after_image_name = f'{i}_{filename}_{action_idx}.png'
            d['after_image_name'].append(after_image_name)
            after_image_path = path / after_image_name
            d['after_image_path'].append(str(after_image_path))
            after_image.save(after_image_path)


            bounding_box = controller.get_bounding_box(objid)
            d['after_image_bb'].append(bounding_box)

            obj_post_action = controller.get_object_by_id(objid)
            d['visible'].append(obj_post_action['visible'])

            sentence = generate_sentence(object_name, action_name)
            d['sentence'].append(sentence)
            parent_receps = obj_post_action['parentReceptacles']
            if not parent_receps or receptacle_id in parent_receps:
                end_recep = ''
            else:
                end_recep = controller.get_object_by_id(parent_receps[0])['objectType']
            detailed_sentence = generate_detailed_sentence(action_name, not attributes['held'], recep_name, end_recep, object_name)
            d['detailed_sentence'].append(detailed_sentence)

    df = pd.DataFrame.from_dict(d)
    df.to_csv(path / 'metadata.csv')


def held_setup(objid, warp_location, rots, receptacle_id):
    controller.force_pick_up(objid)
    controller.teleport(warp_location)
    controller.clear(receptacle_id)
    if receptacle_id is not None:
        larec = controller.get_object_by_id(receptacle_id)
        if larec['position']['y'] < 0.25:
            controller.controller.step("LookDown", degrees=30)
    for _ in range(rots):
        controller.rotate()


def not_held_setup(objid, warp_location, rots, receptacle_id):
    controller.force_pick_up(objid)
    controller.teleport(warp_location)
    controller.clear(receptacle_id)
    if receptacle_id is not None:
        larec = controller.get_object_by_id(receptacle_id)
        if larec['position']['y'] < 0.25:
            controller.controller.step("LookDown", degrees=30)
    for _ in range(rots):
        controller.rotate()
    controller.put_careful()

def wrap_setup_fns(fns):
    def prep(controller, objid, warp_location, rots, receptacle_id):
        if not fns:
            return
        f0 = fns[0]
        f0(objid, warp_location, rots, receptacle_id)
        for fn in fns[1:]:
            fn(objid)
    return prep

def filter_actions_on_object(actions, obj):
    return sorted([action for action in actions if obj[name_to_reqd_attr[action]]])

def generate_actions_list(obj):
    # prepare_env, action_fn, action_name, filename, attributes
    output_tuples = []
    for held_state in ['held', 'not_held']:
        if held_state == 'held':
            actions_to_try = ['drop', 'throw', 'put', 'open', 'close', 'dirty', 'fill', 'toggle', 'useUp', 'break',
                              'cook']
            actions_to_try = filter_actions_on_object(actions_to_try, obj)
            attribute_dicts = [{'action_name': action_name, 'held': True} for action_name in actions_to_try]
        else:
            actions_to_try = ['pickUp', 'push', 'pull', 'open', 'close', 'dirty', 'fill', 'toggle', 'useUp', 'break',
                              'cook', 'slice']
            actions_to_try = filter_actions_on_object(actions_to_try, obj)
            attribute_dicts = [{'action_name': action_name, 'held': False} for action_name in actions_to_try]

        if obj['openable']:
            attribute_dicts = [{'open': b, **adict} for b in [True, False] for adict in attribute_dicts]
        else:
            for adict in attribute_dicts:
                adict['open'] = None

        if obj['dirtyable']:
            attribute_dicts = [{'dirty': b, **adict} for b in [True, False] for adict in attribute_dicts]
        else:
            for adict in attribute_dicts:
                adict['dirty'] = None

        if obj['canFillWithLiquid']:
            attribute_dicts = [{'filled': b, **adict} for b in [True, False] for adict in attribute_dicts]
        else:
            for adict in attribute_dicts:
                adict['filled'] = None

        if obj['toggleable']:
            attribute_dicts = [{'toggled': b, **adict} for b in [True, False] for adict in attribute_dicts]
        else:
            for adict in attribute_dicts:
                adict['toggled'] = None

        for adict in attribute_dicts:
            prepare_env, filepath = gen_prep_and_filepath(adict)
            action_name = adict['action_name']
            action_fn = name_to_action_fn[action_name]

            output_tuples.append((prepare_env, action_fn, action_name, filepath, adict))

    return output_tuples

def dummy(*args):
    return True

def gen_prep_and_filepath(adict):
    prep_fns = []
    fp = ""
    if adict['held']:
        prep_fns.append(held_setup)
        fp = fp + '1'
    else:
        prep_fns.append(not_held_setup)
        fp = fp + '0'


    true_fns = [controller.open, controller.dirty, controller.fill, controller.toggle]
    false_fns = [controller.close, dummy, controller.empty, controller.toggle]
    for i, attr in enumerate(['open', 'dirty', 'filled', 'toggled']):
        if adict[attr] is None:
            fp = fp + '0'
        elif adict[attr]:
            prep_fns.append(true_fns[i])
            fp = fp + '1'
        else:
            prep_fns.append(false_fns[i])
            fp = fp + '0'
    return wrap_setup_fns(prep_fns), fp

def filter_pickupable(objects):
    return filter_objects_on_attributes(objects, ['pickupable'])

def do_one_scene_all(scene, path):
    one_scene_loop(filter_pickupable, generate_actions_list, path, scene)
