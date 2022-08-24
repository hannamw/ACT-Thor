from argparse import ArgumentParser
from pathlib import Path

from dataset import scenes, do_one_scene_all

parser = ArgumentParser()

parser.add_argument('-s', '--scene', default=None)
parser.add_argument('-si', '--start_index', default=None, type=int)
parser.add_argument('-ei', '--end_index', default=None, type=int)
parser.add_argument('-p', '--path', type=str, default='data')


args = parser.parse_args()  # ['--contrast_set', 'pickupable-not-held', '--scene', '11', '-p', 'new-data'])
scene = args.scene
path = Path(args.path)

if not path.exists():
    path.mkdir(parents=True)

if scene == 'all':
    start_index = 0 if args.start_index is None else args.start_index
    end_index = len(scenes) if args.end_index is None else args.end_index
    for s in scenes[start_index:end_index]:
        #print('doing scene', s)
        do_one_scene_all(s, path)

else:
    if scene.isdigit():
        scene = scenes[int(scene)]
    if scene is None:
        scene = scenes[0]
    assert isinstance(scene, str), f'got bad scene ({scene}) of type {type(scene)}'
    do_one_scene_all(scene, path)