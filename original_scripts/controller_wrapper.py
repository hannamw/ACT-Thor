# Certain helper functions adapted from
# https://github.com/rowanz/piglet/blob/main/sampler/ai2thor_env.py

import math
import numpy as np
from PIL import Image


def valid_receptacle(objid, object_metadata):
    if objid not in object_metadata:
        return False
    met = object_metadata[objid]
    return met['receptacle'] and (not met['openable'] or met['isOpen']) and ('Floor' not in met['objectId']) #met['position']['y'] >= 0.25


def visible_receptacle(objid, object_metadata):
    return valid_receptacle(objid, object_metadata) and object_metadata[objid]['visible']


class ControllerWrapper:
    def __init__(self, controller):
        self.controller = controller

    def reset(self, scene=None):
        self.controller.reset(scene=scene, renderInstanceSegmentation=True)

    def get_frame(self):
        return self.controller.last_event.frame

    def get_image(self):
        return Image.fromarray(self.get_frame())

    def get_metadata(self):
        return self.controller.last_event.metadata

    def last_success(self):
        return self.get_metadata()["lastActionSuccess"]

    def drop(self, objid):
        self.controller.step("LookDown", degrees=60)
        if not self.last_success():
            self.controller.step("LookDown", degrees=30)
        self.controller.step(
            action="DropHandObject",
            forceAction=True
        )
        return self.last_success()

    def all_objects(self):
        """Return all object metadata."""
        return self.controller.last_event.metadata["objects"]

    def all_objects_with_properties(self, properties):
        """Find all objects with the given properties."""
        objects = []
        for o in self.all_objects():
            satisfies_all = True
            for k, v in properties.items():
                if o[k] != v:
                    satisfies_all = False
                    break
            if satisfies_all:
                objects.append(o)
        return objects

    def any_held(self):
        return len(self.all_objects_with_properties({'isPickedUp': True})) > 0

    def get_held(self):
        picked_up = self.all_objects_with_properties({'isPickedUp': True})
        return picked_up[0] if picked_up else None

    def get_pickupable(self):
        return self.all_objects_with_properties({'pickupable': True})

    def get_receptacles(self):
        all_objs = self.all_objects()
        object_metadata = {obj['objectId']: obj for obj in self.all_objects()}
        valid_receptacles = [obj for obj in all_objs if valid_receptacle(obj['objectId'], object_metadata)]
        return valid_receptacles

    def get_largest_receptacles(self):
        valid_receptacles = self.get_receptacles()
        largest_receptacles = sorted(valid_receptacles, key=lambda obj: _size3d(obj), reverse=True)
        return largest_receptacles

    def force_pick_up(self, objid=None):
        objid = self.objid if objid is None else objid
        # pick object up
        self.controller.step(
            action="PickupObject",
            objectId=objid,
            forceAction=True,
            manualInteract=False
        )

        return self.last_success()

    def put(self, objid=None):
        if objid:
            self.controller.step(
                action="PutObject",
                objectId=objid,
                forceAction=True,
            )
            return 0
        objects_in_view = list(k for k in self.controller.last_event.object_id_to_color.keys() if "|" in k)
        object_metadata = {obj['objectId']: obj for obj in self.all_objects()}

        receptacles = [objid for objid in objects_in_view if visible_receptacle(objid, object_metadata)]

        for receptacle in receptacles:
            self.controller.step(
                action="PutObject",
                objectId=receptacle,
                forceAction=True,
            )

            if self.last_success():
                print(receptacle)
                return receptacle
        return -1

    def throw(self, objid):
        heldobj = self.get_held()
        throw_strength = 200 * heldobj['mass']
        self.controller.step(
            action="ThrowObject",
            moveMagnitude=throw_strength,
            forceAction=True
        )

        return self.last_success()

    def toggle(self, objid=None):
        objid = objid if objid is not None else self.objid
        event = self.controller.step(
            action="ToggleObjectOn",
            objectId=objid,
            forceAction=False
        )

        return self.last_success()

    def open(self, objid=None):
        objid = objid if objid is not None else self.objid
        self.controller.step(
            action="OpenObject",
            objectId=objid,
            openness=1,
            forceAction=True
        )

        return self.last_success()

    def close(self, objid=None):
        objid = objid if objid is not None else self.objid
        self.controller.step(
            action="CloseObject",
            objectId=objid,
            forceAction=True
        )
        return self.last_success()

    def slice(self, objid=None):
        objid = objid if objid is not None else self.objid
        self.controller.step(
            action="SliceObject",
            objectId=objid,
            forceAction=True
        )
        return self.last_success()

    def dirty(self, objid=None):
        objid = objid if objid is not None else self.objid
        self.controller.step(
            action="DirtyObject",
            objectId=objid,
            forceAction=True
        )
        return self.last_success()

    def empty(self, objid=None):
        objid = objid if objid is not None else self.objid
        self.controller.step(
            action="EmptyLiquidFromObject",
            objectId=objid,
            forceAction=True
        )
        return self.last_success()

    def fill(self, objid=None, fill_liquid='wine'):
        objid = objid if objid is not None else self.objid
        self.controller.step(
            action="FillObjectWithLiquid",
            objectId=objid,
            fillLiquid=fill_liquid,
            forceAction=True
        )
        filled = self.last_success()

        if self.get_held():
            self.controller.step(
                action="RotateHeldObject",
                pitch=-30,
            )

        return filled and self.last_success()

    def use_up(self, objid=None):
        objid = objid if objid is not None else self.get_held()['objectId']
        self.controller.step(
            action="UseUpObject",
            objectId=objid,
            forceAction=True
        )
        return self.last_success()

    def cook(self, objid=None):
        objid = objid if objid is not None else self.get_held()['objectId']
        self.controller.step(
            action="CookObject",
            objectId=objid,
            forceAction=True
        )
        return self.last_success()

    def destroy(self, objid=None):
        objid = objid if objid is not None else self.get_held()['objectId']
        self.controller.step(
            action="BreakObject",
            objectId=objid,
            forceAction=True
        )
        return self.last_success()

    def push(self, objid):
        obj = self.get_object_by_id(objid)
        push_strength = 150 * obj['mass']
        self.controller.step(
        action = "DirectionalPush",
        objectId = objid,
        moveMagnitude = push_strength,
        pushAngle = "0"
        )
        return self.last_success()

    def pull(self, objid):
        obj = self.get_object_by_id(objid)
        push_strength = 60 * obj['mass']
        self.controller.step(
            action="DirectionalPush",
            objectId=objid,
            moveMagnitude=push_strength,
            pushAngle="180"
        )
        return self.last_success()

    def put_careful(self, objid=None):
        receptacle_id = self.receptacle_id
        heldobj = self.get_held()
        recep = self.put(receptacle_id)
        self.controller.step(
            action="GetSpawnCoordinatesAboveReceptacle",
            objectId=receptacle_id,
            anywhere=False
        )
        spawncoords = self.get_metadata()['actionReturn']
        if spawncoords is None:
            return
        agloc = self.get_agent_location()
        recep_loc = self.get_object_by_id(receptacle_id)['position']

        ax = self.get_axis()
        ax = 'z' if ax == 'x' else 'x'

        sorted_sp = sorted(spawncoords, key=lambda sp: penalized_dist(sp, agloc, ax))
        for sp in sorted_sp:
            if dist(sp, agloc) < 0.85:
                continue
            self.controller.step(
                action="PlaceObjectAtPoint",
                objectId=heldobj['objectId'],
                position=sp
            )
            metadata_dict = {obj['objectId']: obj for obj in self.all_objects()}
            heldobjid = heldobj['objectId']
            if self.last_success() and metadata_dict[heldobjid]['visible']:
                print(dist(sp, agloc))
                return True
        return False

    @property
    def currently_reachable_points(self):
        """List of {"x": x, "y": y, "z": z} locations in the scene that are
        currently reachable."""
        self.controller.step({"action": "GetReachablePositions"})
        return self.get_metadata()["reachablePositions"]

    def get_agent_location(self):
        """Gets agent's location."""
        metadata = self.get_metadata()
        location = {
            "x": metadata["agent"]["position"]["x"],
            "y": metadata["agent"]["position"]["y"],
            "z": metadata["agent"]["position"]["z"],
            "rotation": metadata["agent"]["rotation"]["y"],
            "horizon": metadata["agent"]["cameraHorizon"],
            "standing": metadata['agent']["isStanding"],
        }
        return location

    def get_object_by_id(self, object_id: str):
        for o in self.get_metadata()["objects"]:
            if o["objectId"] == object_id:
                return o
        return None

    @staticmethod
    def position_dist(p0, p1, ignore_y=False):
        """Distance between two points of the form {"x": x, "y":y, "z":z"}."""
        return math.sqrt(
            (p0["x"] - p1["x"]) ** 2
            + (0 if ignore_y else (p0["y"] - p1["y"]) ** 2)
            + (p0["z"] - p1["z"]) ** 2
        )

    def attempt_teleport(self, obj):
        objid = obj['objectId']
        objpos = obj['position']
        agentpos = self.get_agent_location()
        for axis in ['x', 'z']:
            for direction in [-1, 1]:
                newpos = {}
                newpos['y'] = agentpos['y']
                newpos['x'] = round_to_quarter(objpos['x'])
                newpos['z'] = round_to_quarter(objpos['z'])
                newpos[axis] += direction

                # print('try', newpos)
                self.controller.step('Teleport', position=newpos)

                if self.last_success():
                    return newpos
        print('failed to teleport')
        return None

    def teleport(self, newpos):
        if newpos is not None:
            self.controller.step('Teleport', position=newpos)

    def rotate(self):
        self.controller.step("RotateLeft")

    def best_direction(self, obj):
        objid = obj['objectId']
        areas = []
        for i in range(4):
            masks = self.controller.last_event.instance_masks
            area = masks[objid].sum() if objid in masks else 0
            areas.append(area)
            self.controller.step("RotateLeft")
        return max(list(range(4)), key=lambda i: areas[i]), max(areas)

    def remove(self, objid):
        self.controller.step('RemoveFromScene', objectId=objid)

    def clear(self, objid):
        if objid is None:
            return
        receptacle = self.get_object_by_id(objid)
        for objid in receptacle['receptacleObjectIds']:
            self.remove(objid)

    def warp_near_largest_receptacle(self, clear=True, rotate=True):
        teleported = False
        newpos = None
        larec = None
        for larec in self.get_largest_receptacles():
            newpos = self.attempt_teleport(larec)
            if self.last_success():
                teleported = True
                break
        if not teleported:
            print('never teleported near largest receptacle')
            return None, None, None

        if clear and larec is not None:
            for objid in larec['receptacleObjectIds']:
                self.remove(objid)

        rots = 0
        if rotate:
            rots, _ = self.best_direction(larec)
            for _ in range(rots):
                self.controller.step("RotateLeft")

        larec_id = larec['objectId'] if teleported else None
        return newpos, rots, larec_id

    def get_axis(self):
        metadata = self.get_metadata()
        rotation = metadata['agent']['rotation']['y']
        if rotation < 45 or 150 <= rotation < 210 or 315 <= rotation:
            return 'z'
        else:
            return 'x'


def _size3d(o):
    xyz = np.array([o['axisAlignedBoundingBox']['size'][k] for k in 'xyz'])
    return np.sqrt(xyz[0] * xyz[2])

def round_to_quarter(x):
    rem = x - int(x)
    rounded_rem = round(rem * 4) / 4
    return int(x) + rounded_rem

def dist(p1, p2):
    return math.sqrt(sum((p1[c]-p2[c])**2 for c in ['x','z']))


def penalized_dist(p1, p2, penalized_direction):
    xpen = 5 if penalized_direction == 'x' else 1
    zpen = 5 if penalized_direction == 'z' else 1
    return math.sqrt(xpen * (p1['x']-p2['x'])**2 + zpen * (p1['z']-p2['z'])**2)
