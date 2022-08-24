

def make_name_to_action_fn(controller):
    return {
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