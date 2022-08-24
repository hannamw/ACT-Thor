import re

action_to_verb = {
    'pickUp': 'The robot picked the <obj> up',
    'drop': 'The robot dropped the <obj>',
    'throw': 'The robot threw the <obj>',
    'put': 'The robot put the <obj> down',
    'push': 'The robot pushed the <obj>',
    'pull': 'The robot pulled the <obj>',
    'open': 'The robot opened the <obj>',
    'close': 'The robot closed the <obj>',
    'slice': 'The robot sliced the <obj>',
    'dirty': 'The robot dirtied the <obj>',
    'fill': 'The robot filled the <obj>',
    'empty': 'The robot emptied the <obj>',
    'toggle': 'The robot toggled the <obj>',
    'useUp': 'The robot used up the <obj>',
    'break': 'The robot broke the <obj>',
    'cook': 'The robot cooked the <obj>'
}


def un_uppercase(s):
    if s == s.upper():
        return s
    uppers = []
    strs = []
    for i, c in enumerate(s):
        if c.isupper():
            uppers.append(i)
    if not uppers:
        return s.lower()
    last = uppers[0]
    uppers = uppers[1:]
    for u in uppers:
        strs.append(s[last:u].lower())
        last = u
    strs.append(s[last:].lower())
    return ' '.join(strs)


def generate_sentence(obj, action, receptacle='', other=''):
    obj = un_uppercase(obj)
    s = re.sub('<obj>', obj, action_to_verb[action])

    if receptacle:
        receptacle = un_uppercase(receptacle)
        if action == 'pickUp':
            preposition = 'from'
        elif action == 'drop':
            preposition = 'onto'
        elif action == 'throw':
            preposition = 'towards'
        else:
            preposition = 'on'
        s += f' {preposition} the {receptacle}'
    if other:
        other = un_uppercase(other)
        if action == 'fill':
            s += f' with {other}'
        elif action == 'empty':
            s += f' of {other}'
        elif action == 'toggle':
            s += f' {other}'
        elif action == 'push' or action == 'pull':
            if other in {'SinkBasin', 'BathBasin',}:
                prep = 'into'
            else:
                prep = 'onto'
            s += f' {prep} the {other}'

    s += '.'
    return s


def regenerate_sentences(df):
    sentences = df['sentence']
    objs = df['object_name']
    actions = df['action_name']
    receptacles = [rec if action == 'put' else '' for rec, action in zip(df['receptacle_name'], df['action_name'])]
    df['old_sentences'] = sentences
    df['sentence'] = [generate_sentence(obj, action, receptacle) for obj, action, receptacle in
                      zip(objs, actions, receptacles)]


def generate_detailed_sentence(action_name, is_placed, recep_name, end_recep, object_name):
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
    return sentence