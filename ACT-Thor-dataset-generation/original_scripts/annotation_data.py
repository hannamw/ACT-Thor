import random
import math
import shutil
from pathlib import Path, PureWindowsPath
from collections import defaultdict

import pandas as pd
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from tqdm import tqdm

from text_descriptions import regenerate_sentences

random.seed(42)
np.random.seed(42)


def generate_contrast_annotation_dataset(n_afters):
    path = Path('data')
    contrast_path = path / f'contrast-{n_afters}'
    contrast_df = pd.read_csv(contrast_path / 'metadata.csv')

    after_list = [list(contrast_df[f'image_name{i}']) for i in range(n_afters)]
    after_list = zip(*after_list)

    for i, (before_image_name, after_image_names, sentence) in enumerate(zip(contrast_df['before_image_name'], after_list, contrast_df['correct_sentence'])):
        before = Image.open(before_image_name)
        afters = [Image.open(after_image_name) for after_image_name in after_image_names]
        canvas = generate_example_contrast(before, afters, sentence, orientation='horizontal')
        canvas.save(contrast_path / f'{i}.png')


def contrast_annotation_dataset_from_csv(csv, n, output_path, makepath=True, orientation='horizontal', text='sentence',
                                         cs_size=4):
    df = pd.read_csv(csv)
    if 0 < n < len(df):
        df = df.sample(n)
    elif n > len(df):
        print(f'asked for {n} samples but only {len(df)} possible; giving {len(df)}')

    output_path = Path(output_path)
    if makepath:
        output_path.mkdir(exist_ok=True)

    if cs_size == 4:
        assert 'distractor2_path' in df.columns
        print('doing 4-way')
        after_list = [list(df[name]) for name in ['after_image_path', 'distractor0_path', 'distractor1_path', 'distractor2_path']]
    else:
        after_list = [list(df[name]) for name in ['after_image_path', 'distractor0_path', 'distractor1_path']]
    after_list = zip(*after_list)

    assert text == 'sentence' or text == 'action_name'
    inputs = zip(df['before_image_path'], after_list, df[f'order_{cs_size}'], df[text])

    image_paths = []
    for i, (before_image_name, after_image_names, order, sentence) in tqdm(list(enumerate(inputs))):
        order = eval(order)
        before = Image.open(before_image_name)
        afters = [Image.open(after_image_names[int(j)]) for j in order]
        canvas = generate_example_contrast(before, afters, sentence, orientation=orientation, answer=order.index(0))
        image_path = output_path / f'{i}.png'
        image_paths.append(image_path)
        canvas.save(image_path)

    df['annotation_image_path'] = image_paths
    df.to_csv(output_path / 'metadata.csv')


def generate_example_contrast(before, afters, sentence, orientation='horizontal', answer=0):
    font = "/usr/share/fonts/dejavu/DejaVuSans.ttf" #"/usr/share/fonts/truetype/freefont/FreeMono.ttf"
    bw, bh = before.size
    aws, ahs = zip(*[after.size for after in afters])

    if orientation == 'horizontal':
        max_height = max(bh, *ahs)
        canvas = Image.new('RGB', (20 + bw + sum(aws) + 10 * len(afters), 10 + max_height + 50), (255, 255, 255))

        font = ImageFont.truetype(font, 20)
        d = ImageDraw.Draw(canvas)

        curr_width = 10
        canvas.paste(before, (10, 10))
        d.text((15, 15), 'before', fill=(255, 255, 255), font=font, stroke_width=3, stroke_fill=(0, 0, 0))
        d.line([(6, 6), (6 + bw, 6), (6 + bw, bh + 12), (6, bh + 12), (6, 6)], (0, 0, 0), width=6)
        curr_width += bw + 10

        for i, (after, aw) in enumerate(zip(afters, aws)):
            canvas.paste(after, (curr_width, 10))
            d.text((curr_width + 5, 15), str(i), fill=(255, 255, 255), font=font, stroke_width=3, stroke_fill=(0, 0, 0))
            curr_width += 10 + aw

        d.text((10, max_height + 20), sentence, fill=(0, 0, 0), font=font)

    elif orientation == 'vertical':
        max_width = max(bw, *aws)
        canvas = Image.new('RGB', (20 + max_width, 10 + bh + sum(ahs) + 10 * len(afters) + 50), (255, 255, 255))

        font = ImageFont.truetype(font, 20)
        d = ImageDraw.Draw(canvas)

        curr_height = 10
        canvas.paste(before, (10, curr_height))
        d.text((15, 15), 'before', fill=(255, 255, 255), font=font, stroke_width=3, stroke_fill=(0, 0, 0))
        d.line([(6, 6), (6 + bw, 6), (6 + bw, bh + 12), (6, bh + 12), (6, 6)], (0, 0, 0), width=6)

        curr_height += bh + 10
        for i, (after, ah) in enumerate(zip(afters, ahs)):
            canvas.paste(after, (10, curr_height))
            d.text((15, curr_height + 5), str(i), fill=(255, 255, 255), font=font, stroke_width=3,
                   stroke_fill=(0, 0, 0))
            curr_height += 10 + ah

        d.text((10, 10 + bh + sum(ahs) + 10 * len(afters)), sentence, fill=(0, 0, 0), font=font)

    elif orientation == 'square':
        canvas = Image.new('RGB', (30 + bw * 2, 30 + bh * 2 + 40), (255, 255, 255))

        font = ImageFont.truetype(font, 20)
        d = ImageDraw.Draw(canvas)
        for (x, y), label, image in zip([(10,10), (20 + bw, 10), (10, 20 + bh), (20 + bw, 20 + bh)],
                                       ['before', '0', '1', '2'], [before, *afters]):
            canvas.paste(image, (x, y))
            d.text((x + 5, y + 5), label, fill=(255, 255, 255), font=font, stroke_width=3,
                   stroke_fill=(0, 0, 0))

        d.line([(6, 6), (6 + bw, 6), (6 + bw, bh + 12), (6, bh + 12), (6, 6)], (0, 0, 0), width=6)
        d.text((10, 30 + bh * 2), sentence, fill=(0, 0, 0), font=font)

    elif orientation == 'square-top':
        canvas = Image.new('RGB', (30 + bw * 2, 30 + bh * 2 + 40), (255, 255, 255))

        font = ImageFont.truetype(font, 20)
        d = ImageDraw.Draw(canvas)
        d.text((10, 10), sentence, fill=(0, 0, 0), font=font)
        for (x, y), label, image in zip([(10, 50), (20 + bw, 50), (10, 60 + bh), (20 + bw, 60 + bh)],
                                        ['before', '0', '1', '2'], [before, *afters]):
            canvas.paste(image, (x, y))
            d.text((x + 5, y + 5), label, fill=(255, 255, 255), font=font, stroke_width=3,
                   stroke_fill=(0, 0, 0))

        d.line([(6, 46), (6 + bw, 46), (6 + bw, bh + 52), (6, bh + 52), (6, 46)], (0, 0, 0), width=6)

    elif orientation == 'rect-four':
        canvas = Image.new('RGB', (40 + bw * 3, 30 + bh * 2 + 40), (255, 255, 255))

        font = ImageFont.truetype(font, 20)
        d = ImageDraw.Draw(canvas)

        begin_image_coords = (10, round(60 + bh / 2) - 80)
        for (x, y), label, image in zip([begin_image_coords, (20 + bw, 50), (30 + 2 * bw, 50), (20+ bw, 60 + bh), (30 + 2 * bw, 60 + bh)],
                                        ['before', '0', '1', '2', '3'], [before, *afters]):
            canvas.paste(image, (x, y))
            d.text((x + 5, y + 5), label, fill=(255, 255, 255), font=font, stroke_width=3,
                   stroke_fill=(0, 0, 0))

        bx, by = begin_image_coords
        lx, ly = bx - 4, by - 4
        d.line([(lx, ly), (lx + bw, ly), (lx + bw, bh + ly + 6), (lx, bh + ly + 6), (lx, ly)], (0, 0, 0), width=6)

        new_sentence = ''
        line = ''
        for word in sentence.split():
            line = line + word + ' '
            if len(line) > 19:
                new_sentence = new_sentence + line + '\n'
                line = ''
        new_sentence = new_sentence + line + '\n'
        d.text((lx+4, bh + ly + 16), new_sentence, fill=(0, 0, 0), font=font)

    elif orientation == 'rect-four-solved':
        canvas = Image.new('RGB', (40 + bw * 3, 30 + bh * 2 + 40), (255, 255, 255))

        font = ImageFont.truetype(font, 20)
        d = ImageDraw.Draw(canvas)

        begin_image_coords = (10, round(60 + bh / 2) - 80)
        for (x, y), label, image in zip([begin_image_coords, (20 + bw, 50), (30 + 2 * bw, 50), (20+ bw, 60 + bh), (30 + 2 * bw, 60 + bh)],
                                        ['before', '0', '1', '2', '3'], [before, *afters]):
            canvas.paste(image, (x, y))
            d.text((x + 5, y + 5), label, fill=(255, 255, 255), font=font, stroke_width=3,
                   stroke_fill=(0, 0, 0))

        bx, by = begin_image_coords
        lx, ly = bx - 4, by - 4
        d.line([(lx, ly), (lx + bw, ly), (lx + bw, bh + ly + 6), (lx, bh + ly + 6), (lx, ly)], (0, 0, 0), width=6)

        new_sentence = ''
        line = ''
        for word in sentence.split():
            line = line + word + ' '
            if len(line) > 19:
                new_sentence = new_sentence + line + '\n'
                line = ''
        new_sentence = new_sentence + line + '\n'
        d.text((lx+4, bh + ly + 16), new_sentence, fill=(0, 0, 0), font=font)
        d.text((lx + 54, bh + ly + 16), f'Ground truth: {answer}', fill=(0, 0, 0), font=font)

    else:
        raise ValueError('orientation must be horizontal, vertical, or square')
    return canvas


action_names = ['drop','throw','put','push','pull','open','close', 'slice',
                'dirty','fill','toggle','useUp','break','cook','pickUp']


def action_annotation_dataset_from_csv(csv, n, output_path, nlabels=5, makepath=True):
    df = pd.read_csv(csv)
    output_path = Path(output_path)

    if makepath:
        output_path.mkdir(exist_ok=True)

    df = df.sample(n)

    excl_dict = {action: [act for act in action_names if act != action] for action in action_names}

    action_sets, correct_indices = [], []
    for action in df['action_name']:
        action_set = random.sample(excl_dict[action], nlabels-1)
        action_set.insert(random.randrange(nlabels), action)

        correct_index = action_set.index(action)
        action_sets.append(action_set)
        correct_indices.append(correct_index)

    actions_by_index = list(zip(*action_sets))
    for i in range(nlabels):
        df[f'action_{i}'] = actions_by_index[i]
    df['correct_index'] = correct_indices


    inputs = zip(df['before_image_path'], df['after_image_path'], action_sets)

    image_paths = []
    for i, (before_image_name, after_image_name, action_set) in tqdm(list(enumerate(inputs))):
        before = Image.open(before_image_name)
        after = Image.open(after_image_name)
        canvas = generate_example_action(before, after, action_set, orientation='horizontal')
        image_path = output_path / f'{i}.png'
        image_paths.append(image_path)
        canvas.save(image_path)

    df['annotation_image_path'] = image_paths
    df.to_csv(output_path / 'metadata.csv')


def generate_example_action(before, after, labels, orientation='horizontal'):
    font = "/usr/share/fonts/dejavu/DejaVuSans.ttf" # "/usr/share/fonts/truetype/freefont/FreeMono.ttf"
    bw, bh = before.size
    aw, ah = after.size

    label_string = ''
    for i, label in enumerate(labels):
        label_string += f'|{i}: {label} '
    if orientation == 'horizontal':
        max_height = max(bh, ah)

        canvas = Image.new('RGB', (20 + bw + aw + 10, 10 + max_height + 50), (255, 255, 255))
        font = ImageFont.truetype(font, 20)
        d = ImageDraw.Draw(canvas)

        canvas.paste(before, (10, 10))
        canvas.paste(after, (10 + bw + 10, 10))
        d.text((15, 15), 'before', fill=(255, 255, 255), font=font, stroke_width=3, stroke_fill=(0, 0, 0))
        d.text((25 + bw, 15), 'after', fill=(255, 255, 255), font=font, stroke_width=3, stroke_fill=(0, 0, 0))

        d.text((10, 10 + max_height + 10), label_string, fill=(0, 0, 0), font=font)

    elif orientation == 'vertical':
        max_width = max(bw, aw)

        canvas = Image.new('RGB', (20 + max_width, 10 + bh + ah + 10 + 50), (255, 255, 255))
        canvas.paste(before, (10, 10))
        canvas.paste(after, (10, 10 + bh + 10))

        font = ImageFont.truetype(font, 20)
        d = ImageDraw.Draw(canvas)
        d.text((10, 10 + bh + ah + 10), label_string, fill=(0, 0, 0), font=font)
    else:
        raise ValueError('orientation must be horizontal or vertical')
    return canvas


def generate_splits(path, output_path, n_samples, names, labels, flip=False):
    path = Path(path)
    df = pd.read_csv(path / 'metadata.csv')
    output_path = Path(output_path)
    chunk_size = (n_samples // 2)
    total_examples = chunk_size * len(names)

    assert len(df) >= total_examples

    if flip:
        df = df.tail(total_examples)
    else:
        df = df.head(total_examples)
    index = list(df.index)
    rot_index = index[-(chunk_size*3):] + index[:-(chunk_size*3)]
    df['annotator1'] = ['' for _ in range(len(index))]
    df['annotator2'] = ['' for _ in range(len(index))]
    output_path.mkdir(exist_ok=True)
    for i, name in enumerate(names):
        name_index1 = index[i * chunk_size: (i+1) * chunk_size]
        name_index2 = rot_index[i * chunk_size: (i+1) * chunk_size]

        df.loc[name_index1, 'annotator1'] = name
        df.loc[name_index2, 'annotator2'] = name
        name_index = name_index1 + name_index2
        name_df = df.loc[name_index]
        files = [Path(f) for f in name_df['annotation_image_path']]
        name_path = output_path / name
        name_path.mkdir(exist_ok=True)
        for file in files:
            shutil.copy(file, name_path / file.name)
        name_df.to_csv(name_path / 'metadata.csv')
        with open(name_path / 'label_list.txt', 'w') as f:
            f.writelines('\n'.join(labels))

    df.to_csv(output_path / 'metadata.csv')


if __name__ == '__main__':
    # collate_datasets('data-improved-descriptions')
    # build_contrast_df(3, 1)
    # generate_contrast_annotation_dataset(3)
    contrast_annotation_dataset_from_csv('data-improved-descriptions/dataset_cs_scene_object_nopt_augmented_recep.csv',
                                         500, 'cs-supp', orientation='rect-four-solved',
                                         text='action_name', cs_size=4)
    # action_annotation_dataset_from_csv('data-improved-descriptions/dataset_cs_scene_object_nopt.csv', 500, 'action-inference-improved-descriptions-clear-text')

    """
    df = pd.read_csv('data-improved-descriptions/metadata_visible.csv')
    df = df[df['action_name'] == 'pickUp']
    df = df[df['object_name'] == 'Cup']
    for i, (before_path, after_path) in enumerate(zip(df['before_image_path'], df['after_image_path'])):
        before = Image.open(before_path) # '/mnt/povobackup/clic/michaelwesley.hanna/action-effect/data-improved-descriptions/pickupable-not-held/FloorPlan6_physics/12_0.png')
        after = Image.open(after_path) # '/mnt/povobackup/clic/michaelwesley.hanna/action-effect/data-improved-descriptions/pickupable-not-held/FloorPlan6_physics/12_16.png')
        generate_example_action(before, after, []).save(f'cup-examples/{i}.png')
    """

    n_samples = 50
    names = ['alberto', 'alessandro', 'federico', 'michael', 'raffaella']
    #generate_splits('dv-4-way-contrast-square-top', 'contrast-set-annotation', n_samples, names, ['0', '1', '2'])
    #generate_splits('action_annotation', 'action-inference-annotation', n_samples, names, [str(i) for i in range(5)], flip=True)