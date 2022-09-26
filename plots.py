import os
from collections import defaultdict
from pprint import pprint

import numpy
import pandas
import seaborn
import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image
from tqdm import tqdm
from matplotlib.colors import ListedColormap

from visual_features.data import __default_dataset_fname__
from visual_features.data import __default_dataset_path__

name_map = {
        'linear': 'Action-Matrix',
        'linear-concat': 'Concat-linear',
        'fcn': 'Concat-multi-layer',
        'linear-infonce': 'Action-Matrix',
        'linear-concat-infonce': 'Concat-linear',
        'fcn-infonce': 'Concat-multi-layer',
        'baseline-random': 'baseline-random',
        'baseline-similarity': 'baseline-similarity'
    }


# Code for boundning box on subplot
def draw_bbox_on_supblot(sb, bratio=0.1):
    ax = sb.axis('off')
    rec = plt.Rectangle((ax[0] - (0.5*bratio) * (ax[1] - ax[0]), ax[-1] - (0.5*bratio) * (ax[2] - ax[3])),
                        (1 + bratio) * (ax[1] - ax[0]),
                        (1 + bratio) * (ax[2] - ax[3]),
                        fill=False, color="red", lw=2)
    rec = sb.add_patch(rec)
    rec.set_clip_on(False)
    return sb


def load_outs_dataframe(pth):
    """
    Loads one DataFrame containing outputs from an experiment.
    """
    full_df = pandas.read_csv(pth)

    try:
        # Eliminates column for the contrast 3 if present
        full_df = full_df[lambda df: df['contrast_type'] == 'CS4']
        full_df.drop(columns=['contrast_type'], inplace=True)
    except KeyError:
        pass

    return full_df


def compute_accuracy(full_df,
                     use_short_names=False,
                     add_visual_baseline=True, add_human_baseline=True,
                     drop_sample_split=True,
                     override_indexer=None):
    """Calculates an accuracy dataframe from the path of the outputs' dataframe. Eventually add baselines for
    visual-only and random models.

    The visual baseline is built with the pre-determined most visually similar vector in the column
    """

    # if use_short_names:
    #     full_df['vect_model'] = full_df['vect_model'].map(lambda name: shorten_model_name(name))
    # else:
    #     full_df['vect_model'] = full_df['vect_model'].map(lambda name: name.replace('-infonce', ''))

    full_df.loc[:, 'vect_model'] = full_df['vect_model'].map(lambda name: name.replace('-infonce', ''))

    indexer = ['extractor_model', 'vect_model', 'hold_out_procedure']
    if override_indexer is not None:
        assert isinstance(override_indexer, list)
        indexer = override_indexer

    full_df['accuracy'] = (full_df['gt_action'] == full_df['pred_action']).astype(int)
    acc = full_df.groupby(indexer).sum()['accuracy']

    acc = acc / full_df.groupby(indexer).count()['accuracy']

    acc_df = full_df.groupby(indexer).count()  # needed to simplify the dataframe (have unique rows)
    acc_df['accuracy'] = acc

    acc_df = acc_df.reset_index()[indexer + ['accuracy']]

    similarity_df = None
    after_only_baseline = None
    if add_visual_baseline:
        visual_indexer = [item for item in indexer if item != 'vect_model']

        similarity_df = full_df[list(set(visual_indexer + ['gt_action', 'visual_distractor_action']))]
        similarity_df['accuracy'] = (similarity_df['gt_action'] == similarity_df['visual_distractor_action']).astype(int)
        similarity_df = similarity_df.groupby(visual_indexer).mean().reset_index()[visual_indexer + ['accuracy']]

        similarity_df['vect_model'] = 'baseline-similarity'

    if add_human_baseline:
        # after-image only baseline (HARD CODED)
        after_only_baseline = pandas.DataFrame({
            'extractor_model': ['moca-rn', 'clip-rn', 'moca-rn', 'clip-rn', 'moca-rn', 'clip-rn'],
            'hold_out_procedure': ['object_name', 'object_name', 'samples', 'samples', 'scene', 'scene'],
            'accuracy': [0.2676, 0.2841, 0.2556, 0.2624, 0.1981, 0.1348]
        })

    if drop_sample_split:
        acc_df.drop(index=acc_df[lambda d: d['hold_out_procedure'] == 'samples'].index, inplace=True)

        if after_only_baseline is not None:
            after_only_baseline.drop(index=after_only_baseline[lambda d: d['hold_out_procedure'] == 'samples'].index,
                                     inplace=True)
        if similarity_df is not None:
            similarity_df.drop(index=similarity_df[lambda d: d['hold_out_procedure'] == 'samples'].index, inplace=True)

    return acc_df, similarity_df, after_only_baseline


def training_appendix():
    df = pandas.read_csv('new-vect-results/statistics/hold-out/infonce/results.csv')
    df['training'] = 'contrastive'
    df.loc[df['vect_model'] == 'baseline-random', 'accuracy'] = 0.25  # TODO remove correction

    df2 = pandas.read_csv('new-vect-results/statistics/hold-out/l2/results.csv')
    df2['training'] = 'L2'
    df2.loc[df2['vect_model'] == 'baseline-random', 'accuracy'] = 0.25  # TODO remove correction

    baseline_df = [df2, df]

    full_df = pandas.concat(baseline_df)

    full_df['vect_model'] = full_df['vect_model'].map(lambda name: name_map[name])

    seaborn.set(font_scale=1.5)

    g = seaborn.catplot(x='extractor_model', y='accuracy', hue='vect_model', data=full_df[~full_df['vect_model'].isin({'baseline-random', 'baseline-similarity'})],
                        col="hold_out_procedure", row='training', kind="bar", height=10, aspect=.75)

    # random baseline
    [ax.axhline(
        baseline_df[i // 3][(baseline_df[i // 3]['vect_model'] == 'baseline-random') &
                            (baseline_df[i // 3]['hold_out_procedure'] == ax.get_title().split()[-1])
                            ]['accuracy'].mean(),
        linestyle='--', color='lightgray', alpha=0.7, linewidth=2)
        for i, ax in enumerate(g.axes.flatten())]

    # similarity baseline
    [ax.axhline(
        baseline_df[i // 3][(baseline_df[i // 3]['vect_model'] == 'baseline-similarity') &
                            (baseline_df[i // 3]['hold_out_procedure'] == ax.get_title().split()[-1])
                            ]['accuracy'].mean(),
        linestyle='--', color='lightcoral', alpha=0.7, linewidth=2)
        for i, ax in enumerate(g.axes.flatten())]

    for i, ax, title in zip(range(len(g.axes.flatten())), g.axes.flatten(), ['Samples', 'Objects', 'Scenes'] * 2):
        ax.set_title("Hold-out " + title + f" ({str('L2') if i // 3 == 0 else str('contrastive')})")

    g.legend.remove()
    plt.legend(loc='upper right', facecolor='white')
    plt.show()


def training(pth='cs-size-results/infonce-embedding/outputs.csv', show=True):
    full_df = load_outs_dataframe(pth)
    acc_df, similarity_df, after_only_baseline = compute_accuracy(full_df)

    seaborn.set(font_scale=1.5)
    g = seaborn.catplot(x='extractor_model', y='accuracy', hue='vect_model',
                        data=acc_df, col="hold_out_procedure", ci=95,
                        kind="bar",
                        height=10, aspect=.75, sharex=False)

    g.set_titles("Holding out {col_name}")

    added_to_legend = False

    # GRAPHING PARAMS
    hline_thickness = 3
    vonly_color = 'coral'
    af_color = 'purple'
    rand_color = 'lightgray'
    human_color = 'red'
    alpha = .9

    plt.ylim((0,.9))

    # # error bars
    # sdf = full_df[['extractor_model', 'vect_model', 'hold_out_procedure', 'iteration', 'accuracy']]
    # err_df = sdf.groupby(['extractor_model', 'vect_model', 'hold_out_procedure', 'iteration']).mean().reset_index().groupby(['extractor_model', 'vect_model', 'hold_out_procedure']).std()['accuracy']

    # hold out procedures where baselines should not be drawn (because they are not computed)
    not_draw_baselines = ['action', 'structural', 'reversible', 'receptacle', 'surface']

    def check_baselines(h):
        return h not in not_draw_baselines

    for i, ax in enumerate(g.axes.flatten()):
        title = ax.get_title()
        if 'object_name' in title:
            ax.set_title('Holding out objects')
        h_o_proc = title.split()[-1]
        print(h_o_proc)

        labels = [tx.get_text() for tx in ax.get_xticklabels()]
        pos = {name: i for i, name in enumerate(labels)}

        # split [0,1] to have correct positions for hlines
        k = 1 / len(labels)
        eps = 1e-3
        rngs = [(k * i + eps, k * (i + 1) - eps) for i in range(len(labels))]

        for name in labels:
            sim_val = similarity_df[lambda df:
                (df['extractor_model'] == name) &
                (df['hold_out_procedure'] == h_o_proc)
            ]['accuracy'].item()

            if check_baselines(h_o_proc):
                af_val = after_only_baseline[lambda df:
                    (df['extractor_model'] == name) &
                    (df['hold_out_procedure'] == h_o_proc)
                ]['accuracy'].item()

            if added_to_legend or i < (len(g.axes.flatten()) - 1):
                # similarity
                ax.axhline(sim_val, xmin=rngs[pos[name]][0], xmax=rngs[pos[name]][1],
                           linestyle='--', color=vonly_color, alpha=alpha, linewidth=hline_thickness)

                if check_baselines(h_o_proc):
                    # after-image-only
                    ax.axhline(af_val, xmin=rngs[pos[name]][0], xmax=rngs[pos[name]][1],
                               linestyle='--', color=af_color, alpha=alpha, linewidth=hline_thickness)

            else:
                # similarity
                ax.axhline(sim_val, xmin=rngs[pos[name]][0], xmax=rngs[pos[name]][1],
                           linestyle='--', color=vonly_color, alpha=alpha, linewidth=hline_thickness, label='visual-only')
                # after-image-only
                if check_baselines(h_o_proc):
                    ax.axhline(af_val, xmin=rngs[pos[name]][0], xmax=rngs[pos[name]][1],
                               linestyle='--', color=af_color, alpha=alpha, linewidth=hline_thickness, label='action-name')
                added_to_legend = True

        ax.set_xlabel("")
        ax.set_xticklabels([el.get_text().replace('-rn', '').upper() for el in ax.get_xticklabels()])

        # random
        ax.axhline(0.25, linestyle='--', color=rand_color, alpha=alpha, linewidth=hline_thickness, label='random')

        # human
        ax.axhline(0.831, linestyle='-', color=human_color, alpha=0.5, linewidth=hline_thickness - 1, label='human')


    g.legend.remove()
    lg = plt.legend(loc='lower right', facecolor='white')
    if show:
        lg.set_draggable(True)
        plt.show()

    return g


def error_analysis(pth='cs-size-results/infonce_3/outputs.csv', show=True):
    seaborn.set(font_scale=1.5)

    full_df = pandas.read_csv(pth)
    full_df = full_df[lambda df: df['contrast_type'] == 'CS4'].drop(columns=['contrast_type'])
    full_df['vect_model'] = full_df['vect_model'].map(lambda el: el.replace('-infonce', ''))

    ### DROPPING SAMPLES SPLIT ###
    full_df.drop(index=full_df[lambda d: d['hold_out_procedure'] == 'samples'].index, inplace=True)
    ##############################

    indexer = ['extractor_model', 'vect_model', 'hold_out_procedure']

    ac_map = {name: i for i, name in enumerate(sorted(list(set(full_df['gt_action']))))}
    n = len(ac_map)

    def rework_conf_mat(cm):
        cm = cm / cm.sum(-1).reshape(n, 1)
        cm[numpy.isnan(cm)] = 0
        return cm

    def get_conf_mat(row):
        total_conf_mat = numpy.zeros((n, n))
        obj_conf_mat = {
            obj: numpy.zeros((n, n)) for obj in set(full_df['object_name'])
        }

        for row in row.to_dict('records'):
            total_conf_mat[ac_map[row['gt_action']], ac_map[row['pred_action']]] += 1
            obj_conf_mat[row['object_name']][ac_map[row['gt_action']], ac_map[row['pred_action']]] += 1

        total_conf_mat = rework_conf_mat(total_conf_mat)
        obj_conf_mat = {k: rework_conf_mat(obj_conf_mat[k]) for k in obj_conf_mat}
        return total_conf_mat, obj_conf_mat

    tb = full_df.groupby(indexer).apply(lambda r: get_conf_mat(r))

    # Computes action accuracies
    action_accuracies = tb.reset_index()
    action_accuracies[0] = action_accuracies[0].map(lambda el: {k: el[0][ac_map[k], ac_map[k]] for k in ac_map})
    for k in ac_map:
        action_accuracies[k] = action_accuracies[0].map(lambda d: d[k])
    action_accuracies.drop(columns=[0], inplace=True)
    action_accuracies.to_csv(Path(*(Path(pth).parts[:-1])) / 'action_accuracies.csv', index=False)
    action_accuracies.to_excel(Path(*(Path(pth).parts[:-1])) / 'action_accuracies.xlsx', index=False)

    # Plots
    cmap = {
        'Action-Matrix': 'Blues',
        'Concat-Multi': 'Greens',
        'Concat-Linear': 'Oranges'
    }

    rows = sorted(set(full_df['hold_out_procedure']))
    cols = sorted(set(full_df['vect_model']))

    col_idx = {k: i+1 for i, k in enumerate(cols)}
    row_idx = {k: i+1 for i, k in enumerate(rows)}

    fig = {
        k: plt.figure(figsize=(18, 12)) for k in set(full_df['extractor_model'])
    }

    for idx, el in zip(tb.index, tb):
        curr_r = row_idx[idx[-1]]
        curr_c = col_idx[idx[1]]

        print(idx, (curr_r, curr_c))

        ax = fig[idx[0]].add_subplot(len(rows), len(cols), (curr_r - 1) * len(cols) + curr_c)

        cm, obj_cm = el
        seaborn.heatmap(ax=ax, data=cm, cbar=True,
                        vmin=0.0, vmax=1.0,
                        cmap=cmap[idx[1]], square=True,
                        xticklabels=list(ac_map.keys()), yticklabels=list(ac_map.keys()))

        ax.set_title(f"{idx[1]}")

        if curr_c == 1:
            ax.set_ylabel(f"Holding out {idx[-1].split('_')[0]}")

    for k in fig:
        fig[k].suptitle("Confusion Matrices for " + k.split("-")[0].upper())
        fig[k].tight_layout()

    if show:
        plt.show()

    if not show:
        for k in fig:
            fig[k].savefig(Path(*(Path(pth).parts[:-1])) / (k + '-conf.png'))


    return fig


def plot_dataset(pth=str(Path(__default_dataset_path__, __default_dataset_fname__)),
                 show=True):
    df = pandas.read_csv(pth, index_col=0)
    df.drop(index=df[(df['action_name'] == 'cook') | (df['distractor0_action_name'] == 'cook') | (
                df['distractor1_action_name'] == 'cook') | ((df['distractor2_action_name'] == 'cook'))].index,
            inplace=True)
    obj_indexer = ['object_split', 'object_name', 'action_name', 'after_image_path']
    obj_split_df = df[obj_indexer]
    obj_split_df.loc[lambda d: d['object_split'] != 'test', 'object_split'] = 'Seen'
    obj_split_df.loc[lambda d: d['object_split'] == 'test', 'object_split'] = 'Unseen'
    scene_indexer = ['scene_split', 'scene', 'action_name', 'after_image_path']
    scene_split_df = df[scene_indexer]
    scene_split_df.loc[lambda d: d['scene_split'] != 'test', 'scene_split'] = 'Seen'
    scene_split_df.loc[lambda d: d['scene_split'] == 'test', 'scene_split'] = 'Unseen'
    obj_split_df.rename(columns={'object_split': 'split'}, inplace=True)
    scene_split_df.rename(columns={'scene_split': 'split'}, inplace=True)
    obj_split_df['split_name'] = 'object'
    scene_split_df['split_name'] = 'scene'
    co_df = obj_split_df.groupby(['split_name', 'split', 'action_name']).count().reset_index()
    cs_df = scene_split_df.groupby(['split_name', 'split', 'action_name']).count().reset_index()
    split_df = pandas.concat([co_df, cs_df])
    split_df.rename(columns={'after_image_path': 'Samples'}, inplace=True)
    seaborn.set(font_scale=2)
    g = seaborn.catplot(data=split_df, x='split', y='Samples', row='split_name', hue='action_name', kind='bar');
    g.set_titles("{row_name} split");
    g.legend.set_draggable(True);
    g.legend.set_title('Action');

    if show:
        plt.show()


def action_accuracies(pth='cs-size-results/infonce-embedding/outputs.csv'):
    df = pandas.read_csv(pth)
    df['vect_model'] = df['vect_model'].map(lambda el: el.replace('-infonce', ''))
    resdf = df[['hold_out_procedure', 'extractor_model', 'vect_model', 'iteration', 'object_name', 'pred_action', 'gt_action']]
    resdf['accuracy'] = (resdf.pred_action == resdf.gt_action).astype(int)

    # TODO choose best model
    resdf = resdf[(resdf.extractor_model == 'moca-rn') & (resdf.vect_model == 'Concat-Multi')]
    tmp = resdf[lambda d: d['hold_out_procedure'] == 'object_name'].groupby(['gt_action', 'object_name', ]).mean()['accuracy']
    tmp = tmp.to_frame()
    tmp = tmp.reset_index().pivot(index='object_name', columns='gt_action')
    print(tmp)


def nearest_samples(pth, k=4, n=100):
    """Saves images with before, after and k nearest images for all models within the specified directory."""
    dataset = pandas.read_csv(Path(__default_dataset_path__, __default_dataset_fname__))
    super_pth = Path(pth).parent / 'images-with-before'

    def wrap_nearest_samples(pth, k=4, n=10):
        df = pandas.read_csv(pth)

        for gt, preds in tqdm(zip(df['pth'][:n], df['neighbors'][:n]), desc='Preparing neighbor images...', total=n):

            preds = [eval(el) for el in preds[1:-1].split(',')]

            fig, axes = plt.subplots(1, k + 2)

            # print(gt.replace('new-dataset/', ''))
            # print(dataset[lambda d: d.after_image_path == gt.replace('new-dataset/', '')]['before_image_path'])
            axes[0].imshow(Image.open("new-dataset/" + dataset[lambda d: d.after_image_path == gt.replace('new-dataset/', '')]['before_image_path'].item()))
            axes[0].set_axis_off()

            axes[1].imshow(Image.open(gt))
            axes[1].set_axis_off()

            for prd, ax in zip(preds[:k], axes[2:]):
                ax.imshow(Image.open(prd))
                ax.set_axis_off()
                if prd == gt:
                    draw_bbox_on_supblot(ax)

            plt.tight_layout()

            model_name = Path(pth).parts[-3].split("+")[0].replace("-infonce", '')
            curr_split = Path(pth).parts[-2].split("_")[0]
            location = Path(gt).parts[-3:-1]
            imname = Path(gt).parts[-1]
            extr_model = 'moca' if 'moca-rn' in pth else 'clip'
            imsave_pth = Path(super_pth, extr_model, curr_split, *location, model_name, imname)

            os.makedirs(imsave_pth.parent, exist_ok=True)
            plt.savefig(imsave_pth, bbox_inches='tight')
            plt.close()

    for r, d, fnames in os.walk(pth):
        for fname in fnames:
            if fname == 'ranking.csv':
                curr_model_name = Path(*[el.replace('_neighbors', '') for el in Path(r).parts[-2:]])
                print(f"----- Neighbors for {shorten_model_name(curr_model_name)} -----")
                wrap_nearest_samples(os.path.join(r, fname), k, n)

    # Creating combined images for all 3 models' results in each sample
    for r, d, fnames in tqdm(os.walk(super_pth), desc='Generating full images...'):
        if "FloorPlan" in Path(r).parts[-1]:
            imgs_dict = defaultdict(list)
            for local_r, _, local_fnames in os.walk(r):
                for fname in local_fnames:
                    imgs_dict[fname].append(os.path.join(local_r, fname))

            for k in imgs_dict.keys():
                # sorted by alphabetical order: Action-Matrix, Concat-Linear, Concat-Multi
                imgs_dict[k] = sorted(imgs_dict[k], key=lambda el: Path(el).parts[-2])
                fig, axes = plt.subplots(3, 1)
                for imgpth, ax in zip(imgs_dict[k], axes):
                    ax.imshow(Image.open(imgpth))
                    ax.text(0, 0, Path(imgpth).parts[-2], fontsize='small')
                    ax.set_axis_off()
                # plt.tight_layout()
                plt.subplots_adjust(wspace=0)
                plt.savefig(os.path.join(r, k), bbox_inches='tight')
                plt.close()


def shorten_model_name(mname):
    name = str(mname)
    name = name.replace("_neighbors", '')
    name = name.replace("-infonce", "")
    name = name.replace('Action-Matrix', 'AM')
    name = name.replace('Concat-Linear', 'CL')
    name = name.replace('Concat-Multi', 'CM')
    name = name.replace('moca-rn', 'moca')
    name = name.replace('clip-rn', 'clip')
    name = name.replace('\\', '-')
    name = name.replace('/', '+')
    if len(name) > 2:
        name = name[:-2]  # exclude final "_1" or similar
    return name


def get_object(pth):
    return Path(pth).parts[-1][:-4].split("_")[0]


def get_action(pth):
    return Path(pth).parts[-1][:-4].split("_")[1]


def stats_nearest_samples(pth, savefig=False, print_results=False):
    """Computes statistics for nearest neighbors analysis. Up to now are planned to include first-position analysis and
    AP/MAP analysis. Path specified should be the one pointing to the super-directory where different model
    experiments are contained; neighbors should have been already extracted and ranked within subdirectories inside it.
    """

    def wrap_stats_nearest_samples(pth):
        df = pandas.read_csv(pth)
        df['neighbors'] = df['neighbors'].apply(lambda el: eval(el))

        first_pos_stats = {
            'gt': [],
            'obj-action': [],
            'action': [],
            'object': []
        }
        ap_stats = first_pos_stats.copy()
        ap_stats.pop('gt')

        def compute_ap(l):
            """Computes a modified version of the Average Precision metric."""
            corrects = 0
            rsum = 0
            for i in range(len(l)):
                if l[i] == 1:  # relevant
                    corrects += 1
                    rsum += (corrects / (i + 1))
            ap = rsum / corrects
            return ap

        it = list(zip(df['pth'], df['neighbors']))
        for gt, neighbors in tqdm(it, desc='Analyzing neighbors...', total=len(it)):
            l = len(neighbors)
            obj, action = get_object(gt), get_action(gt)

            # Computes first positions where GT is found
            first_pos_stats['gt'].append(1 - ([i for i, el in enumerate(neighbors) if el == gt][0] / l))
            first_pos_stats['object'].append(1 - ([i for i, el in enumerate(neighbors) if get_object(el) == obj][0] / l))
            first_pos_stats['action'].append(1 - ([i for i, el in enumerate(neighbors) if get_action(el) == action][0] / l))
            first_pos_stats['obj-action'].append(1 - ([i for i, el in enumerate(neighbors) if (get_object(el) == obj) and (get_action(el) == action)][0] / l))

            # Binary lists containing positives for different measures, then computes average precision
            preds_positives = {
                'obj-action': [
                        int((get_action(n) == action) and (get_object(n) == obj))
                        for n in neighbors],
                'action': [int(get_action(n) == action) for n in neighbors],
                'object': [int(get_object(n) == obj) for n in neighbors]
            }

            for k in preds_positives:
                ap_stats[k].append(compute_ap(preds_positives[k]))

        first_pos_stats = {k: (sum(first_pos_stats[k]) / len(first_pos_stats[k])) for k in first_pos_stats}
        ap_stats = {k: (sum(ap_stats[k]) / len(ap_stats[k])) for k in ap_stats}  # computing MAP for each metric

        return {'1st position': first_pos_stats, 'AP': ap_stats}
    # ################################## wrap end ###################################

    if not os.path.exists(Path(Path(pth).parent, 'stats_results.csv')):
        df = []
        for r, d, fnames in os.walk(pth):
            for fname in fnames:
                if fname == 'ranking.csv':
                    curr_model_name = str(Path(*[el.replace('_neighbors', '') for el in Path(r).parts[-2:]]))

                    # gets model specifications
                    model_specs = {
                        'vect_model': curr_model_name.split("-infonce")[0],
                        'hold_out_procedure': curr_model_name.split("-rn")[-1].split("_")[0].replace('/', ''),
                        'extractor_model': curr_model_name.split('-rn')[0].split('+')[-1].replace('-rn', '').upper()
                    }

                    curr_model_name = shorten_model_name(curr_model_name)
                    print(f"----- Neighbor Stats for {curr_model_name} -----")

                    res = wrap_stats_nearest_samples(os.path.join(r, fname))

                    # adds each single metric value as row in the final dataframe
                    for metric in res.keys():  # 1st position or AP
                        df.extend([{
                            **model_specs,
                            'metric': metric,
                            'analyzed': k,
                            'value': res[metric][k]
                        } for k in res[metric].keys()])

        df = pandas.DataFrame(df)
        df.to_csv(str(Path(Path(pth).parent, 'stats_results.csv')), index=False)
    else:
        df = pandas.read_csv(str(Path(Path(pth).parent, 'stats_results.csv')))

    # sort rows for color consistency
    df = df.sort_values(by=['hold_out_procedure'], axis=0, ascending=False)

    gs = []

    for i, h_o_proc in enumerate(set(df.hold_out_procedure)):
        g = seaborn.catplot(
            data=df[lambda d: d.hold_out_procedure == h_o_proc].sort_values(by=['extractor_model'], axis=0, ascending=True),
            x='extractor_model', y='value', hue='vect_model',
            row='metric', col='analyzed',
            kind='bar', ci=None,
            row_order=sorted(set(df.metric)), col_order=sorted(set(df.analyzed))
        )
        g.fig.suptitle(h_o_proc, y=1.1, size=16)
        if savefig:
            g.savefig(Path(Path(pth).parent, f'results_{h_o_proc}.png'))
        gs.append(g)

    if print_results:
        pandas.set_option('display.max_rows', None)
        print(df.groupby(['extractor_model', 'hold_out_procedure', 'metric', 'analyzed', 'vect_model']).mean())

    return gs


def action_crossval(pth, show=False, savefig=True):
    full_df = load_outs_dataframe(pth)
    indexer = ['extractor_model', 'vect_model', 'hold_out_procedure', 'tested_action', 'tested_object', 'gt_action']

    # TODO manipulate dataset to separate accuracy per action (up to now it is averaging over actions
    #  and computing average performance) -> select rows where gt_action and tested_action coincide?
    crossval_df = full_df[lambda df: df.tested_action == df.gt_action]
    acc_df, _, _ = compute_accuracy(crossval_df, use_short_names=True, add_human_baseline=False, override_indexer=indexer)

    col_wrap = 3
    rows = len(set(acc_df.tested_action)) // col_wrap

    cmap = {
        'Action-Matrix': 'blue',
        'Concat-Multi': 'green',
        'Concat-Linear': 'orange'
    }

    for extr in ['moca-rn', 'clip-rn']:
        g, axes = plt.subplots(nrows=rows, ncols=col_wrap, figsize=(10,9))
        for ax, ac in zip(axes.flatten(), set(acc_df.tested_action)):
            for vect_model in cmap:

                vals = acc_df[
                    lambda d: (d.extractor_model == extr) &
                              (d.tested_action == ac) &
                              (d.vect_model == vect_model)
                ]
                objs, vals = vals.tested_object.tolist(), vals.accuracy.tolist()
                ax.plot(objs, vals, color=cmap[vect_model], label=vect_model)
                ax.set_title(ac)
                ax.tick_params(axis='x', labelrotation=60)

            # g = seaborn.relplot(data=acc_df[lambda d: d.extractor_model == extr].drop(columns='extractor_model'),
            #                     x='tested_object', y='accuracy', hue='vect_model',
            #                     row='tested_action',
            #                     kind='line', ci=None)
            # g.legend.set_draggable(True)

        plt.subplots_adjust(
            left=.10, bottom=.10, right=.926, top=.938, wspace=.2, hspace=.56
        )
        plt.suptitle(extr.replace('-rn', '').upper())

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        g.legend(by_label.values(), by_label.keys(), loc='upper right')

        if savefig:
            g.savefig(str(Path(pth).parent / f"{extr}.png"))

    if show:
        plt.show()

    return g


def mds(vecs, preds, model_name, show=False):
    """Generates 2D scatter plot for the MDS reduction. Each input object should be a dictionary containing
     the vector of 2D points ('vec') and the color indices or names ('category')."""

    # Assign colors for actions
    categories = list(set(vecs['category']))

    categories = {el: i for i, el in enumerate(sorted(categories))}
    original_cm = plt.get_cmap('hsv')
    sp = numpy.linspace(0, .95, len(categories))
    colors = original_cm(sp)
    colors[:, :-1] -= 0.1
    colors = colors.clip(0, 1)
    cm = ListedColormap(colors)  # colormap for vectors

    # Make colors of predictions of a lighter shade
    light_colors = (colors + 0.3).clip(0, 1)
    light_colors[:, :-1] = light_colors[:, :-1].clip(0, .93)
    light_cm = ListedColormap(light_colors)  # colormap for predictions

    # if 2 figures no need for different colors
    light_cm = cm

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    standard_size = 50
    standard_alpha = 0.8
    # plots real vectors
    axes[0].scatter(vecs['vec'][:, 0], vecs['vec'][:, 1],
                    color=[cm(categories[name] / len(categories)) for name in vecs['category']],
                    s=[standard_size for i in range(len(vecs['vec']))],
                    alpha=standard_alpha)
    axes[0].set_title('Reals')

    # plots predictions
    axes[1].scatter(preds['vec'][:, 0], preds['vec'][:, 1],
                    color=[light_cm(categories[name] / len(categories)) for name in preds['category']],
                    s=[standard_size for i in range(len(preds['vec']))],
                    alpha=standard_alpha)
    axes[1].set_title('Predicted')

    # creates handles for the legend
    lp = lambda name: plt.plot([], color=cm(categories[name] / len(categories)), ms=standard_size ** .5, mec="none",
                               label=f"{name}", ls="", marker="o")[0]
    handles = [lp(i) for i in categories]
    fig.legend(handles=handles)
    fig.suptitle(model_name)

    if show:
        plt.show()

    return fig


if __name__ == '__main__':

    # training('exp-action-split/0_@5/outputs.csv')

    # error_analysis('cs-size-results/alternative-obj-split/outputs.csv')

    # plot_dataset('new-dataset/data-improved-descriptions/alternative_obj_split_dataset.csv')

    # nearest_samples("experiment-nearest-neighbors/models")

    stats_nearest_samples("experiment-nearest-neighbors/models", savefig=True)

    pass
