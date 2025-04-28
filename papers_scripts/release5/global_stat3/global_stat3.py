"""
Global statistical analysis of SPM maps produced by first-level analyis
of the dataset.
* Tease out effect of subject, task and phase encoding direction
* Study global similarity effects
Authors: Bertrand Thirion, Ana Luisa Pinho 2020
Last update: Fernanda Ponce, June 2023
Compatibility: Python 3.5
"""

""" NOTES
* This script generates the figures presented at OHBM 2023
* Data form 11 subjects but only 14 tasks (out of 18)
* This scrip generates more images that what we actually use.
"""

# %%
import os
import json
import warnings
import numpy as np
import pandas as pd
import nibabel as nib
import scipy.stats as st
import matplotlib.pyplot as plt
from joblib import Memory
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.manifold import TSNE
from nilearn.maskers import NiftiMasker
from nilearn import plotting
from nilearn.image import math_img
from nilearn.glm.second_level import SecondLevelModel
from nilearn.glm import threshold_stats_img
from ibc_public.utils_data import (data_parser, DERIVATIVES,
                                   SMOOTH_DERIVATIVES, ALL_CONTRASTS,
                                   CONDITIONS)

# %%

participants = [4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15]

# Fifth Release 
task_list = ['MathLanguage', 'SpatialNavigation', 'EmoReco', 'EmoMem',
             'StopNogo', 'Catell', 'FingerTapping', 'VSTMC',
             'BiologicalMotion1', 'BiologicalMotion2',
             'Checkerboard','FingerTap','ItemRecognition', 'BreathHolding'
             ]

suffix = '7jun'

cache = '/storage/store3/work/aponcema/IBC_paperFigures/global_stat3/'\
        'cache_global_stat3_7jun'
mem = Memory(location=cache, verbose=0)

# %%
sub_path = [os.path.join(DERIVATIVES, 'sub-%02d' % s) for s in participants]
PTS = [os.path.basename(full_path) for full_path in sub_path]

# BIDS conversion of task names
with open(os.path.join('bids_postprocessed.json'), 'r') as f:
    task_dic = json.load(f)

TASKS = [task_dic[tkey] for tkey in task_list]
TASKS = [item for sublist in TASKS for item in sublist]

# # Some tricks to find the right maps (to be discussed)
df_conds = CONDITIONS

trick_cond = (df_conds['task'] == 'Catell') & (df_conds['contrast'] == 
                                               'easy_oddball')
df_conds.loc[trick_cond, 'contrast'] = 'easy'

trick_cond = (df_conds['task'] == 'Catell') & (df_conds['contrast'] == 
                                               'hard_oddball')
df_conds.loc[trick_cond, 'contrast'] = 'hard'

trick_cond = df_conds['contrast'].isna()
df_conds.loc[trick_cond, 'contrast'] = 'null'

index = df_conds['task'].tolist().index('FingerTap')
df_conds = df_conds.append({'task': 'FingerTap', 'contrast': 'fingertap-rest'},
                           ignore_index=True)
df_conds = df_conds.reindex(list(range(index + 1)) +
                            [len(df_conds) - 1] +
                            list(range(index + 1, len(df_conds) - 1)))

index = df_conds['task'].tolist().index('ItemRecognition')
df_conds = df_conds.append({'task': 'ItemRecognition', 'contrast': 'encode'},
                           ignore_index=True)
df_conds = df_conds.reindex(list(range(index + 1)) +
                            [len(df_conds) - 1] +
                            list(range(index + 1, len(df_conds) - 1)))

index = df_conds['task'].tolist().index('Checkerboard')
df_conds = df_conds.append({'task': 'Checkerboard',
                            'contrast': 'checkerboard-fixation'},
                           ignore_index=True)
df_conds = df_conds.reindex(list(range(index + 1)) +
                            [len(df_conds) - 1] +
                            list(range(index + 1, len(df_conds) - 1)))

# %%
def tags(tags_lists):
    """
    Extract all tags of a set of contrasts from all_contrasts.tsv,
    clean labels and return their unique tags.
    """
    tags_lists = [tl.replace("'", "") for tl in tags_lists]
    tags_lists = [list(tg.strip('][]').split(',')) for tg in tags_lists]
    # Some cleaning:
    # 1. remove extra-spaces;
    # 2. replace spaces between words by underscores
    tags_clean = []
    for tags_list in tags_lists:
        tag_clean = []
        for tc in tags_list:
            tc = tc.replace(" ","_")
            if tc == '_':
                continue
            if tc[0] == '_':
                tc = tc[1:]
            if tc[-1] == '_':
                tc = tc[:-1]
            tag_clean.append(tc)
        tags_clean.append(tag_clean)
    # Get array with unique tags
    tags_flatten = [item for sublist in tags_clean for item in sublist]
    utags_array = np.array(tags_flatten)
    utags_array = np.unique(utags_array)
    unique_tags = utags_array.tolist()
    return tags_clean, tags_flatten, unique_tags

# %%
def design(feature):
    enc = LabelEncoder().fit(feature)
    feature_label, feature_ = enc.transform(feature), enc.classes_
    dmtx = OneHotEncoder(sparse_output=False).fit_transform(
        feature_label.reshape(-1, 1))
    return dmtx, feature_

# %%
def anova(db, masker):
    """perform a big ANOVA of brain activation with three factors:
    acquisition, subject, contrast"""
    df = db[(db.acquisition == 'ap') | (db.acquisition == 'pa')]

    _cond = df['task'] == 'BiologicalMotion2'
    df_copy = df.copy()  # Create a copy of the DataFrame
    df_copy.loc[_cond, 'contrast'] = df_copy.loc[_cond, 'contrast'] + '_biomo2'
    df = df_copy 

    # make the design matrix
    subject_dmtx, subject_ = design(df.subject)
    contrast_dmtx, contrast_ = design(df.contrast)
    acq_dmtx, acq_ = design(df.acquisition)
    dmtx = np.hstack((subject_dmtx[:, : -1],
                      contrast_dmtx[:, : -1],
                      acq_dmtx[:, : -1],
                      np.ones((len(df), 1))))
    labels = np.hstack((subject_[: -1], contrast_[: -1], acq_[: -1],
                        ['intercept']))
    design_matrix = pd.DataFrame(dmtx, columns=labels)
    _, singular, _ = np.linalg.svd(design_matrix.values, 0)
    dof_subject = len(subject_) - 1
    dof_contrast = len(contrast_) - 1
    dof_acq = len(acq_) - 1

    # fit the model
    second_level_model = SecondLevelModel(mask_img=masker.mask_img_)
    second_level_model = second_level_model.fit(list(df.path.values),
                                                design_matrix=design_matrix)
    subject_map = second_level_model.compute_contrast(
        np.eye(len(labels))[:dof_subject], output_type='z_score')
    contrast_map = second_level_model.compute_contrast(
        np.eye(len(labels))[dof_subject: dof_subject + dof_contrast],
        output_type='z_score')
    acq_map = second_level_model.compute_contrast(
        np.eye(len(labels))[-1 -dof_acq: -1], output_type='z_score')
    subject_map = math_img('img * (img > -8.2095)', img=subject_map)
    contrast_map = math_img('img * (img > -8.2095)', img=contrast_map)
    acq_map =  math_img('img * (img > -8.2095)', img=acq_map)
    return design_matrix, subject_map, contrast_map, acq_map

# %%
def global_similarity(db, masker):
    """Study the global similarity of ffx activation maps"""
    df = db[db.acquisition == 'ffx']

    df_copy = df.copy()    
    _cond = df['task'] == 'BiologicalMotion2'
    df_copy.loc[_cond, 'contrast'] = df_copy.loc[_cond, 'contrast'] + '_biomo2'
    _cond = df['task'] == 'FingerTapping'
    df_copy.loc[_cond, 'contrast'] = df_copy.loc[_cond, 'contrast'] + '_ftap'
    df = df_copy

    X = masker.transform(df.path)
    xcorr = np.corrcoef(X)
    subject_dmtx, subject_ = design(df.subject)
    contrast_dmtx, contrast_ = design(df.contrast)
    scorr = np.dot(subject_dmtx, subject_dmtx.T)
    ccorr = np.dot(contrast_dmtx, contrast_dmtx.T)
    plt.figure(figsize=(7.2, 5))
    ax = plt.axes([0.01, 0.01, .58, .94])
    ax.imshow(xcorr, interpolation='nearest', cmap=plotting.cm.bwr)
    ax.axis('off')
    ax.set_title('Between image correlation', fontdict={'fontsize':14})
    ax = plt.axes([.61, 0.01, .38, .44])
    ax.imshow(scorr, interpolation='nearest', cmap=plotting.cm.bwr)
    ax.axis('off')
    ax.set_title('Correlation due to subject')
    ax = plt.axes([.61, 0.51, .38, .44])
    ax.imshow(ccorr, interpolation='nearest', cmap=plotting.cm.bwr)
    ax.axis('off')
    ax.set_title('Correlation due to contrast')
    plt.savefig(os.path.join(cache, 'similarity.pdf'))

    model = TSNE(n_components=2, random_state=0)
    Y = model.fit_transform(X)
    plt.figure()
    color_code = plt.cm.jet(np.linspace(0, 255, 12).astype(np.int))
    colors = color_code[LabelEncoder().fit_transform(df.subject) - 1]
    plt.scatter(Y[:, 0], Y[:, 1], color=colors)
    plt.show()

# %%
def condition_similarity(db, masker):
    """
    Look at the similarity across conditions, averaged across
    subjects and phase encoding
    """
    df = db[db.acquisition == 'ffx']

    df_copy = df.copy()  # Create a copy of the DataFrame

    _cond = df['task'] == 'BiologicalMotion2'
    df_copy.loc[_cond, 'contrast'] = df_copy.loc[_cond, 'contrast'] + '_biomo2'

    _cond = df['task'] == 'FingerTapping'
    df_copy.loc[_cond, 'contrast'] = df_copy.loc[_cond, 'contrast'] + '_ftap'
    
    df = df_copy
    conditions = df.contrast.unique()
    n_conditions = len(conditions)
    print('The number of elementary contrasts is %s.' % n_conditions)
    correlation = np.zeros((n_conditions, n_conditions))
    X = {}
    unique_subjects = df.subject.unique()
    n_subjects = len(unique_subjects)
    for subject in unique_subjects:
        paths = []
        tasks = []
        task_pos = [0]
        for condition in conditions:
            selection = df[df.subject == subject][df.contrast == condition]
            tasks.append(selection.task.values[-1])
            if len(tasks) > 2 and tasks[-2] != tasks[-1]:
                task_pos.append(float(len(tasks)-1))
            paths.append(selection.path.values[-1])
        x = masker.transform(paths)
        correlation += np.corrcoef(x)
        X[subject] = x
    correlation /= n_subjects
    tasks = np.array(tasks)
    _, idx = np.unique(tasks, return_index=True)
    unique_tasks = tasks[np.sort(idx)]
    nice_tasks = []
    for task in unique_tasks:
        task = task.replace('_', ' ')
        task = task.title()
        nice_tasks.append(task)
    nice_tasks = np.array(nice_tasks)
    # plot with subject correlations
    fig = plt.figure(figsize=(6., 5.))
    ax = plt.axes()
    ax.set_yticks(task_pos)
    ax.set_yticklabels(nice_tasks)
    ax.set_xticks(task_pos)
    ax.set_xticklabels(nice_tasks, rotation=60, ha='right')
    cax = ax.imshow(correlation, interpolation='nearest', vmin=-1,
                    cmap=plotting.cm.cyan_copper)
    # Add colorbar
    # Make sure to specify tick locations to match desired ticklabels
    cbar = fig.colorbar(cax, ticks=[-0.95, 0., .95], shrink=.925)
    # vertically oriented colorbar
    cbar.ax.set_yticklabels(['-1', '0', '1'])
    plt.subplots_adjust(left=.3, top=.99, right=.99, bottom=.275)
    plt.savefig(os.path.join(cache, 'condition_similarity_within' + '_' +
                             suffix + '.png'), dpi=1200)

    tasks_ = np.unique(tasks).tolist()
    df_all_contrasts = pd.read_csv(ALL_CONTRASTS, sep='\t')

    df_copy = df_all_contrasts.copy()  # Create a copy of the DataFrame

    _cond = df_all_contrasts['task'] == 'BiologicalMotion2'
    df_copy.loc[_cond, 'contrast'] = df_copy.loc[_cond, 'contrast'] + '_biomo2'

    _cond = df_copy['task'] == 'FingerTapping'
    df_copy.loc[_cond, 'contrast'] = df_copy.loc[_cond, 'contrast'] + '_ftap'

    trick_cond = df_copy['contrast'].isna()
    df_copy.loc[trick_cond, 'contrast'] = 'null_ftap'

    df_all_contrasts = df_copy

    # Get contrasts and tags lists
    contrasts_list = []
    all_tags = []
    pos_task = {}
    for tk in tasks_:
        num_con = len(contrasts_list)
        contrasts_list.extend(df_all_contrasts[df_all_contrasts.task == \
                                               tk].contrast.tolist())
        all_tags.extend(df_all_contrasts[df_all_contrasts.task == \
                                         tk].tags.tolist())
        pos_task[tk] = num_con
    tgs_clean, tgs_flatten, unique_tgs = tags(all_tags)

    print('The total number of contrasts is %s' % len(contrasts_list))
    print('The number of cognitive components present ' + \
              'in all contrasts is %s.' % len(unique_tgs))

    # Create occurrences matrix
    occur_mtx = []
    for tlist in tgs_clean:
        occur = []
        for component in unique_tgs:
            if component in tlist:
                occur.append(1)
            else:
                occur.append(0)
        occur_mtx.append(occur)
    df = pd.DataFrame(occur_mtx, columns=unique_tgs, index=contrasts_list)

    plt.figure()
    plt.imshow(df)
    for task, position in pos_task.items():
        plt.text(-0.5, position, task, ha='right', va='center',
                 color='black')
    plt.yticks([])
    plt.savefig(os.path.join(cache, 'tags_occurrence' + '_' +
                             suffix + '.png'), dpi=1200)


    cog_model = np.zeros((n_conditions, len(df.columns)))
    for i, condition in enumerate(conditions):
        if not df[df.index == condition].values.tolist():
            msg = 'Condition "%s" not found!' % condition
            warnings.warn(msg)
        else:
            cog_model[i] = df[df.index == condition].values
    cog_comp = [ccomp for ccomp in cog_model.T if np.any(ccomp)]
    print('The number of cognitive components present only in ' + \
          'the elementary contrasts is %s.' % len(cog_comp))
    ccorrelation = np.corrcoef(cog_model)
    fig = plt.figure(figsize=(6., 5.))
    ax = plt.axes()
    ax.set_yticks(task_pos)
    ax.set_yticklabels(nice_tasks)
    ax.set_xticks(task_pos)
    ax.set_xticklabels(nice_tasks, rotation=60, ha='right')
    cax = ax.imshow(ccorrelation, interpolation='nearest', vmin=-1,
                    cmap=plotting.cm.cyan_copper)
    # Add colorbar
    # Make sure to specify tick locations to match desired ticklabels
    cbar = fig.colorbar(cax, ticks=[-0.95, 0., .95], shrink=.925)
    # vertically oriented colorbar
    cbar.ax.set_yticklabels(['-1', '0', '1'])
    plt.subplots_adjust(left=.3, top=.99, right=.99, bottom=.275)
    plt.savefig(os.path.join(cache, 'condition_similarity_cognitive' + '_' +
                             suffix + '.png'), dpi=1200)
    # plt.show()
    x = np.triu(correlation, 1)
    y = np.triu(ccorrelation, 1)
    x = x[x != 0]
    y = y[y != 0]
    print('pearson', st.pearsonr(x,y))
    print('spearman', st.spearmanr(x,y))
    # PearsonR(statistic=0.41868416057824975, pvalue=1.3771091641528076e-73)
    # spearman (statistic=0.22693191657859985, pvalue=2.0146645657106837e-21)

# %%
def condition_similarity_across_subjects(db, masker):
    """
    Look at the similarity across conditions,
    averaged across subjects and phase encoding
    """
    df = db[db.acquisition == 'ffx']

    df_copy = df.copy()  # Create a copy of the DataFrame

    _cond = df['task'] == 'BiologicalMotion2'
    df_copy.loc[_cond, 'contrast'] = df_copy.loc[_cond, 'contrast'] + '_biomo2'

    _cond = df['task'] == 'FingerTapping'
    # df_copy = df.copy()
    df_copy.loc[_cond, 'contrast'] = df_copy.loc[_cond, 'contrast'] + '_ftap'
    df = df_copy

    conditions = df.contrast.unique()
    n_conditions = len(conditions)
    correlation = np.zeros((n_conditions, n_conditions))
    correlations = {}
    unique_subjects = df.subject.unique()
    n_voxels =  np.asanyarray(masker.mask_img_.dataobj).sum()
    x_sum = np.zeros((n_conditions, n_voxels))
    X = []
    for subject in unique_subjects:
        paths = []
        tasks = []
        for condition in conditions:
            selection = df[df.subject == subject][df.contrast == condition]
            tasks.append(selection.task.values[-1])
            paths.append(selection.path.values[-1])
        x = masker.transform(paths)
        correlation = np.corrcoef(x)
        x_sum += x
        correlations[subject] = correlation
        X.append(x)

    X = np.array(X) 
    tasks = np.array(tasks) 
    unique_tasks = np.unique(tasks)
    task_pos = np.array(
        [np.mean(np.where(tasks == task)[0]) for task in unique_tasks])
    ## Ugly trick, but just to maks the labels readable :-(
    #task_pos = np.array([25.5, 18.5, 12. ,4.5, 40, 34, 28.5, 37., 43, 31.5,
    #                     47.5, 55. ])
    nice_tasks = np.array([task.replace('_', ' ') for task in unique_tasks])
    #mean_correlation = np.mean(np.array(correlations.values()), 0)

    def complexity(correlation):
        _, s, _ = np.linalg.svd(correlation, 0)
        return(np.log(s).sum())

    def bootstrap_complexity_correlation_mean(X, n_bootstrap=100):
        """X is meant to be an array(n_subjects, n_voxels, n_contrasts)"""
        complexities = []
        for _ in range(n_bootstrap):
            X_sum = np.zeros_like(X[0])
            indexes = np.random.randint(0, X.shape[0], X.shape[0])
            for i in indexes:
                X_sum += X[i]
            correlation_mean = np.corrcoef(X_sum)
            complexities.append(complexity(correlation_mean))
        return complexities

    def bootstrap_complexity_mean_correlation(correlations, n_bootstrap=100):
        """X is meant to be an array(n_subjects, n_voxels, n_contrasts)"""
        complexities = []
        for _ in range(n_bootstrap):
            indexes = np.random.randint(0, correlations.shape[0],
                                        correlations.shape[0])
            mean_correlation = np.mean(correlations[indexes], 0)
            complexities.append(complexity(mean_correlation))
        return complexities

    # plot with subject correlations
    fig = plt.figure(figsize=(12., 10))
    #fig, ax = plt.subplots()
    for i, subject in enumerate(unique_subjects):
        ax = plt.subplot(3, 4, i + 1)
        ax.axis('off')
        #ax.set_yticks(task_pos)
        #ax.set_yticklabels(nice_tasks)
        #ax.set_xticks(task_pos)
        #ax.set_xticklabels(nice_tasks, rotation=60, ha='right')
        cax = ax.imshow(correlations[subject], interpolation='nearest',
                        cmap=plotting.cm.bwr)
    ## Add colorbar,
    # make sure to specify tick locations to match desired ticklabels
    #cbar = fig.colorbar(cax, ticks=[0, .95])
    #cbar.ax.set_yticklabels(['0', '1'])  # vertically oriented colorbar
    plt.subplots_adjust(left=.02, top=.98, right=.98, bottom=.05)
    plt.savefig(os.path.join(cache, 'condition_similarities.png'))
    correlation_mean = np.corrcoef(x_sum)
    fig = plt.figure(figsize=(6., 5))
    ax = plt.axes()
    ax.set_yticks(task_pos)
    ax.set_yticklabels(nice_tasks)
    ax.set_xticks(task_pos)
    ax.set_xticklabels(nice_tasks, rotation=60, ha='right')
    cax = ax.imshow(correlation_mean, interpolation='nearest',
                    cmap=plotting.cm.bwr)
    # Add colorbar
    # make sure to specify tick locations to match desired ticklabels
    cbar = fig.colorbar(cax, ticks=[0, 0.95])
    cbar.ax.set_yticklabels(['0', '1'])  # vertically oriented colorbar
    plt.subplots_adjust(left=.25, top=.99, right=.99, bottom=.22)
    #plt.savefig(os.path.join(cache, 'condition_similarity_of_mean.pdf'))
    plt.savefig(os.path.join(cache, 'condition_similarity_of_mean.png'))
    C1 = bootstrap_complexity_correlation_mean(X, n_bootstrap=100)
    C2 = bootstrap_complexity_mean_correlation(np.array(list(
                                               correlations.values())),
                                               n_bootstrap=100)
    plt.figure(figsize=(6, 3))
    bp = plt.boxplot(C1, vert=0, positions=[0], widths=.8)
    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians',
                    'caps']: plt.setp(bp[element], color='g', linewidth=3)
    bp = plt.boxplot(C2, vert=0, positions=[1], widths=.8)
    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians',
                    'caps']: plt.setp(bp[element], color='r', linewidth=3)
    plt.yticks([0, 1], ['correlation of average', 'mean correlation'])
    plt.axis([-130, -50, -.5, 1.5])
    plt.title('Complexity of correlation matrices')
    plt.subplots_adjust(left=.35, bottom=.1, right=.95, top=.9)
    plt.savefig(os.path.join(cache, 'correlation_complexity.pdf'))


# %%
if __name__ == '__main__':
    db = data_parser(derivatives=SMOOTH_DERIVATIVES, subject_list = PTS,
                     task_list=TASKS, conditions=df_conds)
    
    mask_gm = nib.load(os.path.join(DERIVATIVES, 'group', 'anat',
                                    'gm_mask.nii.gz'))
    masker = NiftiMasker(mask_img=mask_gm, memory=mem).fit()
    
    # %%
    # Compute the ANOVAs
    design_matrix, subject_map, contrast_map, acq_map = anova(db, masker)
    # Store them...
    subject_map.to_filename(os.path.join(cache, 'subject_effect' + '_' +
                                         suffix + '.nii.gz'))
    contrast_map.to_filename(os.path.join(cache, 'condition_effect' + '_' +
                                          suffix + '.nii.gz'))
    acq_map.to_filename(os.path.join(cache, 'acq_effect' + '_' + suffix +
                                     '.nii.gz'))
    # ...or load them
    # subject_map = os.path.join(cache,
    #                           'subject_effect' + '_' + suffix + '.nii.gz')
    # contrast_map = os.path.join(cache,
    #                            'condition_effect' + '_' + suffix + '.nii.gz')
    # acq_map = os.path.join(cache, 'acq_effect' + '_' + suffix + '.nii.gz')

    # Plots
    _, threshold_ = threshold_stats_img(subject_map, alpha=.05, height_control='fdr')
    sub_effect = plotting.plot_stat_map(subject_map, cut_coords=[10, -50, 10],
                                        threshold=threshold_, vmax=37,
                                        title='Subject effect',
                                        draw_cross = False)
    sub_effect.savefig(os.path.join(
        cache, 'subject_effect' + '_' + suffix + '.png'), dpi=1200)
    #
    _, threshold_ = threshold_stats_img(contrast_map, alpha=.05, height_control='fdr')
    cond_effect = plotting.plot_stat_map(contrast_map, cut_coords=[10, -50, 10],
                                         threshold=threshold_,
                                         title='Condition effect',
                                         draw_cross = False)
    cond_effect.savefig(os.path.join(
        cache, 'condition_effect' + '_' + suffix + '.png'), dpi=1200)
    #
    _, threshold_ = threshold_stats_img(acq_map, alpha=.05, height_control='fdr')
    phase_effect = plotting.plot_stat_map(acq_map,
                           threshold=threshold_, vmax=37,
                           title='Phase-encoding effect', draw_cross = False)
    phase_effect.savefig(os.path.join(
        cache, 'acq_effect' + '_' + suffix + '.png'), dpi=1200)
    
    # %%
    # Other analyses 
    global_similarity(db, masker)
    condition_similarity(db, masker)
    condition_similarity_across_subjects(db, masker)
    plt.show()