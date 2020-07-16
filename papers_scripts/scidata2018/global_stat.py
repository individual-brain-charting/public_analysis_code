"""
Global statistical analysis of SPM maps produced by first-level analyis  of the dataset.
* tease out effect of subject, task and phase encoding direction
* Study global similarity effects  

Author: Bertrand Thirion, 2017
"""
import glob
import os
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.input_data import NiftiMasker
from joblib import Memory, Parallel, delayed
from nilearn import plotting
from nilearn.image import math_img
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from nistats.thresholding import map_threshold
import matplotlib.pyplot as plt

from ibc_public.utils_data import (
    CONDITIONS, data_parser, SUBJECTS, DERIVATIVES, SMOOTH_DERIVATIVES)

cache = '/neurospin/tmp/bthirion'
mem = Memory(cachedir=cache, verbose=0)


def design(feature):
    enc = LabelEncoder().fit(feature)
    feature_label, feature_ = enc.transform(feature), enc.classes_
    dmtx = OneHotEncoder(sparse=False).fit_transform(feature_label.reshape(-1, 1))
    return dmtx, feature_ 


def anova(db, masker):
    """perform a big ANOVA of brain activation with three factors:
    acquisition, subject, contrast"""
    df = db[(db.acquisition == 'ap') | (db.acquisition == 'pa')]

    # make the design matrix
    subject_dmtx, subject_ = design(df.subject)
    contrast_dmtx, contrast_ = design(df.contrast)
    acq_dmtx, acq_ = design(df.acquisition)
    dmtx = np.hstack((subject_dmtx[:, : -1],
                      contrast_dmtx[:, : -1],
                      acq_dmtx[:, : -1],
                      np.ones((len(df), 1)))) 
    labels = np.hstack((subject_[: -1], contrast_[: -1], acq_[: -1], ['intercept']))
    design_matrix = pd.DataFrame(dmtx, columns=labels)
    _, singular, _ = np.linalg.svd(design_matrix.values, 0)
    dof_subject = len(subject_) - 1
    dof_contrast = len(contrast_) - 1
    dof_acq = len(acq_) - 1
    
    # fit the model
    from nistats.second_level_model import SecondLevelModel
    second_level_model = SecondLevelModel(mask=masker.mask_img_)
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


def global_similarity(db, masker):
    """Study the global similarity of ffx activation maps"""
    import matplotlib.pyplot as plt
    df = db[db.acquisition == 'ffx']
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
    plt.savefig(os.path.join('output', 'similarity.pdf'))

    from sklearn.manifold import TSNE
    model = TSNE(n_components=2, random_state=0)
    Y = model.fit_transform(X)
    plt.figure()
    color_code = plt.cm.jet(np.linspace(0, 255, 12).astype(np.int))
    colors = color_code[LabelEncoder().fit_transform(df.subject) - 1]
    plt.scatter(Y[:, 0], Y[:, 1], color=colors)
    plt.show()


def condition_similarity(db, masker):
    """ Look at the similarity across conditions, averaged across subjects and phase encoding"""
    df = db[db.acquisition == 'ffx']
    conditions = df.contrast.unique()
    n_conditions = len(conditions)
    correlation = np.zeros((n_conditions, n_conditions))
    X = {}
    unique_subjects = df.subject.unique()
    n_subjects = len(unique_subjects)
    for subject in unique_subjects:
        paths = []
        tasks = []
        for condition in conditions:
            selection = df[df.subject == subject][df.contrast == condition]
            tasks.append(selection.task.values[-1])
            paths.append(selection.path.values[-1])
        x = masker.transform(paths)
        correlation += np.corrcoef(x)
        X[subject] = x

    correlation /= n_subjects
    tasks = np.array(tasks) 
    unique_tasks = np.unique(tasks)
    task_pos = np.array(
        [np.mean(np.where(tasks == task)[0]) for task in unique_tasks])
    task_pos = np.array([25.5, 18.5, 12. ,  4.5, 40, 34, 28.5, 37. , 43, 31.5, 47.5,
                      55. ]) ## Ugly trick, but just to maks the labels readable :-(
    nice_tasks = np.array([task.replace('_', ' ') for task in unique_tasks])

    # plot with subject correlations
    fig = plt.figure(figsize=(6., 5))
    #fig, ax = plt.subplots()
    ax = plt.axes()
    ax.set_yticks(task_pos)
    ax.set_yticklabels(nice_tasks)
    ax.set_xticks(task_pos)
    ax.set_xticklabels(nice_tasks, rotation=60, ha='right')
    cax = ax.imshow(correlation, interpolation='nearest', cmap=plotting.cm.bwr)
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    cbar = fig.colorbar(cax, ticks=[0, .95])
    cbar.ax.set_yticklabels(['0', '1'])  # vertically oriented colorbar
    plt.subplots_adjust(left=.25, top=.99, right=.99, bottom=.22)
    plt.savefig(os.path.join('output', 'condition_similarity_within.pdf'))
    
    # plot cross-subject correlation
    correlation_ = np.zeros((n_conditions, n_conditions))
    for i in range(n_subjects):
        for j in range(i):
            X_ = np.vstack((X[unique_subjects[i]], X[unique_subjects[j]]))
            correlation_ += np.corrcoef(X_)[n_conditions:, :n_conditions]
            
    correlation_ /= (n_subjects * (n_subjects - 1) * .5)
    fig = plt.figure(figsize=(6., 5))
    ax = plt.axes()
    ax.set_yticks(task_pos)
    ax.set_yticklabels(nice_tasks)
    ax.set_xticks(task_pos)
    ax.set_xticklabels(nice_tasks, rotation=60, ha='right')
    cax = ax.imshow(correlation_, interpolation='nearest', cmap=plotting.cm.bwr)
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    cbar = fig.colorbar(cax, ticks=[0, .4])
    cbar.ax.set_yticklabels(['0', '.4'])  # vertically oriented colorbar
    plt.subplots_adjust(left=.25, top=.99, right=.99, bottom=.22)
    plt.savefig(os.path.join('output', 'condition_similarity_across.pdf'))

    # similarity at the level of conditions
    cognitive_atlas = 'cognitive_atlas.csv'
    df = pd.DataFrame().from_csv(cognitive_atlas, index_col=1, sep='\t')
    df = df.fillna(0)
    df = df.drop('Tasks', 1)
    cog_model = np.zeros((n_conditions, len(df.columns)))
    for i, condition in enumerate(conditions):
        cog_model[i] = df[df.index == condition].values
        print(condition, [df.columns[i] for i in range(50)
                          if df[df.index == condition].values[0][i]])

    ccorrelation = np.corrcoef(cog_model)
    fig = plt.figure(figsize=(6., 5))
    ax = plt.axes()
    ax.set_yticks(task_pos)
    ax.set_yticklabels(nice_tasks)
    ax.set_xticks(task_pos)
    ax.set_xticklabels(nice_tasks, rotation=60, ha='right')
    cax = ax.imshow(ccorrelation, interpolation='nearest', cmap=plotting.cm.bwr)
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    cbar = fig.colorbar(cax, ticks=[0, 0.95])
    cbar.ax.set_yticklabels(['0', '1'])  # vertically oriented colorbar
    plt.subplots_adjust(left=.25, top=.99, right=.99, bottom=.22)
    plt.savefig(os.path.join('output', 'condition_similarity_cognitive.pdf'))
    plt.show()
    x = np.triu(correlation, 1)
    y = np.triu(ccorrelation, 1)
    x = x[x != 0]
    y = y[y != 0]
    import scipy.stats as st
    print('pearson', st.pearsonr(x,y))
    print('spearman', st.spearmanr(x,y))
    
    
def condition_similarity_across_subjects(db, masker):
    """ Look at the similarity across conditions, averaged across subjects and phase encoding"""
    df = db[db.acquisition == 'ffx']
    conditions = df.contrast.unique()
    n_conditions = len(conditions)
    correlation = np.zeros((n_conditions, n_conditions))
    correlations = {}
    unique_subjects = df.subject.unique()
    n_voxels =  masker.mask_img_.get_data().sum()
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
    task_pos = np.array([25.5, 18.5, 12. ,  4.5, 40, 34, 28.5, 37. , 43, 31.5, 47.5,
                      55. ]) ## Ugly trick, but just to maks the labels readable :-(
    nice_tasks = np.array([task.replace('_', ' ') for task in unique_tasks])
    mean_correlation = np.mean(np.array(correlations.values()), 0)

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
            indexes = np.random.randint(0, correlations.shape[0], correlations.shape[0])
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
    ## Add colorbar, make sure to specify tick locations to match desired ticklabels
    #cbar = fig.colorbar(cax, ticks=[0, .95])
    #cbar.ax.set_yticklabels(['0', '1'])  # vertically oriented colorbar
    plt.subplots_adjust(left=.02, top=.98, right=.98, bottom=.05)
    plt.savefig(os.path.join('output', 'condition_similarities.pdf'))
    correlation_mean = np.corrcoef(x_sum)
    fig = plt.figure(figsize=(6., 5))
    ax = plt.axes()
    ax.set_yticks(task_pos)
    ax.set_yticklabels(nice_tasks)
    ax.set_xticks(task_pos)
    ax.set_xticklabels(nice_tasks, rotation=60, ha='right')
    cax = ax.imshow(correlation_mean, interpolation='nearest', cmap=plotting.cm.bwr)
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    cbar = fig.colorbar(cax, ticks=[0, 0.95])
    cbar.ax.set_yticklabels(['0', '1'])  # vertically oriented colorbar
    plt.subplots_adjust(left=.25, top=.99, right=.99, bottom=.22)
    plt.savefig(os.path.join('output', 'condition_similarity_of_mean.pdf'))
    C1 = bootstrap_complexity_correlation_mean(X, n_bootstrap=100)
    C2 = bootstrap_complexity_mean_correlation(np.array(correlations.values()),
                                               n_bootstrap=100)
    plt.figure(figsize=(6, 3))
    bp = plt.boxplot(C1, vert=0, positions=[0], widths=.8)
    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']: plt.setp(bp[element], color='g', linewidth=3)
    bp = plt.boxplot(C2, vert=0, positions=[1], widths=.8)
    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']: plt.setp(bp[element], color='r', linewidth=3)
    plt.yticks([0, 1], ['correlation of average', 'mean correlation'])
    plt.axis([-95, -40, -.5, 1.5])
    plt.title('Complexity of correlation matrices')
    plt.subplots_adjust(left=.35, bottom=.1, right=.95, top=.9)
    plt.savefig(os.path.join('output', 'correlation_complexity.pdf'))
    
    
if __name__ == '__main__':
    db = data_parser(derivatives=SMOOTH_DERIVATIVES)
    mask_gm = nib.load(os.path.join(DERIVATIVES, 'group', 'anat', 'gm_mask.nii.gz'))
    masker = NiftiMasker(mask_img=mask_gm, memory=mem).fit()
    """
    design_matrix, subject_map, contrast_map, acq_map = anova(db, masker)
    subject_map.to_filename(os.path.join('output', 'subject_effect.nii.gz'))
    contrast_map.to_filename(os.path.join('output', 'contrast_effect.nii.gz'))
    acq_map.to_filename(os.path.join('output', 'acq_effect.nii.gz'))
    # 
    _, threshold_ = map_threshold(subject_map, threshold=.05, height_control='fdr')
    plotting.plot_stat_map(subject_map, cut_coords=[10, -50, 10],
                           threshold=threshold_, title='Subject effect',
                           output_file=os.path.join('output', 'subject_effect.pdf'))
    #
    _, threshold_ = map_threshold(contrast_map, threshold=.05, height_control='fdr')
    plotting.plot_stat_map(contrast_map, cut_coords=[10, -50, 10],
                           threshold=threshold_, title='Condition effect',
                           output_file=os.path.join('output', 'contrast_effect.pdf'))
    #
    _, threshold_ = map_threshold(acq_map, threshold=.05, height_control='fdr')
    plotting.plot_stat_map(acq_map,
                           threshold=threshold_, title='Phase encoding effect',
                           output_file=os.path.join('output', 'acq_effect.pdf'))
    
    global_similarity(db, masker)
    """
    condition_similarity_across_subjects(db, masker)
    plt.show()
