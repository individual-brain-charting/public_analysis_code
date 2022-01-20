"""
Script to compare intra- and inter-subject variability
ie ap- / pa- images vs cross-subjects images

Authors: Bertrand Thirion, Ana Luisa Pinho, February 2020

Compatibility: Python 3.5

"""
import os
import json
import numpy as np

from math import *

from joblib import Memory
from nilearn.input_data import NiftiMasker
from ibc_public.utils_data import (
    data_parser, SMOOTH_DERIVATIVES, DERIVATIVES, SUBJECTS, CONTRASTS,
    LABELS)
import ibc_public
import matplotlib
# matplotlib.use('Agg')
from matplotlib import pyplot as plt


def flatten(li):
    return sum(([x] if not isinstance(x, list) else flatten(x)
                for x in li), [])

# caching
main_parent_dir = '/neurospin/tmp/'
alt_parent_dir = '/storage/tompouce/'
main_dir = 'bthirion'
alt_dir = 'agrilopi'

if os.path.exists(main_parent_dir):
    cache = main_parent_dir + alt_dir
else:
    cache = alt_parent_dir + main_dir

# output directory
mem = Memory(cachedir=cache, verbose=0)
write_dir = cache
if not os.path.exists(write_dir):
    os.mkdir(write_dir)

# Access to the data
subject_list = SUBJECTS

task_list = ['ArchiEmotional', 'ArchiSocial', 'ArchiSpatial',
             'ArchiStandard', 'HcpEmotion', 'HcpGambling',
             'HcpLanguage', 'HcpMotor', 'HcpRelational',
             'HcpSocial', 'HcpWm',
             'RSVPLanguage']

# BIDS conversion of task names
# Load dictionary file
with open(os.path.join('bids_postprocessed.json'), 'r') as f:
    task_dic = json.load(f)

TASKS = [task_dic[tkey] for tkey in task_list]
TASKS = flatten(TASKS)

df = data_parser(derivatives=SMOOTH_DERIVATIVES, subject_list=SUBJECTS,
                 conditions=CONTRASTS, task_list=TASKS)

# Mask of the grey matter across subjects
_package_directory = os.path.dirname(
    os.path.abspath(ibc_public.utils_data.__file__))
mask_gm = os.path.join(
    _package_directory, '../ibc_data', 'gm_mask_1_5mm.nii.gz')

data_dir = SMOOTH_DERIVATIVES
masker = NiftiMasker(mask_img=mask_gm, memory=cache).fit()
fig = plt.figure(figsize=(17, 11))
q = 0
column = 0

inter_allmeans = []
intra_allmeans = []
for j, task in enumerate(task_list):
    task_df = df[df.task == task]
    contrasts = task_df.contrast.unique()
    inter_correlations = []
    intra_correlations = []
    if task == 'ArchiStandard':
        order = [2, 1, 3, 0, 5, 4]
        contrasts = [contrasts[c] for c in order]
        contrasts = contrasts[1:]
    elif task == 'ArchiSpatial':
        order = [4, 3, 0, 1, 2]
        contrasts = [contrasts[c] for c in order]
        contrasts = contrasts[2:]
    elif task == 'ArchiSocial':
        order = [5, 4, 3, 2, 0, 6, 1]
        contrasts = [contrasts[c] for c in order]
        contrasts = contrasts[4:]
    elif task == 'ArchiEmotional':
        order = [2, 0, 1, 3]
        contrasts = [contrasts[c] for c in order]
        contrasts = contrasts[2:]
    elif task == 'HcpEmotion':
        contrasts = contrasts[1:]
    elif task in ['HcpGambling', 'HcpLanguage',
                  'HcpRelational', 'HcpSocial']:
        order = [1, 0]
        contrasts = [contrasts[c] for c in order]
        contrasts = contrasts[1:]
    elif task in ['HcpMotor', 'HcpWm']:
        order = [4, 3, 2, 1, 0]
        contrasts = [contrasts[c] for c in order]
    elif task == 'RSVPLanguage':
        order = [7, 4, 0, 6, 3, 5, 1, 2]
        contrasts = [contrasts[c] for c in order]
        contrasts = contrasts[3:]
    n_contrasts = len(contrasts)
    for contrast in contrasts:
        imgs = task_df[task_df.contrast == contrast][df.acquisition == 'ffx'].path.values
        X = masker.transform(imgs)
        corr_matrix = np.triu(np.corrcoef(X), 1)
        correlations_ = corr_matrix[corr_matrix != 0]
        inter_correlations.append(correlations_)
        subjects = task_df[task_df.contrast == contrast].subject.unique()
        correlations_ = []
        for subject in subjects:
            imgs = task_df[task_df.contrast == contrast]\
                        [(task_df.acquisition == 'ap') + (task_df.acquisition == 'pa')]\
                        [task_df.subject ==subject].path.values
            if len(imgs) == 0:
                print(contrast, 0)
            else:
                print(contrast, len(imgs))
                X = masker.transform(imgs)
                corr_matrix = np.triu(np.corrcoef(X), 1)
                correlations_.append(corr_matrix[corr_matrix != 0])

        intra_correlations.append(np.hstack(correlations_))

    if len(intra_correlations) == 0:
        intra_correlations = [0]
    inter_correlations = np.array(inter_correlations)
    intra_correlations = np.array(intra_correlations)

    # Create arrays to compute means across contrasts
    inter_allmeans.extend(inter_correlations.mean(1))
    intra_allmeans.extend(intra_correlations.mean(1))

    # Define subplot box for bar charts and its pos in the fig
    ax = plt.axes([.4175 + .474 * column, q * .048 + .1125, .09,
                   n_contrasts * .049])
    # Compute 95% confidence interval
    ci_95_inter = []
    ci_95_intra = []
    ci_95_inter = 1.96 * inter_correlations.std(1)/sqrt(len(subjects))
    ci_95_intra = 1.96 * intra_correlations.std(1)/sqrt(len(subjects))
    ax.barh(np.arange(n_contrasts), inter_correlations.mean(1), .425,
            xerr=ci_95_inter, error_kw=dict(capsize=2, captick=3),
            color='r', alpha=.5)
    ax.barh(np.arange(n_contrasts) + .425, intra_correlations.mean(1), .425,
            xerr=ci_95_intra, error_kw=dict(capsize=2, captick=3),
            color='k', alpha=.5)
    ax.set_yticks(np.arange(n_contrasts))
    new_labels = []
    for i in range(n_contrasts):
        new_labels.append(LABELS[contrasts[i]][1].values[0] + ' vs. ' +
                          LABELS[contrasts[i]][0].values[0])
    ax.set_yticklabels(new_labels)
    if j == 1:
        ax.spines['bottom'].set_position(('axes', .005))
    elif j in [2, 5, 11]:
        ax.spines['bottom'].set_position(('axes', .01))
    elif j in [3, 4]:
        ax.spines['bottom'].set_position(('axes', .008))
    elif j == 6:
        ax.spines['bottom'].set_position(('axes', .0233))
    elif j == 7:
        ax.spines['bottom'].set_position(('axes', -.03))
    elif j == 8:
        ax.spines['bottom'].set_position(('axes', .012))
    elif j == 9:
        ax.spines['bottom'].set_position(('axes', .019))
    elif j == 10:
        ax.spines['bottom'].set_position(('axes', .02))

    if q > 0:
        ax.tick_params(length=1.5)

        ax.set_xticklabels('')
    ax.xaxis.set_ticks(np.arange(0, .6, .25))
    ax.tick_params(labelsize=20)
    q += n_contrasts
    if j == 5:
        q = 0
        column = 1

# Compute means across all contrasts and print
inter_allmeans = np.array(inter_allmeans)
intra_allmeans = np.array(intra_allmeans)

print(inter_allmeans.mean(), intra_allmeans.mean())

# Define box for text and its pos in the fig
from matplotlib.patches import Rectangle
ax = plt.axes([0.0, .88, .49, .1])
ax.add_patch(Rectangle((.5, .0), .1, .25, alpha=.5, color='r',
             transform=ax.transAxes))
ax.add_patch(Rectangle((.5, .5), .1, .25, alpha=.5, color='k',
                          transform=ax.transAxes))
ax.text(.61, .1, 'inter-subject correlation', fontsize=20)
ax.text(.61, .6, 'intra-subject correlation', fontsize=20)

plt.axis('off')
ax = plt.axes([0, 0, .99, .035])
plt.text(.3, .75,
         'Intra- and inter- subject correlation of brain maps',
         weight='bold', fontsize='26')
plt.axis('off')
plt.savefig(os.path.join(cache, 'consistency_intra_inter.pdf'), dpi=300)
