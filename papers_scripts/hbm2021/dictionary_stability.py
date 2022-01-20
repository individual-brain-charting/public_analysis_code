"""
Script that learns dictionary of functional loadings on the acquired data
fpr ap and pa images respectively and then compares the outcome.

Author: Bertrand Thirion

Last update: June 2020

Compatibility: Python 3.5

"""

import os
import json
import matplotlib.pyplot as plt
from nilearn.input_data import NiftiMasker
import nibabel as nib
import numpy as np
from joblib import Memory
# from utils_group_analysis import sorted_contrasts
from ibc_public.utils_data import (
    data_parser, SMOOTH_DERIVATIVES, DERIVATIVES, SUBJECTS, CONTRASTS)
import ibc_public
from utils_dictionary import make_dictionary


def flatten(li):
    return sum(([x] if not isinstance(x, list) else flatten(x)
                for x in li), [])


def make_labels(contrasts, task_list):
    labels = {}
    for i in range(len(contrasts)):
        if CONTRASTS.task[i] in task_list:
            labels[CONTRASTS.contrast[i]] = [contrasts['left label'][i],
                                             contrasts['right label'][i]]
    return labels


if 1:
    cache = '/neurospin/tmp/bthirion'
else:
    cache = '/storage/workspace/bthirion/'

write_dir = '/storage/store/work/agrilopi/individual_topographies'
if not os.path.exists(write_dir):
    os.mkdir(write_dir)

mem = Memory(cachedir=write_dir, verbose=0)
subject_list = SUBJECTS
n_subjects = len(subject_list)

# Access to the data
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

# Mask of the grey matter across subjects
db = data_parser(derivatives=SMOOTH_DERIVATIVES, subject_list=subject_list,
                 conditions=CONTRASTS, task_list=TASKS)
_package_directory = os.path.dirname(
    os.path.abspath(ibc_public.utils_data.__file__))
mask_gm = os.path.join(
    _package_directory, '../ibc_data', 'gm_mask_1_5mm.nii.gz')
n_voxels = nib.load(mask_gm).get_data().sum()

# reference image for display
mean_anat = os.path.join(DERIVATIVES, 'group', 'anat', 'T1avg.nii.gz')

# Gather a set of images organized along subjects and contrasts dimensions
# Extract a subset of images from the database
df = db[db.task.isin(task_list)]
df = df.sort_values(by=['subject', 'task', 'contrast'])

contrasts = df.contrast.unique()
n_contrasts = len(contrasts)
ap_paths = []
pa_paths = []
for contrast in contrasts:
    for subject in subject_list:
        mask = (df.contrast == contrast).values *\
               (df.subject == subject).values *\
               (df.acquisition == 'ap')
        if len(df[mask]) == 0:
            print(subject, contrast)
        ap_paths.append(df[mask].path.values[-1])
        mask = (df.contrast == contrast).values *\
               (df.subject == subject).values *\
               (df.acquisition == 'pa')
        if len(df[mask]) == 0:
            print(subject, contrast)
        pa_paths.append(df[mask].path.values[-1])

# image masking
masker = NiftiMasker(
    mask_img=mask_gm, memory=write_dir, smoothing_fwhm=None).fit()
X1 = masker.transform(ap_paths).reshape(
    n_contrasts, int(n_subjects * n_voxels))
X2 = masker.transform(pa_paths).reshape(
    n_contrasts, int(n_subjects * n_voxels))

# learn a dictionary of elements
n_components = 20
alpha = .6
X1[np.isnan(X1)] = 0
X2[np.isnan(X2)] = 0

dictionary1, components_1 = make_dictionary(
    X1, n_components=n_components, alpha=alpha, write_dir=write_dir,
    contrasts=contrasts, method='multitask', l1_ratio=.25)
dictionary2, components_2 = make_dictionary(
    X2, n_components=n_components, alpha=alpha, write_dir=write_dir,
    contrasts=contrasts, method='multitask', l1_ratio=.25)

components1 = np.reshape(components_1, (n_subjects, n_voxels, n_components))
mean_components1 = np.median(components1, 0).T
components2 = np.reshape(components_2, (n_subjects, n_voxels, n_components))
mean_components2 = np.median(components2, 0).T

print(np.sum(components1 != 0) / components1.size)
print(np.sum(components2 != 0) / components2.size)

# idenntify the correspondence of the components
K = np.corrcoef(mean_components1, mean_components2)[:n_components,
                                                    n_components:]
from scipy.optimize import linear_sum_assignment
indexes = linear_sum_assignment(1 - K)
plt.imshow(K[:, indexes[1]], interpolation='nearest')

intra = []
inter = []
for i in range(n_subjects):
    K = np.corrcoef(components1[i].T, components2[i].T)[:n_components,
                                                        n_components:]
    intra.append(np.trace(K[:, indexes[1]]))
    for j in range(i):
        K = np.corrcoef(components1[i].T, components2[j].T)[:n_components,
                                                            n_components:]
        inter.append(np.trace(K[:, indexes[1]]))

intra = np.array(intra) / n_components
inter = np.array(inter) / n_components

print(intra, inter)

# ############### PLOT ###############

# ax1 --> Main subplot
# ax2 --> Subplot inside subplot
fig, ax1 = plt.subplots(figsize=(20, 5))
ax2 = plt.axes([.1575,.19,.83,.6])

# General settings of the horizontal boxplots
boxprops = dict(linewidth=3.)
flierprops = dict(marker='+', markersize=20., markeredgewidth=3.)
whiskerprops  = dict(linewidth=3.)
capprops = dict(linewidth=3.)
bplots = ax2.boxplot([intra, inter], patch_artist=True, flierprops=flierprops,
                      widths=(.5, .5), vert=False, boxprops=boxprops,
                      whiskerprops=whiskerprops, capprops=capprops)

# Remove frame
for axis in ['top', 'bottom', 'left', 'right']:
    ax1.spines[axis].set_linewidth(0)
    ax2.spines[axis].set_linewidth(3)

# Set length and width of tick parameters
ax2.xaxis.set_tick_params(width=3, length=10)
ax2.yaxis.set_tick_params(width=3, length=10)

# Remove ticks and labels
ax1.xaxis.set_tick_params(width=0)
ax1.yaxis.set_tick_params(width=0)
ax1.set_xticklabels([])
ax1.set_yticklabels([])

# Fill with colors
colors = ['lightgray', 'lightgray']
for i, (patch, color) in enumerate(zip(bplots['boxes'], colors)):
    patch.set_facecolor(color)

# Median line
for median in bplots['medians']:
    median.set(color='k', linewidth=6.)

# Which ticks shall have the labels
plt.yticks(range(1, 3),
           ['Intra-subject\n correlation', 'Inter-subject\n correlation'],
           fontsize=32)
# Limits of the y-axis
# ax2.set_xlim([.45, 2.55])

# Fontsize of x-axis labels
plt.xticks(fontsize=32)

# Set the title...
plt.title('Correlations of the dictionary components on split-half data',
          fontweight='bold', fontsize=34, y=1.05)
# ...and save pdf
plt.savefig(os.path.join(write_dir, 'dictionary_consistency.pdf'))

# Show results
print(n_subjects, inter.mean(), intra.mean())
plt.show(block=False)
