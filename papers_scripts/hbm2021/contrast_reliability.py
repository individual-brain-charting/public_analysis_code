"""
Script to obtain and plot a basic measure of reliability across contrasts

Authors: Bertrand Thirion, Ana Luisa Pinho

Last update: June 2020

Compatibility: Python 3.5

"""
import os
import json
import numpy as np
from joblib import Memory
import matplotlib
matplotlib.use('Agg')
import pandas as pd
from matplotlib import pyplot as plt
from nilearn.input_data import NiftiMasker
from ibc_public.utils_data import (
    data_parser, SMOOTH_DERIVATIVES, SUBJECTS, CONTRASTS, LABELS)
import ibc_public


def flatten(li):
    return sum(([x] if not isinstance(x, list) else flatten(x)
                for x in li), [])

# caching
main_parent_dir = '/neurospin/tmp/'
alt_parent_dir = '/storage/workspace/'
main_dir = 'bthirion'

if os.path.exists(main_parent_dir):
    cache = main_parent_dir + main_dir
else:
    cache = alt_parent_dir + main_dir

# output directory
write_dir = os.path.join(main_parent_dir, 'agrilopi')
mem = Memory(cachedir=write_dir, verbose=0)

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
masker = NiftiMasker(mask_img=mask_gm, memory=write_dir).fit()


def append_correlation(imgs, masker, correlations=[]):
    X = masker.transform(imgs)
    corr_matrix = np.triu(np.corrcoef(X), 1)
    correlations_ = corr_matrix[corr_matrix != 0]
    correlations.append(correlations_)
    return correlations


fig = plt.figure(figsize=(17, 11))
q = 0
column = 0
# Adjust colormap
color_array = np.linspace(0, 255, len(task_list)).astype(np.int)
color_array[5] += 22
color_array[6] += 10
color_array[9] += 7
color_array[10] += -2
colors = plt.cm.gist_rainbow(color_array)
# Reorder task_list
torder = [0, 1, 2, 3, 8, 6, 7, 4, 9, 5, 10, 11]
task_list = [task_list[t] for t in torder]
for j, task in enumerate(task_list):
    task_df = df[df.task == task][df.acquisition == 'ffx']
    contrasts = task_df.contrast.unique()
    correlations = []
    # Reorder contrasts and
    # remove those that are not part of the main text
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
        imgs = task_df[task_df.contrast == contrast].path.values
        X = masker.transform(imgs)
        correlations = append_correlation(imgs, masker, correlations)

    correlations = np.array(correlations)
    # Define subplot box for bar charts and its pos in the fig
    ax = plt.axes([.3 + .48 * column, q * .048 + .115, .21,
                   n_contrasts * .049])
    ax.barh(np.arange(n_contrasts), correlations.mean(1), .75,
            xerr=correlations.std(1), error_kw=dict(capsize=2, captick=3),
            color=colors[j])
    ax.set_yticks(np.arange(n_contrasts))
    new_labels = []
    for i in range(n_contrasts):
        new_labels.append(LABELS[contrasts[i]][1].tolist()[0] + ' vs. ' +
                          LABELS[contrasts[i]][0].tolist()[0])
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
    ax.set_xlim(0, 0.45)
    ax.tick_params(labelsize=14)
    q += n_contrasts
    if j == 5:
        q = 0
        column = 1

# Define box for text and its pos in the fig
ax = plt.axes([0, 0, .99, .035])
ax = plt.text(.355, 1.,
              'Average and standard deviation of ' +
              'map correlation across subjects',
              weight='bold', fontsize='18')
plt.axis('off')
plt.savefig(os.path.join(write_dir, 'consistency.pdf'), dpi=300)

#############################################################################
# Comparison with archi
df_archi = pd.read_csv('archi_contrasts.csv')
archi_dir = '/neurospin/tmp/archi/glm/unsmoothed'
import glob


def compute_mean_corr(data_dir, archi_contrast):
    archi_name = '%s_z_map.nii.gz' % archi_contrast
    wc = os.path.join(data_dir, '*', archi_name)
    imgs = glob.glob(wc)
    return append_correlation(imgs, masker, [])


def compute_mean_corr_ibc(df, contrast):
    imgs = []
    # pick only 1 image per subject
    for subject in df.subject.unique():
        img = df[df.contrast == contrast]\
                [df.acquisition == 'ffx']\
                [df.subject == subject].path.values[-1]
        imgs.append(img)
    return append_correlation(imgs, masker, [])


correlations_archi = []
correlations_ibc = []
plot_labels = []
for (archi_contrast, ibc_contrast) in zip(
     df_archi['archi name'], df_archi['IBC name']):
    correlations_archi.append(compute_mean_corr(archi_dir, archi_contrast)[0])
    correlations_ibc.append(compute_mean_corr_ibc(df, ibc_contrast)[0])
    plot_labels.append(LABELS[ibc_contrast][1].tolist()[0] + ' vs. ' +
                       LABELS[ibc_contrast][0].tolist()[0])

correlations_archi = np.array(correlations_archi)
n_archi = len(
    glob.glob(os.path.join(archi_dir, '*', df_archi['archi name'][0] +
                           '_z_map.nii.gz')))
dof_archi = .5 * n_archi * (n_archi - 1)
mean_archi = correlations_archi.mean(1)
std_archi = 2 * correlations_archi.std(1) / np.sqrt(dof_archi)
n_ibc = len(df.subject.unique())
dof_ibc = .5 * n_ibc * (n_ibc - 1)
mean_ibc = np.array([x.mean() for x in correlations_ibc])
std_ibc = 2 * np.array([x.std() for x in correlations_ibc]) / np.sqrt(dof_ibc)

n_contrasts = len(correlations_archi)
plt.figure()
ax = plt.axes()
ax.barh(np.arange(n_contrasts), mean_archi, .4,
        xerr=std_archi, error_kw=dict(capsize=2, captick=3),
        label='archi'
        )
ax.barh(np.arange(n_contrasts) + .4, mean_ibc, .4,
        xerr=std_ibc, error_kw=dict(capsize=2, captick=3),
        label='ibc')
ax.legend()
ax.set_yticks(np.arange(n_contrasts))
ax.set_yticklabels(plot_labels)
ax.set_title('Inter-subject brain maps\n correlation', fontweight='bold')
plt.subplots_adjust(left=.6, right=.99)
plt.savefig(os.path.join(write_dir, 'reliability_archi.pdf'))
plt.show(block=False)
