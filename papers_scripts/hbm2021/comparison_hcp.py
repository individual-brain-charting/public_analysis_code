"""
Get average and std maps for key archi and HCP contrasts
and compare these values to those of IBC contrasts

Authors: Bertrand Thirion, Ana Luisa Pinho

Last update: June 2020

Compatibility: Python 3.5

"""
import os
import glob
import pandas as pd
from nilearn.input_data import NiftiMasker
from nilearn.image import mean_img, resample_to_img
from joblib import Memory
from ibc_public.utils_data import (
    data_parser, SMOOTH_DERIVATIVES, SUBJECTS, LABELS)
from ibc_public.utils_data import all_contrasts as CONTRASTS
import ibc_public
import numpy as np

import matplotlib
matplotlib.use('Agg') # to work in Drago
from matplotlib import pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm


class MidpointNormalize(colors.Normalize):
    """
    Class to make my own norm for the color map of the correlation matrix.
    It gives two different linear ramps.
    Original author: Joe Kington 
    """
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


##############################################################################
# Get HCP average maps

main_dir = '/storage/store/data/HCP900/glm/'
mask = 'gm_mask_2mm.nii.gz'
masker = NiftiMasker(mask_img=mask).fit()
ibc_mask = '../../ibc_public/ibc_data/gm_mask_1_5mm.nii.gz'


def compute_mean_hcp(hcp_task, hcp_contrast, ibc_contrast, ibc_mask):
    hcp_name = 'z_%s.nii.gz' % hcp_contrast
    ibc_name = 'z_%s.nii.gz' % ibc_contrast
    wc = os.path.join(main_dir, '*', hcp_task, 'level2', 'z_maps', hcp_name)
    imgs = glob.glob(wc)
    X = masker.transform(imgs)
    x = X.mean(0)
    s = X.std(0)

    mean_img_ = resample_to_img(masker.inverse_transform(x), ibc_mask)
    mean_img_.to_filename(os.path.join('/storage/workspace/bthirion/',
                                       'mean_' + ibc_name))
    std_img_ = resample_to_img(masker.inverse_transform(s), ibc_mask)
    std_img_.to_filename(os.path.join('/storage/workspace/bthirion/',
                                      'std_' + ibc_name))


df_hcp = pd.read_csv('hcp_contrasts.csv')
"""
for (hcp_task, hcp_contrast, ibc_contrast) in zip(
        df_hcp['HCP task'],  df_hcp['HCP name'], df_hcp['IBC name']):
    compute_mean_hcp(hcp_task, hcp_contrast, ibc_contrast, ibc_mask)
"""
##############################################################################
# Get archi average maps
# obtain a grey matter mask
"""
_package_directory = os.path.dirname(
    os.path.abspath(ibc_public.utils_data.__file__))
mask = os.path.join(
    _package_directory, '../ibc_data', 'gm_mask_3mm.nii.gz')

main_dir = '/neurospin/tmp/archi/glm/unsmoothed'
masker = NiftiMasker(mask_img=mask).fit()
ibc_mask = os.path.join(
    _package_directory, '../ibc_data', 'gm_mask_1_5mm.nii.gz')


def compute_mean_archi(archi_contrast, ibc_contrast, ibc_mask):
    archi_name = '%s_z_map.nii.gz' % archi_contrast
    ibc_name = 'z_%s.nii.gz' % ibc_contrast
    wc = os.path.join(main_dir, '*', archi_name)
    imgs = glob.glob(wc)
    X = masker.transform(imgs)
    x = X.mean(0)
    s = X.std(0)

    mean_img_ = resample_to_img(masker.inverse_transform(x), ibc_mask)
    mean_img_.to_filename(os.path.join('/neurospin/tmp/bthirion/',
                                       'mean_' + ibc_name))
    std_img_ = resample_to_img(masker.inverse_transform(s), ibc_mask)
    std_img_.to_filename(os.path.join('/neurospin/tmp/bthirion/',
                                      'std_' + ibc_name))


df = pd.read_csv('archi_contrasts.csv')
for (archi_contrast, ibc_contrast) in zip(
     df['archi name'], df['IBC name']):
    compute_mean_archi(archi_contrast, ibc_contrast, ibc_mask)
"""
##############################################################################
# Compare IBC against HCP

main_parent_dir = '/neurospin/tmp/'
alt_parent_dir = '/storage/workspace/'
main_dir = 'bthirion'

if os.path.exists(main_parent_dir):
    cache = main_parent_dir + main_dir
else:
    cache = alt_parent_dir + main_dir

# output directory
mem = Memory(cachedir=cache, verbose=0)
# write_dir = cache
write_dir = '/storage/store/work/agrilopi/reproducibility_mtx'
if not os.path.exists(write_dir):
    os.mkdir(write_dir)

# Access to the data
subject_list = SUBJECTS

task_list = ['ArchiEmotional', 'ArchiSocial', 'ArchiSpatial',
             'ArchiStandard', 'HcpEmotion', 'HcpGambling',
             'HcpLanguage', 'HcpMotor', 'HcpRelational',
             'HcpSocial', 'HcpWm',
             'RSVPLanguage']
df = data_parser(derivatives=SMOOTH_DERIVATIVES, subject_list=SUBJECTS,
                 conditions=CONTRASTS, task_list=task_list)

# Mask of the grey matter across subjects
_package_directory = os.path.dirname(
    os.path.abspath(ibc_public.utils_data.__file__))
mask_gm = os.path.join(
    _package_directory, '../ibc_data', 'gm_mask_1_5mm.nii.gz')

data_dir = SMOOTH_DERIVATIVES
masker = NiftiMasker(mask_img=mask_gm).fit()
X = []
Y = []
names = []
for j, task in enumerate(task_list):
    task_df = df[df.task == task][df.acquisition == 'ffx']
    contrasts = sorted(task_df.contrast.unique())
    if task == 'ArchiStandard':
        contrasts = ['horizontal-vertical', 'computation-sentences',
                     'reading-listening', 'reading-checkerboard',
                     'left-right_button_press']
    elif task == 'ArchiSpatial':
        contrasts = ['saccades', 'hand-side', 'grasp-orientation']
    elif task == 'ArchiSocial':
        contrasts = ['triangle_mental-random',
                     'false_belief-mechanistic_video',
                     'false_belief-mechanistic_audio']
    elif task == 'ArchiEmotional':
        contrasts = ['face_trusty-gender', 'expression_intention-gender']
    elif task in ['hcp_emotion', 'hcp_gambling', 'hcp_language',
                  'hcp_relational', 'hcp_social', 'hcp_motor', 'hcp_wm']:
        contrasts = [c for c in contrasts if c in df_hcp['IBC name'].unique()]
    
    image_dir = cache
    if task in ['ArchiStandard', 'ArchiSpatial', 'ArchiSocial',
             'ArchiEmotional']:
        image_dir = os.path.join(cache, 'archi')
    for contrast in contrasts:
        hcp_img = os.path.join(image_dir, 'mean_z_%s.nii.gz' % contrast)
        imgs = task_df[task_df.contrast == contrast].path.values
        x = masker.transform(imgs).mean(0)
        X.append(x)
        y = masker.transform(hcp_img)[0]
        Y.append(y)
        names.append(contrast)

pretty_names = [CONTRASTS[CONTRASTS.contrast == name]['positive label'].values[0]
                + ' vs. ' +
                CONTRASTS[CONTRASTS.contrast == name]['negative label'].values[0]
                for name in names]
n_contrasts = len(X)
C = np.corrcoef(np.array(X), np.array(Y))
plt.figure(figsize=(16, 13.2))
ax = plt.subplot(111)
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(2)
plt.imshow(C[:n_contrasts, n_contrasts:], interpolation='nearest',
           vmin=-1., vmax=1.,
           norm=MidpointNormalize(midpoint=0., vmin=-1., vmax=1.),
           cmap=cm.bwr)
ax.xaxis.set_tick_params(width=2)
ax.yaxis.set_tick_params(width=2)
plt.xticks(np.arange(n_contrasts) + .5, pretty_names, rotation=45, ha='right',
           fontsize=14)
plt.yticks(np.arange(n_contrasts), pretty_names, fontsize=14)
cbar = plt.colorbar(shrink=.93)
cbar.ax.tick_params(labelsize=24)
plt.text(.115, 1.01, 'HCP contrasts', transform=ax.transAxes,
         fontweight='bold', fontsize=24)
plt.text(.59, 1.01, 'ARCHI contrasts', transform=ax.transAxes,
         fontweight='bold', fontsize=24)
plt.text(1.01, .58, 'IBC contrasts', rotation=90, transform=ax.transAxes,
         fontweight='bold', fontsize=24)
plt.plot([14.5, 14.5], [-.5, 27.5], linewidth=4, color=[.2, .2, .2])
plt.plot([-.5, 27.5], [14.5, 14.5], linewidth=4, color=[.2, .2, .2])
plt.subplots_adjust(left=.32, bottom=.26, right=.99, top=.96)
plt.show(block=False)
plt.savefig(os.path.join(write_dir, 'comparison_archi_hcp.pdf'))
