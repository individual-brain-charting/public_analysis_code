"""
this script aims at comparings the intra- and inter- subject stability
of contrasts across IBC and HCP

Authors: Bertrand Thirion, Ana Luisa Pinho

Last update: June 2020

Compatibility: Python 3.5

"""
import pandas as pd
import glob
import os
import numpy as np
from nilearn.input_data import NiftiMasker
from ibc_public.utils_data import (
    data_parser, SMOOTH_DERIVATIVES, SUBJECTS, CONTRASTS, LABELS)
import ibc_public
import matplotlib.pyplot as plt

# caching
main_parent_dir = '/neurospin/tmp/'
alt_parent_dir = '/storage/workspace/'

if os.path.exists(main_parent_dir):
    cache = main_parent_dir
else:
    cache = alt_parent_dir
cache += 'bthirion'

# output directory
write_dir = '/storage/store/work/agrilopi'

subject_list = SUBJECTS

task_list = ['hcp_language', 'hcp_social', 'hcp_gambling',
             'hcp_motor', 'hcp_emotion', 'hcp_relational', 'hcp_wm']

ibc_df = data_parser(derivatives=SMOOTH_DERIVATIVES, subject_list=SUBJECTS,
                     conditions=CONTRASTS, task_list=task_list)


# Mask of the grey matter across subjects
_package_directory = os.path.dirname(
    os.path.abspath(ibc_public.utils_data.__file__))
mask_gm = os.path.join(
    _package_directory, '../ibc_data', 'gm_mask_1_5mm.nii.gz')
masker = NiftiMasker(mask_img=mask_gm, memory=write_dir).fit()

# HCP data
hcp_df = pd.read_csv('hcp_contrasts.csv')
hcp_dir = '/storage/store/data/HCP900/glm/'
hcp_mask = 'gm_mask_2mm.nii.gz'
hcp_masker = NiftiMasker(mask_img=hcp_mask).fit()


def append_correlation(imgs, masker, correlations=[]):
    X = masker.transform(imgs)
    corr_matrix = np.triu(np.corrcoef(X), 1)
    correlations_ = corr_matrix[corr_matrix != 0]
    correlations.append(correlations_)
    return correlations


def inter_corr_hcp(data_dir, hcp_task, hcp_contrast):
    hcp_name = 'z_%s.nii.gz' % hcp_contrast
    main_dir = '/storage/store/data/HCP900/glm/'
    wc = os.path.join(main_dir, '*', hcp_task, 'level2', 'z_maps', hcp_name)
    imgs = glob.glob(wc)
    if len(imgs) == 0:
        stop
    return append_correlation(imgs, masker, [])


def inter_corr_ibc(ibc_df, contrast):
    imgs = []
    # pick only 1 image per subject
    for subject in ibc_df.subject.unique():
        img = ibc_df[ibc_df.contrast == contrast]\
                    [ibc_df.acquisition == 'ffx']\
                    [ibc_df.subject == subject].path.values[-1]
        imgs.append(img)
    return append_correlation(imgs, masker, [])


def intra_corr_ibc(ibc_df, contrast):
    correlation = []
    for subject in ibc_df.subject.unique():
        img1 = ibc_df[ibc_df.contrast == contrast]\
                    [ibc_df.acquisition == 'ap']\
                    [ibc_df.subject == subject].path.values[-1]
        img2 = ibc_df[ibc_df.contrast == contrast]\
                    [ibc_df.acquisition == 'pa']\
                    [ibc_df.subject == subject].path.values[-1]
        imgs = [img1, img2]
        correlation.append(append_correlation(imgs, masker, [])[0][0])
    return correlation


def intra_corr_hcp(data_dir, hcp_task, hcp_contrast):
    hcp_name = 'z_%s.nii.gz' % hcp_contrast
    main_dir = '/storage/store/data/HCP900/glm/'
    wc1 = os.path.join(main_dir, '*', hcp_task, 'RL', 'z_maps', hcp_name)
    imgs = glob.glob(wc1)
    correlations = []
    for img1 in imgs:
        subject = img1.split('/')[-5]
        img2 = os.path.join(main_dir, subject, hcp_task, 'LR', 'z_maps', hcp_name) 
        if not os.path.exists(img2):
            continue
        correlations.append(append_correlation([img1, img2], masker, [])[0][0])
    return correlations

from joblib import Parallel, delayed

# inter_hcp = []
# inter_ibc = []
# intra_hcp = []
# intra_ibc = []

inter_ibc = Parallel(n_jobs=15, verbose=True)(
    delayed(inter_corr_ibc)(ibc_df, ibc_contrast)
    for (hcp_task, hcp_contrast, ibc_contrast) in zip(
            hcp_df['HCP task'], hcp_df['HCP name'], hcp_df['IBC name']))

inter_hcp = Parallel(n_jobs=15, verbose=True)(
    delayed(inter_corr_hcp)(hcp_dir, hcp_task, hcp_contrast)
    for (hcp_task, hcp_contrast, ibc_contrast) in zip(
            hcp_df['HCP task'], hcp_df['HCP name'], hcp_df['IBC name']))

    
intra_hcp = Parallel(n_jobs=15, verbose=True)(
    delayed(intra_corr_hcp)(hcp_dir, hcp_task, hcp_contrast)
    for (hcp_task, hcp_contrast, ibc_contrast) in zip(
            hcp_df['HCP task'], hcp_df['HCP name'], hcp_df['IBC name']))

intra_ibc =  Parallel(n_jobs=15, verbose=True)(
    delayed(intra_corr_ibc)(ibc_df, ibc_contrast)
    for (hcp_task, hcp_contrast, ibc_contrast) in zip(
            hcp_df['HCP task'], hcp_df['HCP name'], hcp_df['IBC name']))
    
plot_labels = []    
for (hcp_task, hcp_contrast, ibc_contrast) in zip(
     hcp_df['HCP task'], hcp_df['HCP name'], hcp_df['IBC name']):
    #intra_hcp.append(intra_corr_hcp(hcp_dir, hcp_task, hcp_contrast))
    # intra_ibc.append(intra_corr_ibc(ibc_df, ibc_contrast))
    #inter_hcp.append(inter_corr_hcp(hcp_dir, hcp_task,
    #                                       hcp_contrast)[0])
    # inter_ibc.append(inter_corr_ibc(ibc_df, ibc_contrast)[0])
    plot_labels.append(LABELS[ibc_contrast][1] + ' vs. ' +
                       LABELS[ibc_contrast][0])

    
# inter_hcp = np.array(inter_hcp)
n_hcp = len(
    glob.glob(os.path.join(hcp_dir, '*', hcp_df['HCP task'][0], 'level2',
                           'z_maps', 'z_%s.nii.gz' % hcp_df['HCP name'][0])))
dof_hcp = .5 * n_hcp * (n_hcp - 1)
mean_hcp = np.array([np.mean(x) for x in inter_hcp])
std_hcp = 2 * np.array([np.std(x) for x in inter_hcp]) / np.sqrt(dof_hcp)
n_ibc = len(ibc_df.subject.unique())
dof_ibc = .5 * n_ibc * (n_ibc - 1)
mean_ibc = np.array([np.mean(x) for x in inter_ibc])
std_ibc = 2 * np.array([np.std(x) for x in inter_ibc]) / np.sqrt(dof_ibc)

dof_intra_hcp = n_hcp
mean_intra_hcp = np.array([np.mean(x) for x in intra_hcp])
std_intra_hcp = 2 * np.array([np.std(x) for x in intra_hcp]) / np.sqrt(dof_intra_hcp)
dof_intra_ibc = n_ibc
mean_intra_ibc = np.array([np.mean(x) for x in intra_ibc])
std_intra_ibc = 2 * np.array([np.std(x) for x in intra_ibc]) / np.sqrt(dof_intra_ibc)

n_contrasts = len(inter_hcp)

plt.figure(figsize=(7., 4.8))
ax = plt.axes([0.46, 0.05, .26, .85])
ax.barh(np.arange(n_contrasts), mean_hcp, .4,
        xerr=std_hcp, error_kw=dict(capsize=2, captick=3),
        label='hcp')
ax.barh(np.arange(n_contrasts) + .4, mean_ibc, .4,
        xerr=std_ibc, error_kw=dict(capsize=2, captick=3),
        label='ibc')
ax.legend(loc=(.55, .2))
ax.set_yticks([])
ax.set_yticklabels(plot_labels)
ax.set_title('Inter-subject \n correlation', fontweight='bold')
ax = plt.axes([0.73, 0.05, .26, .85])
ax.barh(np.arange(n_contrasts), mean_intra_hcp, .4,
        xerr=std_intra_hcp, error_kw=dict(capsize=2, captick=3),
        label='hcp')
ax.barh(np.arange(n_contrasts) + .4, mean_intra_ibc, .4,
        xerr=std_intra_ibc, error_kw=dict(capsize=2, captick=3),
        label='ibc')
ax.set_title('Intra-subject \n correlation', fontweight='bold')
ax.set_yticks([])
ax = plt.axes([0., 0.05, .45, .85])
ax.axis('off')
for i in range(n_contrasts):
    plt.text(1., (i + 1) * .06, plot_labels[i].tolist()[0], fontsize=10,
             ha='right')
plt.savefig(os.path.join(write_dir, 'reliability_hcp.pdf'))
plt.show(block=False)
