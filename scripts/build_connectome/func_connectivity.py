"""
This script creates correlation and partial correlation matrices (representing 
functional connectivity) from resting-state fMRI for ROIs from a given atlas
See: https://nilearn.github.io/stable/connectivity/functional_connectomes.html
and https://nilearn.github.io/stable/connectivity/connectome_extraction.html

The correlations here are between the time-series of resting-state BOLD signal
for two given ROIs.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from nilearn import datasets
from nilearn.maskers import NiftiLabelsMasker
from nilearn import plotting
from sklearn.covariance import GraphicalLassoCV

DATA_ROOT = '/neurospin/ibc/derivatives/'
sub_ses = {'sub-01': 'ses-12', 'sub-04': 'ses-08', 'sub-05': 'ses-08',
           'sub-06': 'ses-09', 'sub-07': 'ses-09', 'sub-08': 'ses-09',
           'sub-09': 'ses-09', 'sub-11': 'ses-09', 'sub-12': 'ses-09',
           'sub-13': 'ses-09', 'sub-14': 'ses-05', 'sub-15': 'ses-12'}

for sub, ses in sub_ses.items():
    # setup tmp dir for saving figures
    tmp_dir = os.path.join(DATA_ROOT, sub, ses, 'func', 'connectome_tmp')
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    # get atlas
    atlas = datasets.fetch_atlas_schaefer_2018(data_dir=tmp_dir,
                                               resolution_mm=1, n_rois=400)
    # give atlas a custom name
    atlas['name'] = 'schaefer400'
    # define regions using the atlas
    masker = NiftiLabelsMasker(labels_img=atlas.maps, standardize=True)
    rs_fmri = os.path.join(DATA_ROOT, sub, ses, 'func',
                           (f'wrdc{sub}_{ses}_task-RestingState_dir'
                            f'-ap_bold.nii.gz'))
    # path to confounds file
    confounds = os.path.join(DATA_ROOT, sub, ses, 'func',
                             (f'{sub}_{ses}_task-RestingState_dir-ap_desc'
                              f'-confounds_timeseries.tsv'))
    # extract time series for those regions
    time_series = masker.fit_transform(rs_fmri, confounds=confounds)
    # define a sparse iinverse covariance estimator
    estimator = GraphicalLassoCV(cv=10)
    estimator.fit(time_series)
    # save correlation and partial correlation matrices as csv
    corr = os.path.join(tmp_dir, f'{atlas.name}_corr_{sub}_{ses}.csv')
    part_corr = os.path.join(tmp_dir, f'{atlas.name}_part_corr_{sub}_{ses}.csv')
    np.savetxt(corr, estimator.covariance_, delimiter=',')
    np.savetxt(part_corr, -estimator.precision_, delimiter=',')
    # plot heatmaps and save figs                     
    fig = plt.figure(figsize = (50,50))
    plotting.plot_matrix(estimator.covariance_, labels=atlas.labels,
                         figure=fig, vmax=1, vmin=-1,
                         title='Covariance')
    corr_fig = os.path.join(tmp_dir, f'{atlas.name}_corr_{sub}_{ses}.png')
    fig.savefig(corr_fig, bbox_inches='tight')
    fig = plt.figure(figsize = (50,50))
    plotting.plot_matrix(-estimator.precision_, labels=atlas.labels, figure=fig,
                         vmax=1, vmin=-1, title='Sparse inverse covariance')
    part_corr_fig = os.path.join(tmp_dir, 
                                 f'{atlas.name}_part_corr_{sub}_{ses}.png')
    fig.savefig(part_corr_fig, bbox_inches='tight')

