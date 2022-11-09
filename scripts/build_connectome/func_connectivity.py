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
from ibc_public.utils_data import get_subject_session


 
DATA_ROOT = '/neurospin/ibc/derivatives/'
mem = '/neurospin/tmp/bthirion/'
#sub_ses = {'sub-01': ['ses-14', 'ses-15'], 'sub-04': ['ses-11', 'ses-12'],
#           'sub-05': ['ses-11', 'ses-12'], 'sub-06': ['ses-11', 'ses-12'],
#           'sub-07': ['ses-11', 'ses-12'], 'sub-08': ['ses-12', 'ses-13'],
#           'sub-09': ['ses-12', 'ses-13'], 'sub-11': ['ses-11', 'ses-12'],
#           'sub-12': ['ses-11', 'ses-12'], 'sub-13': ['ses-11', 'ses-12'],
#           'sub-14': ['ses-11', 'ses-12'], 'sub-15': ['ses-14', 'ses-15']}

subject_sessions = sorted(get_subject_session(['mtt1', 'mtt2']))
sub_ses = dict([(subject_sessions[2 * i][0],
                 [subject_sessions[2 * i][1], subject_sessions[2 * i + 1][1]])
                for i in range(len(subject_sessions) // 2)])

# get atlas
atlas = datasets.fetch_atlas_schaefer_2018(data_dir=mem,
                                           resolution_mm=2,
                                           n_rois=400)
# give atlas a custom name
atlas['name'] = 'schaefer400'
# define regions using the atlas
masker = NiftiLabelsMasker(
    labels_img=atlas.maps,
    standardize=True,
    low_pass=.1,
    high_pass=.01,
    t_r=.76,
    verbose=1,
    memory=mem,
).fit()


for sub, sess in sub_ses.items():
    for ses in sess:
        for direction in ['ap', 'pa']:
            # setup tmp dir for saving figures
            tmp_dir = os.path.join(DATA_ROOT, sub, ses, 'func',
                                   'connectivity_tmp')
            if not os.path.exists(tmp_dir):
                os.makedirs(tmp_dir)
            
            rs_fmri = os.path.join(DATA_ROOT, sub, ses, 'func',
                                   (f'wrdc{sub}_{ses}_task-RestingState_dir'
                                    f'-{direction}_bold.nii.gz'))
            # path to confounds file
            confounds = os.path.join(DATA_ROOT, sub, ses, 'func',
                                    (f'rp_dc{sub}_{ses}_task-RestingState_'
                                     f'dir-{direction}_bold.txt'))
            # todo add high-variance confounds

            # extract time series for those regions
            time_series = masker.transform(rs_fmri, confounds=confounds)

            # define a sparse inverse covariance estimator
            estimator = GraphicalLassoCV()
            estimator.fit(time_series)
            
            # save correlation and partial correlation matrices as csv
            # todo: also use pearson correlation
            # todo: use sparse group inverse across 4 runs for beter estimates
            corr = os.path.join(tmp_dir, (f'{atlas.name}_corr_{sub}_{ses}_'
                                          f'dir-{direction}.csv'))
            part_corr = os.path.join(tmp_dir, (f'{atlas.name}_part_corr_{sub}_'
                                               f'{ses}_dir-{direction}.csv'))
            np.savetxt(corr, estimator.covariance_, delimiter=',')
            np.savetxt(part_corr, -estimator.precision_, delimiter=',')

            # plot heatmaps and save figs                     
            fig = plt.figure(figsize = (50,50))
            plotting.plot_matrix(estimator.covariance_, labels=atlas.labels,
                                 figure=fig, vmax=1, vmin=-1,
                                 title='Covariance')
            corr_fig = os.path.join(tmp_dir, (f'{atlas.name}_corr_{sub}_{ses}_'
                                              f'dir-{direction}.png'))
            fig.savefig(corr_fig, bbox_inches='tight')
            fig = plt.figure(figsize = (50, 50))
            plotting.plot_matrix(-estimator.precision_, labels=atlas.labels,
                                 figure=fig, vmax=1, vmin=-1,
                                 title='Sparse inverse covariance')
            part_corr_fig = os.path.join(tmp_dir, 
                                         (f'{atlas.name}_part_corr_{sub}_{ses}_'
                                          f'dir-{direction}.png'))
            fig.savefig(part_corr_fig, bbox_inches='tight')

            plt.close('all')
            