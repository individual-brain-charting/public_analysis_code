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
from sklearn.covariance import GraphicalLassoCV
from nilearn import datasets
from nilearn.maskers import NiftiLabelsMasker
from nilearn import plotting
from nilearn.connectome import ConnectivityMeasure
from nilearn.connectome import GroupSparseCovarianceCV
from ibc_public.utils_data import get_subject_session
from nilearn.image import high_variance_confounds

if 0:
    DATA_ROOT = '/neurospin/ibc/derivatives/'
    mem = '/neurospin/tmp/bthirion/'
else:
    DATA_ROOT = '/storage/store2/data/ibc/derivatives/'
    mem = '/storage/store/work/bthirion/'
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
    low_pass=.2,
    high_pass=.01,
    t_r=.76,
    verbose=1,
    memory=mem,
).fit()

subject_sessions = sorted(get_subject_session(['mtt1', 'mtt2']))
sub_ses = dict([(subject_sessions[2 * i][0],
                 [subject_sessions[2 * i][1], subject_sessions[2 * i + 1][1]])
                for i in range(len(subject_sessions) // 2)])

correlation_measure = ConnectivityMeasure(kind='correlation')
glc = GraphicalLassoCV()
gsc = GroupSparseCovarianceCV(verbose=2)

for sub, sess in sub_ses.items():
    all_time_series = []
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
            # todo: add high-variance confounds
            compcor = high_variance_confounds(rs_fmri)
            confounds = np.hstack((np.loadtxt(confounds), compcor))
            
            # extract time series for those regions
            time_series = masker.transform(rs_fmri, confounds=confounds)
            all_time_series.append(time_series)

            correlation_matrix = correlation_measure.fit_transform([time_series])[0]
            corr = os.path.join(tmp_dir, (f'{atlas.name}_corr_{sub}_{ses}_'
                                  f'dir-{direction}.csv'))
            np.savetxt(corr, correlation_matrix, delimiter=',')
            fig = plt.figure(figsize=(10, 10))
            plotting.plot_matrix(correlation_matrix, labels=atlas.labels,
                                 figure=fig, vmax=1, vmin=-1,
                                 title='Covariance')
            corr_fig = os.path.join(tmp_dir, (f'{atlas.name}_corr_{sub}_{ses}_'
                                              f'dir-{direction}.png'))
            fig.savefig(corr_fig, bbox_inches='tight')
            
            # define a sparse inverse covariance estimator
            glc.fit(time_series)
    
            # save correlation and partial correlation matrices as csv
            part_corr = os.path.join(tmp_dir, (f'{atlas.name}_part_corr_{sub}_'
                                               f'{ses}_dir-{direction}.csv'))
            np.savetxt(part_corr, -glc.precision_, delimiter=',')
    
            # plot heatmaps and save figs                     
            fig = plt.figure(figsize = (10, 10))
            plotting.plot_matrix(-glc.precision_, labels=atlas.labels,
                                 figure=fig, vmax=1, vmin=-1,
                                 title='Sparse inverse covariance')
            part_corr_fig = os.path.join(tmp_dir, 
                                         (f'{atlas.name}_part_corr_{sub}_{ses}_'
                                          f'dir-{direction}.png'))
            fig.savefig(part_corr_fig, bbox_inches='tight')

    # Use sparse group inverse across 4 runs for beter estimates
    # gsc.fit(all_time_series)
    
    glc.fit(np.concatenate(all_time_series))
    # save correlation and partial correlation matrices as csv
    part_corr = os.path.join(tmp_dir, f'{atlas.name}_part_corr_{sub}_all_compcorr.csv')
    np.savetxt(part_corr, -glc.precision_, delimiter=',')
    corr = os.path.join(tmp_dir, f'{atlas.name}_corr_{sub}_all.csv')
    np.savetxt(corr, glc.covariance_, delimiter=',')
    
    # plot heatmaps and save figs                     
    fig = plt.figure(figsize = (10, 10))
    plotting.plot_matrix(-glc.precision_, labels=atlas.labels,
                         figure=fig, vmax=1, vmin=-1,
                         title='Sparse inverse covariance')
    part_corr_fig = os.path.join(
        tmp_dir, f'{atlas.name}_part_corr_{sub}_all.png')
    fig.savefig(part_corr_fig, bbox_inches='tight')
    #
    fig = plt.figure(figsize = (10, 10))
    plotting.plot_matrix(glc.covariance_, labels=atlas.labels,
                         figure=fig, vmax=1, vmin=-1,
                         title='Covariance with sparse inverse')
    corr_fig = os.path.join(
        tmp_dir, f'{atlas.name}_corr_{sub}_all.png')
    fig.savefig(corr_fig, bbox_inches='tight')
    plt.close('all')
