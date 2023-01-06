from nilearn import plotting, datasets, connectome
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

func_connectivity_matrices = ['each', 'all_corr', 'all_part_corr']

DATA_ROOT = '/neurospin/ibc/derivatives/'
# resting session numbers for each subject
sub_ses_fmri = {'sub-01': ['ses-14','ses-15'], 'sub-04': ['ses-11','ses-12'],
                'sub-05': ['ses-11','ses-12'], 'sub-06': ['ses-11','ses-12'],
                'sub-07': ['ses-11','ses-12'], 'sub-08': ['ses-12','ses-13'],
                'sub-09': ['ses-12','ses-13'], 'sub-11': ['ses-11','ses-12'],
                'sub-12': ['ses-11','ses-12'], 'sub-13': ['ses-11','ses-12'],
                'sub-14': ['ses-11','ses-12'], 'sub-15': ['ses-14','ses-15']}

tmp_dir = os.path.join(DATA_ROOT, 'sub-04', 'ses-08', 'dwi', 'tract2mni_tmp')

atlas = datasets.fetch_atlas_schaefer_2018(data_dir=tmp_dir,
                                           resolution_mm=1,
                                           n_rois=400)

for sub_func, sess_funcs in sub_ses_fmri.items():
    plot_count = 1
    fig = plt.figure(figsize=(40, 10))
    for func_conn in func_connectivity_matrices:
        sess_sift = func_conn.split('_')
        if sess_sift[0] == 'each':
            all_func = []
            for sess_func in sess_funcs:
                for direction in ['dir-ap', 'dir-pa']:
                    func_mat = os.path.join(DATA_ROOT, sub_func, 
                                            sess_func, 'func',
                                            'connectivity_tmp', 
                                            f'schaefer400_corr_{sub_func}_{sess_func}_{direction}.csv')
                    func = pd.read_csv(func_mat, names=atlas.labels)
                    all_func.append(func)
            func = np.mean(all_func, axis=0)
        elif sess_sift[0] == 'all':
            if sess_sift[1] == 'corr':
                filename = f'schaefer400_corr_{sub_func}_all.csv'
            elif sess_sift[1] == 'part':
                filename = f'schaefer400_part_corr_{sub_func}_all.csv'
            func_mat = os.path.join(DATA_ROOT, sub_func, 
                                    sess_funcs[1],
                                    'func', 'connectivity_tmp', filename)
            func = pd.read_csv(func_mat, names=atlas.labels)
        ax = plt.subplot(1, 4, plot_count)
        plotting.plot_matrix(func, title=f'{sub_func}, {func_conn}',axes=ax, tri='lower')
        # sns.heatmap(func, ax=ax)
        # ax.set_title(f'{sub_func}, {func_conn}')
        plot_count = plot_count + 1
    plt.subplots_adjust(wspace=0.4, hspace=0)
    plt.savefig(f'new-connectomes/{sub_func}_conn_func_mats.png', bbox_inches='tight')
    plt.close()

for sub_func, sess_funcs in sub_ses_fmri.items():
    plot_count = 1
    fig = plt.figure(figsize=(40, 5))
    for func_conn in func_connectivity_matrices:
        sess_sift = func_conn.split('_')
        if sess_sift[0] == 'each':
            all_func = []
            for sess_func in sess_funcs:
                for direction in ['dir-ap', 'dir-pa']:
                    func_mat = os.path.join(DATA_ROOT, sub_func, 
                                            sess_func, 'func',
                                            'connectivity_tmp', 
                                            f'schaefer400_corr_{sub_func}_{sess_func}_{direction}.csv')
                    func = pd.read_csv(func_mat, names=atlas.labels)
                    all_func.append(func)
            func = np.mean(all_func, axis=0)
        elif sess_sift[0] == 'all':
            if sess_sift[1] == 'corr':
                filename = f'schaefer400_corr_{sub_func}_all.csv'
            elif sess_sift[1] == 'part':
                filename = f'schaefer400_part_corr_{sub_func}_all.csv'
            func_mat = os.path.join(DATA_ROOT, sub_func, 
                                    sess_funcs[1],
                                    'func', 'connectivity_tmp', filename)
            func = pd.read_csv(func_mat, names=atlas.labels)
        coords = pd.read_csv(
            '/neurospin/ibc/derivatives/sub-04/ses-08/dwi/tract2mni_tmp/schaefer_2018/ras_coords_400.csv')
        coords = coords[['R', 'A', 'S']]
        coords = coords.to_numpy()
        ax = plt.subplot(1, 4, plot_count)
        plotting.plot_connectome(
            func, coords, edge_threshold="99.8%", node_size=5, axes=ax, title=f'{sub_func}, {func_conn}')
        plot_count = plot_count + 1
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(f'new-connectomes/{sub_func}_conn_func_net.png', bbox_inches='tight')
    plt.close()
    