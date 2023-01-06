from nilearn import plotting, datasets
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

struc_connectivity_matrices = ['mni_sift', 'native_sift', 'mni_no_sift',
                               'native_no_sift']

DATA_ROOT = '/neurospin/ibc/derivatives/'
# dwi session numbers for each subject
sub_ses_dwi = {'sub-01': 'ses-12', 'sub-04': 'ses-08', 'sub-05': 'ses-08',
               'sub-06': 'ses-09', 'sub-07': 'ses-09', 'sub-08': 'ses-09',
               'sub-09': 'ses-09', 'sub-11': 'ses-09', 'sub-12': 'ses-09',
               'sub-13': 'ses-09', 'sub-14': 'ses-05', 'sub-15': 'ses-12'}

tmp_dir = os.path.join(DATA_ROOT, 'sub-04', 'ses-08', 'dwi', 'tract2mni_tmp')

atlas = datasets.fetch_atlas_schaefer_2018(data_dir=tmp_dir,
                                           resolution_mm=1,
                                           n_rois=400)

for sub_dwi, ses_dwi in sub_ses_dwi.items():
    plot_count = 1
    fig = plt.figure(figsize=(40, 10))
    for struc_conn in struc_connectivity_matrices:
        space_sift = struc_conn.split('_')
        if space_sift[0] == 'native':
            if space_sift[1] == 'sift':
                filename = f'sift-weighted_{sub_dwi}_{ses_dwi}_t2.csv'
            elif space_sift[1] == 'no':
                filename = f'{sub_dwi}_{ses_dwi}_t2.csv'
            struc_mat = os.path.join(DATA_ROOT, sub_dwi, ses_dwi, 'dwi',
                                    'atlas2dwi', f'schaefer400_connectome_{filename}')
        elif space_sift[0] == 'mni':
            if sub_dwi == 'sub-06':
                continue
            if space_sift[1] == 'sift':
                filename = f'sift-weighted_{sub_dwi}_{ses_dwi}_t2.csv'
            elif space_sift[1] == 'no':
                filename = f'{sub_dwi}_{ses_dwi}_t2.csv'
            struc_mat = os.path.join(DATA_ROOT, sub_dwi, ses_dwi, 'dwi',
                                    f'schaefer400_connectome_{filename}')
        struc = pd.read_csv(struc_mat, names=atlas.labels)
        ax = plt.subplot(1, 4, plot_count)
        # sns.heatmap(struc, ax=ax)
        # ax.set_title(f'{sub_dwi}, {struc_conn}')
        plotting.plot_matrix(np.log(struc), title=f'{sub_dwi}, {struc_conn}', axes=ax, tri='lower')
        plot_count = plot_count + 1
    plt.subplots_adjust(wspace=0.4, hspace=0)
    plt.savefig(f'{sub_dwi}_conn_mats.png', bbox_inches='tight')
    plt.close()

for sub_dwi, ses_dwi in sub_ses_dwi.items():
    plot_count = 1
    fig = plt.figure(figsize=(40, 5))
    for struc_conn in struc_connectivity_matrices:
        space_sift = struc_conn.split('_')
        if space_sift[0] == 'native':
            if space_sift[1] == 'sift':
                filename = f'sift-weighted_{sub_dwi}_{ses_dwi}_t2.csv'
            elif space_sift[1] == 'no':
                filename = f'{sub_dwi}_{ses_dwi}_t2.csv'
            struc_mat = os.path.join(DATA_ROOT, sub_dwi, ses_dwi, 'dwi',
                                    'atlas2dwi', f'schaefer400_connectome_{filename}')
        elif space_sift[0] == 'mni':
            if sub_dwi == 'sub-06':
                continue
            if space_sift[1] == 'sift':
                filename = f'sift-weighted_{sub_dwi}_{ses_dwi}_t2.csv'
            elif space_sift[1] == 'no':
                filename = f'{sub_dwi}_{ses_dwi}_t2.csv'
            struc_mat = os.path.join(DATA_ROOT, sub_dwi, ses_dwi, 'dwi',
                                    f'schaefer400_connectome_{filename}')
        coords = pd.read_csv(
            '/neurospin/ibc/derivatives/sub-04/ses-08/dwi/tract2mni_tmp/schaefer_2018/ras_coords_400.csv')
        coords = coords[['R', 'A', 'S']]
        coords = coords.to_numpy()
        struc = pd.read_csv(struc_mat, names=atlas.labels)
        ax = plt.subplot(1, 4, plot_count)
        plotting.plot_connectome(
            struc, coords, edge_threshold="99.5%", node_size=5, axes=ax, title=f'{sub_dwi}, {struc_conn}')
        plot_count = plot_count + 1
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(f'{sub_dwi}_conn_net.png', bbox_inches='tight')
    plt.close()
