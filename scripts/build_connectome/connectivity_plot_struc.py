from nilearn import plotting, datasets
import pandas as pd
import os
import matplotlib.pyplot as plt

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

for hemisphere in ['left', 'right', 'both']:
    plot_count = 1
    fig = plt.figure(figsize=(36, 16))
    for sub_dwi, ses_dwi in sub_ses_dwi.items():
        if sub_dwi == 'sub-06':
            continue
        coords = pd.read_csv(
            '/neurospin/ibc/derivatives/sub-04/ses-08/dwi/tract2mni_tmp/schaefer_2018/ras_coords_400.csv')
        coords = coords[['R', 'A', 'S']]
        coords = coords.to_numpy()
        struc_mat = os.path.join(DATA_ROOT, sub_dwi, ses_dwi,
                                 'dwi', (f'schaefer400_connectome_'
                                         f'sift-weighted_{sub_dwi}_{ses_dwi}'
                                         f'.csv'))
        struc = pd.read_csv(struc_mat, names=atlas.labels)
        struc = struc.to_numpy()
        if hemisphere == 'left':
            struc = struc[0:200, 0:200]
            coords = coords[0:200]
        elif hemisphere == 'right':
            struc = struc[200:, 200:]
            coords = coords[200:]
        ax = plt.subplot(4, 3, plot_count)
        plotting.plot_connectome(
            struc, coords, edge_threshold="99.5%", node_size=5, axes=ax, title=sub_dwi)
        plot_count = plot_count + 1

    plt.subplots_adjust(wspace=0, hspace=0)
    fig.suptitle(f'top 0.5% struc connectivity in {hemisphere} hemisphere')
    fig.savefig(f'struct_conn_{hemisphere}_point5.png', bbox_inches='tight')
    plt.close()
