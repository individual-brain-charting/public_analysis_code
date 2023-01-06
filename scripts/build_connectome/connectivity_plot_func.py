from nilearn import plotting, datasets
import pandas as pd
import os
import matplotlib.pyplot as plt

DATA_ROOT = '/neurospin/ibc/derivatives/'
# fmri session numbers for each subject
sub_ses = {'sub-01': ['ses-14', 'ses-15'], 'sub-04': ['ses-11', 'ses-12'],
          'sub-05': ['ses-11', 'ses-12'], 'sub-06': ['ses-11', 'ses-12'],
          'sub-07': ['ses-11', 'ses-12'], 'sub-08': ['ses-12', 'ses-13'],
          'sub-09': ['ses-12', 'ses-13'], 'sub-11': ['ses-11', 'ses-12'],
          'sub-12': ['ses-11', 'ses-12'], 'sub-13': ['ses-11', 'ses-12'],
          'sub-14': ['ses-11', 'ses-12'], 'sub-15': ['ses-14', 'ses-15']}

tmp_dir = os.path.join(DATA_ROOT, 'sub-04', 'ses-08', 'dwi', 'tract2mni_tmp')

atlas = datasets.fetch_atlas_schaefer_2018(data_dir=tmp_dir, 
                                           resolution_mm=1,
                                           n_rois=400)
atlas['name'] = 'schaefer400'

for hemisphere in ['left', 'right', 'both']:
    plot_count = 1
    fig = plt.figure(figsize=(36,16))
    for sub, ses in sub_ses.items():
        ses = ses[1]
        # if sub == 'sub-06':
        #     continue
        coords = pd.read_csv('/neurospin/ibc/derivatives/sub-04/ses-08/dwi/tract2mni_tmp/schaefer_2018/ras_coords_400.csv')
        coords = coords[['R','A','S']]
        coords = coords.to_numpy()
        func_mat = os.path.join(DATA_ROOT, sub, ses, 'func',
                                'connectivity_tmp', (f'{atlas.name}_part_corr_{sub}_all_compcorr.csv'))
        func = pd.read_csv(func_mat, names=atlas.labels)
        func = func.to_numpy()
        if hemisphere == 'left':
            func = func[0:200, 0:200]
            coords = coords[0:200]
        elif hemisphere == 'right':
            func = func[200:, 200:]
            coords = coords[200:]
        ax = plt.subplot(4, 3, plot_count)
        plotting.plot_connectome(func, coords, edge_threshold="99.8%", node_size=5, axes=ax, title=sub)
        plot_count = plot_count + 1

    plt.subplots_adjust(wspace=0, hspace=0)
    fig.suptitle(f'top 0.2% func connectivity in {hemisphere} hemisphere')
    fig.savefig(f'func_conn_{hemisphere}_point2.png', bbox_inches='tight')
