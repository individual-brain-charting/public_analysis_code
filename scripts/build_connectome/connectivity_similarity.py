"""
This script calculates and plots {mean, median} over 
{cosine similarities, correlations} between functional and structural 
connectivity vectors of each parcel in {left, right, both} hemispheres for 
all IBC subject pairs
"""

import numpy as np
import matplotlib.pyplot as plt
from nilearn import datasets
import pandas as pd
import os
from sklearn.metrics import pairwise
import seaborn as sns

aggregations = ['median', 'mean']
similarity_metrics = ['correlation', 'cosine similarity']
hemispheres = ['LH', 'RH', 'both hemispheres']

DATA_ROOT = '/neurospin/ibc/derivatives/'
# dwi session numbers for each subject
sub_ses_dwi = {'sub-01': 'ses-12', 'sub-04': 'ses-08', 'sub-05': 'ses-08',
               'sub-06': 'ses-09', 'sub-07': 'ses-09', 'sub-08': 'ses-09',
               'sub-09': 'ses-09', 'sub-11': 'ses-09', 'sub-12': 'ses-09',
               'sub-13': 'ses-09', 'sub-14': 'ses-05', 'sub-15': 'ses-12'}

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

for aggregation in aggregations:
    for similarity_metric in similarity_metrics:
        for hemisphere in hemispheres:
            fig = plt.figure(figsize=(10,10))
            plot_count = 1

            for rs_func_ses_num in [0,1]:
                for direction in ['ap', 'pa']:
                    all_sub_agg_similarities = []
                    for sub_dwi, ses_dwi in sub_ses_dwi.items():
                        struc_mat = os.path.join(DATA_ROOT, sub_dwi, ses_dwi,
                                                'dwi', (f'schaefer400_connectome_'
                                                        f'sift-weighted_{sub_dwi}_{ses_dwi}'
                                                        f'.csv'))
                        struc = pd.read_csv(struc_mat, names=atlas.labels)
                        if hemisphere == 'LH':
                            struc = struc.iloc[0:200, 0:200]
                        elif hemisphere == 'RH':
                            struc = struc.iloc[200:, 200:]
                        similarities = []
                        agg_similarities = []
                        for sub_func, ses_func in sub_ses_fmri.items():
                            func_mat = os.path.join(DATA_ROOT, sub_func, 
                                                    ses_func[rs_func_ses_num],
                                                    'func', 'connectivity_tmp', 
                                                    (f'schaefer400_part_corr_{sub_func}_'
                                                    f'{ses_func[rs_func_ses_num]}_'
                                                    f'dir-{direction}.csv'))
                            func = pd.read_csv(func_mat, names=atlas.labels)
                            if hemisphere == 'LH':
                                func = func.iloc[0:200, 0:200]
                            elif hemisphere == 'RH':
                                func = func.iloc[200:, 200:]
                            if similarity_metric == "cosine similarity":
                                similarity = pairwise.cosine_similarity(struc, func)
                            elif similarity_metric == "correlation":
                                similarity = np.corrcoef(struc, func)
                                similarity = np.hsplit(np.vsplit(similarity, 2)[0], 2)[1]
                            similarities.append(similarity.diagonal())
                            if aggregation == 'median':
                                agg_similarity = np.median(similarity.diagonal())
                            elif aggregation == 'mean':
                                agg_similarity = np.mean(similarity.diagonal())
                            agg_similarities.append(agg_similarity)
                        all_sub_agg_similarities.append(agg_similarities)
                    
                    print(plot_count)
                    ax = plt.subplot(2, 2, plot_count)
                    title = f'func_{rs_func_ses_num+1}_{direction}-vs-struc'
                    sns.heatmap(all_sub_agg_similarities, yticklabels=sub_ses_dwi.keys(),
                                xticklabels=sub_ses_fmri.keys(), vmax=0.11, vmin=0, ax=ax, cmap='Greens')
                    ax.set_title(title)
                    ax.set_xlabel('functional')
                    ax.set_ylabel('structural')
                    plot_count = plot_count + 1

            plt.suptitle((f"median over {similarity_metric} between functional and \n"
                            f"structural connectivity vector of each parcel in {hemisphere}"))
            plt.subplots_adjust(hspace=0.35, wspace=0.25)
