"""
This script calculates and plots {mean, median} over 
{cosine similarities, correlations} between functional and structural 
connectivity vectors of each parcel in {left, right, both} hemispheres for 
all IBC subject pairs
"""

import numpy as np
import matplotlib.pyplot as plt
from nilearn import datasets, plotting, connectome
import pandas as pd
import os
from sklearn.metrics import pairwise
from sklearn.preprocessing import normalize
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
coords_file = os.path.join(tmp_dir, 'schaefer_2018', 'ras_coords_400.csv')
coords = pd.read_csv(coords_file)
coords = coords[['R','A','S']]

rs_func_ses_num = 1
hemisphere = 'both'
similarity_metric = 'correlation'
aggregation = 'mean'

all_sub_similarities = []
for sub_dwi, ses_dwi in sub_ses_dwi.items():
    # if sub_dwi == 'sub-06':
    #     continue
    struc_mat = os.path.join(DATA_ROOT, sub_dwi, ses_dwi,
                            'dwi', 'atlas2dwi', (f'schaefer400_connectome_'
                                    f'sift-weighted_{sub_dwi}_{ses_dwi}_t2'
                                    f'.csv'))
    struc = pd.read_csv(struc_mat, names=atlas.labels)
    if hemisphere == 'LH':
        struc = struc[0:200]
    elif hemisphere == 'RH':
        struc = struc[200:]
    struc = connectome.sym_matrix_to_vec(struc.to_numpy(), discard_diagonal=True)
    struc = struc.reshape(1, -1)
    similarities = []
    agg_similarities = []
    for sub_func, ses_func in sub_ses_fmri.items():
        # if sub_func == 'sub-06':
        #     continue
        func_mat = os.path.join(DATA_ROOT, sub_func, 
                                ses_func[rs_func_ses_num],
                                'func', 'connectivity_tmp', 
                                f'schaefer400_part_corr_{sub_func}_all_compcorr.csv')
        func = pd.read_csv(func_mat, names=atlas.labels)
        if hemisphere == 'LH':
            func = func[0:200]
        elif hemisphere == 'RH':
            func = func[200:]
        func = connectome.sym_matrix_to_vec(func.to_numpy(), discard_diagonal=True)
        func = func.reshape(1, -1)
        if similarity_metric == "cosine similarity":
            similarity = pairwise.cosine_similarity(struc, func)
        elif similarity_metric == "correlation":
            similarity = np.corrcoef(struc, func)
            similarity = np.hsplit(np.vsplit(similarity, 2)[0], 2)[1]
        similarities.append(similarity[0][0])
    all_sub_similarities.append(similarities)

all_sub_similarities = np.array(all_sub_similarities)
# all_sub_similarities = normalize(all_sub_similarities, axis=0, norm='l1')
# all_sub_similarities = (all_sub_similarities - all_sub_similarities.mean(axis=0)) / all_sub_similarities.std(axis=0)
print(all_sub_similarities)
fig = plt.figure(figsize=(10,10))
title = f'{similarity_metric}, {hemisphere}, func-vs-struc'
subs = list(sub_ses_dwi.keys())
# subs.remove('sub-06')
sns.heatmap(all_sub_similarities, yticklabels=subs,
            xticklabels=subs, #vmax=0.11, vmin=0,ax=ax,
            # vmin=0,
            cmap='Greens')
plt.title(title)
plt.xlabel('functional')
plt.ylabel('structural')
plt.show()