"""
This script calculates and plots {cosine similarities, correlations} between
functional and structural connectivity vectors of each parcel in 
{left, right, both} hemispheres for all IBC subject pairs
"""

import numpy as np
import matplotlib.pyplot as plt
from nilearn import datasets, plotting, connectome
import pandas as pd
import os
from sklearn.metrics import pairwise
from sklearn.preprocessing import normalize
import seaborn as sns
from scipy import stats
from termcolor import colored

similarity_metrics = ['pearson', 'spearman', 'cosine']
hemispheres = ['both hemispheres', 'LH', 'RH']
struc_connectivity_matrices = ['mni_sift', 'mni_no_sift', 'native_sift',
                               'native_no_sift']
func_connectivity_matrices = ['each', 'all_corr', 'all_part_corr']

DATA_ROOT = '/data/parietal/store2/data/ibc/derivatives/'
# dwi session numbers for each subject
# subject_sessions_dwi = sorted(get_subject_session(['anat1']))
# sub_ses_dwi = dict([(subject_sessions_dwi[2 * i][0],
#                  [subject_sessions_dwi[2 * i][1], subject_sessions_dwi[2 * i + 1][1]])
#                 for i in range(len(subject_sessions_dwi) // 2)])
sub_ses_dwi = {'sub-01': 'ses-12', 'sub-04': 'ses-08', 'sub-05': 'ses-08',
               'sub-06': 'ses-09', 'sub-07': 'ses-09', 'sub-08': 'ses-09',
               'sub-09': 'ses-09', 'sub-11': 'ses-09', 'sub-12': 'ses-09',
               'sub-13': 'ses-09', 'sub-14': 'ses-05', 'sub-15': 'ses-12'}
# resting session numbers for each subject
subject_sessions_fmri = sorted(get_subject_session(['raiders1', 'raiders2']))
sub_ses_fmri = dict([(subject_sessions_fmri[2 * i][0],
                 [subject_sessions_fmri[2 * i][1], subject_sessions_fmri[2 * i + 1][1]])
                for i in range(len(subject_sessions_fmri) // 2)])
# sub_ses_fmri = {'sub-01': ['ses-14','ses-15'], 'sub-04': ['ses-11','ses-12'],
#                 'sub-05': ['ses-11','ses-12'], 'sub-06': ['ses-11','ses-12'],
#                 'sub-07': ['ses-11','ses-12'], 'sub-08': ['ses-12','ses-13'],
#                 'sub-09': ['ses-12','ses-13'], 'sub-11': ['ses-11','ses-12'],
#                 'sub-12': ['ses-11','ses-12'], 'sub-13': ['ses-11','ses-12'],
#                 'sub-14': ['ses-11','ses-12'], 'sub-15': ['ses-14','ses-15']}
tmp_dir = os.path.join(DATA_ROOT, 'sub-04', 'ses-08', 'dwi', 'tract2mni_tmp')
atlas = datasets.fetch_atlas_schaefer_2018(data_dir=tmp_dir, 
                                           resolution_mm=1,
                                           n_rois=400)
hemisphere = 'LH'
similarity_metric = 'spearman'
struc_conn = 'mni_no_sift'
func_conn = 'all_part_corr'
count = 0
for hemisphere in hemispheres:
    for similarity_metric in similarity_metrics:
        for struc_conn in struc_connectivity_matrices:
            for func_conn in func_connectivity_matrices:
                all_sub_similarities = []
                for sub_dwi, ses_dwi in sub_ses_dwi.items():
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
                    if hemisphere == 'LH':
                        struc = struc[0:200]
                    elif hemisphere == 'RH':
                        struc = struc[200:]
                    struc = connectome.sym_matrix_to_vec(struc.to_numpy(), discard_diagonal=True)
                    struc = struc.reshape(1, -1)
                    similarities = []
                    agg_similarities = []
                    for sub_func, sess_funcs in sub_ses_fmri.items():
                        if sub_func == 'sub-06' and space_sift[0] == 'mni':
                            continue
                        sess_sift = func_conn.split('_')
                        if sess_sift[0] == 'each':
                            all_func = []
                            for sess_func in sess_funcs:
                                runs = glob(os.path.join(DATA_ROOT, sub_func, sess_func, 'func', 'connectivity_func', 'schaefer400_corr*ses*.csv'))
                                for run in runs:
                                    func = pd.read_csv(run, names=atlas.labels)
                                    if hemisphere == 'LH':
                                        func = func[0:200]
                                    elif hemisphere == 'RH':
                                        func = func[200:]
                                    func = connectome.sym_matrix_to_vec(func.to_numpy(), discard_diagonal=True)
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
                            if hemisphere == 'LH':
                                func = func[0:200]
                            elif hemisphere == 'RH':
                                func = func[200:]
                            func = connectome.sym_matrix_to_vec(func.to_numpy(), discard_diagonal=True)
                        func = func.reshape(1, -1)
                        if similarity_metric == "cosine":
                            similarity = pairwise.cosine_similarity(struc, func)
                            similarity = similarity[0][0]
                        elif similarity_metric == "pearson":
                            similarity = np.corrcoef(struc, func)
                            similarity = np.hsplit(np.vsplit(similarity, 2)[0], 2)[1]
                            similarity = similarity[0][0]
                        elif similarity_metric == "spearman":
                            similarity, _ = stats.spearmanr(struc, func, axis=1)
                        similarities.append(similarity)
                    all_sub_similarities.append(similarities)
                all_sub_similarities = np.array(all_sub_similarities)
                # all_sub_similarities = normalize(all_sub_similarities, axis=0, norm='max')
                all_sub_similarities = (all_sub_similarities - all_sub_similarities.mean(axis=0))
                all_sub_similarities = (all_sub_similarities.T - all_sub_similarities.mean(axis=1)).T
                off_diag = all_sub_similarities[np.where(~np.eye(all_sub_similarities.shape[0],dtype=bool))]
                diag = np.diagonal(all_sub_similarities)
                t_test = stats.ttest_ind(diag, off_diag, alternative='greater')
                # print(t_test)
                # # print(all_sub_similarities)
                fig = plt.figure(figsize=(10,10))
                title = f'{similarity_metric}, {hemisphere}, func({func_conn})-vs-struc({struc_conn})'
                subs = list(sub_ses_dwi.keys())
                if space_sift[0] == 'mni':
                    subs.remove('sub-06')
                sns.heatmap(all_sub_similarities, yticklabels=subs,
                            xticklabels=subs, #vmax=0.11, vmin=0,ax=ax,
                            # vmin=0,
                            cmap='Greens')
                plt.title(title)
                plt.xlabel('functional')
                plt.ylabel('structural')
                if t_test[1] < 0.05:
                    print(colored(f'{count}_{similarity_metric}, {hemisphere}, func({func_conn})-vs-struc({struc_conn})', 'yellow'))
                    print(colored(t_test, 'yellow'))
                else:
                    print(f'{count}_{similarity_metric}, {hemisphere}, func({func_conn})-vs-struc({struc_conn})')
                    print(t_test)
                plt.savefig(f'movie_similarity/{count}_{similarity_metric}_{hemisphere}_func-{func_conn}_struc-{struc_conn}.png', bbox_inches='tight')
                count = count + 1
                plt.close()
                # # plt.show()
