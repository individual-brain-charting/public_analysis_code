"""
This script calculates correlations between three functional conn
estimates and structural connectivity vectors of each parcel in both
hemispheres for all IBC subject pairs and then plots the diagonals as
bar plots for each variation.
"""

import numpy as np
import matplotlib.pyplot as plt
from nilearn import datasets, connectome
import pandas as pd
import os
from sklearn.metrics import pairwise
import seaborn as sns
from scipy import stats
from ibc_public.utils_data import get_subject_session
from glob import glob

func_types = ['movie', 'resting']
similarity_metrics = ['pearson']
hemispheres = ['both hemispheres']
struc_connectivity_matrices = ['mni_sift']
func_connectivity_matrices = ['pearson_corr', 'GLC_cov', 'GLC_inv_cov', 'GSC_cov', 'GSC_inv_cov']

DATA_ROOT = '/storage/store2/data/ibc/derivatives/'
# dwi session numbers for each subject
sub_ses_dwi = {'sub-01': 'ses-12', 'sub-04': 'ses-08', 'sub-05': 'ses-08',
               'sub-06': 'ses-09', 'sub-07': 'ses-09', 'sub-08': 'ses-09',
               'sub-09': 'ses-09', 'sub-11': 'ses-09', 'sub-12': 'ses-09',
               'sub-13': 'ses-09', 'sub-14': 'ses-05', 'sub-15': 'ses-12'}

tmp_dir = os.path.join(DATA_ROOT, 'sub-04', 'ses-08', 'dwi', 'tract2mni_tmp')
atlas = datasets.fetch_atlas_schaefer_2018(data_dir=tmp_dir, 
                                           resolution_mm=1,
                                           n_rois=400)
count = 0

for hemisphere in hemispheres:
    for similarity_metric in similarity_metrics:
        for struc_conn in struc_connectivity_matrices:
            diags = []
            diag_labels = []
            off_diags = []
            off_diag_labels = []
            t_tests = []
            diags_norm = []
            diag_labels_norm = []
            off_diags_norm = []
            off_diag_labels_norm = []
            for func_type in func_types:
                if func_type == 'movie':
                    subject_sessions_fmri = sorted(get_subject_session(['raiders1', 'raiders2']))
                    sub_ses_fmri = dict([(subject_sessions_fmri[2 * i][0],
                 [subject_sessions_fmri[2 * i][1], subject_sessions_fmri[2 * i + 1][1]])
                for i in range(len(subject_sessions_fmri) // 2)])
                elif func_type == 'resting':
                    sub_ses_fmri = {'sub-01': ['ses-14','ses-15'], 'sub-04': ['ses-11','ses-12'], 'sub-05': ['ses-11','ses-12'], 'sub-06': ['ses-11','ses-12'], 'sub-07': ['ses-11','ses-12'], 'sub-08': ['ses-12','ses-13'], 'sub-09': ['ses-12','ses-13'], 'sub-11': ['ses-11','ses-12'], 'sub-12': ['ses-11','ses-12'], 'sub-13': ['ses-11','ses-12'], 'sub-14': ['ses-11','ses-12'], 'sub-15': ['ses-14','ses-15']}
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
                            struc_mat = os.path.join(DATA_ROOT, sub_dwi, ses_dwi, 'dwi', 'tract2mni_tmp',
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
                            sess_funcs = sorted(sess_funcs)
                            if sub_func == 'sub-06' and space_sift[0] == 'mni':
                                continue
                            sess_sift = func_conn.split('_')
                            if sess_sift[0] == 'pearson':
                                all_func = []
                                for sess_func in sess_funcs:
                                    runs = glob(os.path.join(DATA_ROOT, sub_func, sess_func, 'func', 'connectivity_tmp', 'schaefer400_corr*ses*.csv'))
                                    for run in runs:
                                        func = pd.read_csv(run, names=atlas.labels)
                                        if hemisphere == 'LH':
                                            func = func[0:200]
                                        elif hemisphere == 'RH':
                                            func = func[200:]
                                        func = connectome.sym_matrix_to_vec(func.to_numpy(), discard_diagonal=True)
                                        all_func.append(func)
                                func = np.mean(all_func, axis=0)
                            elif sess_sift[0] == 'GLC':
                                if sess_sift[1] == 'cov':
                                    filename = f'schaefer400_corr_{sub_func}_all.csv'
                                elif sess_sift[1] == 'inv':
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
                            elif sess_sift[0] == 'GSC':
                                if sess_sift[1] == 'cov':
                                    filename = f'schaefer400_corr_{sub_func}_all_gsc.csv'
                                elif sess_sift[1] == 'inv':
                                    filename = f'schaefer400_part_corr_{sub_func}_all_gsc.csv'
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
                    subs = list(sub_ses_dwi.keys())
                    if space_sift[0] == 'mni':
                        subs.remove('sub-06')
                    all_sub_similarities = np.array(all_sub_similarities)
                    matrix_df = pd.DataFrame(all_sub_similarities, columns = subs, index = subs)
                    fig = plt.figure(figsize=(10,10))
                    sns.heatmap(matrix_df)
                    title = f'{similarity_metric} corr b/w ({func_conn}) func conn vs. struc conn'
                    plt.title(title)
                    plt.xlabel('func conn')
                    plt.ylabel('struc conn')
                    plt.savefig(f'similarity_mat_{func_conn}.png', bbox_inches='tight')
                    plt.close('all')
                    
                    all_sub_similarities_norm = (all_sub_similarities - all_sub_similarities.mean(axis=0))
                    all_sub_similarities_norm = (all_sub_similarities_norm.T - all_sub_similarities_norm.mean(axis=1)).T
                    matrix_df = pd.DataFrame(all_sub_similarities_norm, columns = subs, index = subs)
                    fig = plt.figure(figsize=(10,10))
                    sns.heatmap(matrix_df)
                    title = f'double mean centered {similarity_metric} corr b/w ({func_conn}) func conn vs. struc conn'
                    plt.title(title)
                    plt.xlabel('func conn')
                    plt.ylabel('struc conn')
                    plt.savefig(f'similarity_mat_{func_conn}_mean_centered.png', bbox_inches='tight')
                    plt.close('all')

                    off_diag = all_sub_similarities[np.where(~np.eye(all_sub_similarities.shape[0],dtype=bool))]
                    off_diags.append(off_diag)
                    off_diag_labels.append([f'{func_type}_{func_conn}' for i in range(len(off_diag))])
                    diag = np.diagonal(all_sub_similarities)
                    diags.append(diag)
                    diag_labels.append([f'{func_type}_{func_conn}' for i in range(len(diag))])

                    off_diag_norm = all_sub_similarities_norm[np.where(~np.eye(all_sub_similarities_norm.shape[0],dtype=bool))]
                    off_diags_norm.append(off_diag_norm)
                    off_diag_labels_norm.append([f'{func_type}_{func_conn}' for i in range(len(off_diag_norm))])
                    diag_norm = np.diagonal(all_sub_similarities_norm)
                    diags_norm.append(diag_norm)
                    diag_labels_norm.append([f'{func_type}_{func_conn}' for i in range(len(diag_norm))])

                    print(f'{func_type}_{func_conn}')
                    t_test = stats.ttest_ind(diag_norm, off_diag_norm, alternative='greater')
                    t_tests.append(t_test)
                    print(t_test)

            subs = list(sub_ses_dwi.keys())
            if space_sift[0] == 'mni':
                subs.remove('sub-06')
            diags = np.array(diags).flatten()
            diag_labels = np.array(diag_labels).flatten()
            diags_df = pd.DataFrame(data=diags, columns=[f'{similarity_metric} corr func conn vs. struc conn'])
            diags_df['func conn estimates'] = diag_labels
            diags_df['comparison'] = ['same subject' for i in range(len(diags_df))]
            off_diags = np.array(off_diags).flatten()
            off_diag_labels = np.array(off_diag_labels).flatten()
            off_diags_df = pd.DataFrame(data=off_diags, columns=[f'{similarity_metric} corr func conn vs. struc conn'])
            off_diags_df['func conn estimates'] = off_diag_labels
            off_diags_df['comparison'] = ['cross subject' for i in range(len(off_diags_df))]
            df = pd.concat([diags_df, off_diags_df])
            fig = plt.figure(figsize=(10,10))
            title = f''
            sns.boxplot(df, orient = 'h', x=f'{similarity_metric} corr func conn vs. struc conn', y='func conn estimates', hue='comparison', fliersize=0)
            plt.title(title)
            plt.savefig('similarity_boxplot_func_conns_gsc.png', bbox_inches='tight')
            plt.close()


            diags_norm = np.array(diags_norm).flatten()
            diag_labels_norm = np.array(diag_labels_norm).flatten()
            diags_df = pd.DataFrame(data=diags_norm, columns=[f'double mean centered {similarity_metric} corr func conn vs. struc conn'])
            diags_df['func conn estimates'] = diag_labels_norm
            diags_df['comparison'] = ['same subject' for i in range(len(diags_df))]
            off_diags_norm = np.array(off_diags_norm).flatten()
            off_diag_labels_norm = np.array(off_diag_labels_norm).flatten()
            off_diags_df = pd.DataFrame(data=off_diags_norm, columns=[f'double mean centered {similarity_metric} corr func conn vs. struc conn'])
            off_diags_df['func conn estimates'] = off_diag_labels_norm
            off_diags_df['comparison'] = ['cross subject' for i in range(len(off_diags_df))]
            df = pd.concat([diags_df, off_diags_df])
            fig = plt.figure(figsize=(10,10))
            title = f''
            sns.boxplot(df, orient = 'h', x=f'double mean centered {similarity_metric} corr func conn vs. struc conn', y='func conn estimates', hue='comparison', fliersize=0)
            plt.title(title)
            plt.savefig('similarity_boxplot_func_conns_gsc_mean_center.png', bbox_inches='tight')
            plt.close()