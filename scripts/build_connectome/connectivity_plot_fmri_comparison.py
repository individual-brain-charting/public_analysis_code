"""
This script calculates correlations between three functional conn estimates 
from movie and resting state of each subject and then plots the diagonals as 
bar plots for each variation.
"""

import numpy as np
import matplotlib.pyplot as plt
from nilearn import datasets, plotting, connectome
import pandas as pd
import os
from sklearn.metrics import pairwise
import seaborn as sns
from scipy import stats
from ibc_public.utils_data import get_subject_session
from glob import glob

similarity_metric = "pearson"
func_connectivity_matrices = ["pearson_corr", "GLC_cov", "GLC_inv_cov"]
func_connectivity_matrices = [
    "pearson_corr",
    "GLC_cov",
    "GLC_inv_cov",
    "GSC_cov",
    "GSC_inv_cov",
]

DATA_ROOT = "/storage/store2/data/ibc/derivatives/"
# dwi session numbers for each subject
sub_ses_dwi = {
    "sub-01": "ses-12",
    "sub-04": "ses-08",
    "sub-05": "ses-08",
    "sub-06": "ses-09",
    "sub-07": "ses-09",
    "sub-08": "ses-09",
    "sub-09": "ses-09",
    "sub-11": "ses-09",
    "sub-12": "ses-09",
    "sub-13": "ses-09",
    "sub-14": "ses-05",
    "sub-15": "ses-12",
}

tmp_dir = os.path.join(DATA_ROOT, "sub-04", "ses-08", "dwi", "tract2mni_tmp")
atlas = datasets.fetch_atlas_schaefer_2018(
    data_dir=tmp_dir, resolution_mm=1, n_rois=400
)
count = 0


subject_sessions_fmri = sorted(get_subject_session(["raiders1", "raiders2"]))
movie_sub_ses_fmri = dict(
    [
        (
            subject_sessions_fmri[2 * i][0],
            [
                subject_sessions_fmri[2 * i][1],
                subject_sessions_fmri[2 * i + 1][1],
            ],
        )
        for i in range(len(subject_sessions_fmri) // 2)
    ]
)

resting_sub_ses_fmri = {
    "sub-01": ["ses-14", "ses-15"],
    "sub-04": ["ses-11", "ses-12"],
    "sub-05": ["ses-11", "ses-12"],
    "sub-06": ["ses-11", "ses-12"],
    "sub-07": ["ses-11", "ses-12"],
    "sub-08": ["ses-12", "ses-13"],
    "sub-09": ["ses-12", "ses-13"],
    "sub-11": ["ses-11", "ses-12"],
    "sub-12": ["ses-11", "ses-12"],
    "sub-13": ["ses-11", "ses-12"],
    "sub-14": ["ses-11", "ses-12"],
    "sub-15": ["ses-14", "ses-15"],
}

matrices = []
diags = []
off_diags = []
diag_labels = []
off_diag_labels = []
t_tests = []
for func_conn in func_connectivity_matrices:
    matrix = []
    for sub_movie, ses_movies in movie_sub_ses_fmri.items():
        print("movie", sub_movie)
        ses_movies = sorted(ses_movies)
        sess_sift = func_conn.split("_")
        if sess_sift[0] == "pearson":
            movie_all_fun = []
            for ses_movie in ses_movies:
                runs = glob(
                    os.path.join(
                        DATA_ROOT,
                        sub_movie,
                        ses_movie,
                        "func",
                        "connectivity_tmp",
                        "schaefer400_corr*ses*.csv",
                    )
                )
                for run in runs:
                    movie_func = pd.read_csv(run, names=atlas.labels)
                    movie_func = connectome.sym_matrix_to_vec(
                        movie_func.to_numpy(), discard_diagonal=True
                    )
                    movie_all_fun.append(movie_func)
            movie_func = np.mean(movie_all_fun, axis=0)
        elif sess_sift[0] == "GLC":
            if sess_sift[1] == "cov":
                filename = f"schaefer400_corr_{sub_movie}_all.csv"
            elif sess_sift[1] == "inv":
                filename = f"schaefer400_part_corr_{sub_movie}_all.csv"
            func_mat = os.path.join(
                DATA_ROOT,
                sub_movie,
                ses_movies[1],
                "func",
                "connectivity_tmp",
                filename,
            )
            movie_func = pd.read_csv(func_mat, names=atlas.labels)
            movie_func = connectome.sym_matrix_to_vec(
                movie_func.to_numpy(), discard_diagonal=True
            )
        elif sess_sift[0] == "GSC":
            if sess_sift[1] == "cov":
                filename = f"schaefer400_corr_{sub_movie}_all_gsc.csv"
            elif sess_sift[1] == "inv":
                filename = f"schaefer400_part_corr_{sub_movie}_all_gsc.csv"
            func_mat = os.path.join(
                DATA_ROOT,
                sub_movie,
                ses_movies[1],
                "func",
                "connectivity_tmp",
                filename,
            )
            movie_func = pd.read_csv(func_mat, names=atlas.labels)
            movie_func = connectome.sym_matrix_to_vec(
                movie_func.to_numpy(), discard_diagonal=True
            )
        movie_func = movie_func.reshape(1, -1)

        row = []
        for sub_func, sess_funcs in resting_sub_ses_fmri.items():
            print("rest", sub_func)
            sess_funcs = sorted(sess_funcs)
            sess_sift = func_conn.split("_")
            if sess_sift[0] == "pearson":
                all_func = []
                for sess_func in sess_funcs:
                    runs = glob(
                        os.path.join(
                            DATA_ROOT,
                            sub_func,
                            sess_func,
                            "func",
                            "connectivity_tmp",
                            "schaefer400_corr*ses*.csv",
                        )
                    )
                    for run in runs:
                        func = pd.read_csv(run, names=atlas.labels)
                        func = connectome.sym_matrix_to_vec(
                            func.to_numpy(), discard_diagonal=True
                        )
                        all_func.append(func)
                func = np.mean(all_func, axis=0)
            elif sess_sift[0] == "GLC":
                if sess_sift[1] == "cov":
                    filename = f"schaefer400_corr_{sub_func}_all.csv"
                elif sess_sift[1] == "inv":
                    filename = f"schaefer400_part_corr_{sub_func}_all.csv"
                func_mat = os.path.join(
                    DATA_ROOT,
                    sub_func,
                    sess_funcs[1],
                    "func",
                    "connectivity_tmp",
                    filename,
                )
                func = pd.read_csv(func_mat, names=atlas.labels)
                func = connectome.sym_matrix_to_vec(
                    func.to_numpy(), discard_diagonal=True
                )
            elif sess_sift[0] == "GSC":
                if sess_sift[1] == "cov":
                    filename = f"schaefer400_corr_{sub_func}_all_gsc.csv"
                elif sess_sift[1] == "inv":
                    filename = f"schaefer400_part_corr_{sub_func}_all_gsc.csv"
                func_mat = os.path.join(
                    DATA_ROOT,
                    sub_func,
                    sess_funcs[1],
                    "func",
                    "connectivity_tmp",
                    filename,
                )
                func = pd.read_csv(func_mat, names=atlas.labels)
                func = connectome.sym_matrix_to_vec(
                    func.to_numpy(), discard_diagonal=True
                )
            func = func.reshape(1, -1)

            if similarity_metric == "cosine":
                similarity = pairwise.cosine_similarity(movie_func, func)
                similarity = similarity[0][0]
            elif similarity_metric == "pearson":
                similarity = np.corrcoef(movie_func, func)
                similarity = np.hsplit(np.vsplit(similarity, 2)[0], 2)[1]
                similarity = similarity[0][0]
            elif similarity_metric == "spearman":
                similarity, _ = stats.spearmanr(movie_func, func, axis=1)

            row.append(similarity)
        matrix.append(row)

    subs = list(resting_sub_ses_fmri.keys())
    matrix = np.array(matrix)
    diag = np.diagonal(matrix)
    diags.append(diag)
    diag_labels.append([f"{func_conn}" for i in range(len(diag))])
    off_diag = matrix[np.where(~np.eye(matrix.shape[0], dtype=bool))]
    off_diags.append(off_diag)
    off_diag_labels.append([f"{func_conn}" for i in range(len(off_diag))])
    matrix_df = pd.DataFrame(matrix, columns=subs, index=subs)
    fig = plt.figure(figsize=(10, 10))
    sns.heatmap(matrix_df, cmap="RdYlBu")
    plt.xlabel("resting")
    plt.ylabel("movie")
    plt.savefig(f"func_conns_corr_mat_{func_conn}.png", bbox_inches="tight")
    matrices.append(matrix)
    plt.close("all")

    print(func_conn)
    t_test = stats.ttest_ind(diag, off_diag, alternative="greater")
    t_tests.append(t_test)
    print(t_test)

diags = np.array(diags).flatten()
diag_labels = np.array(diag_labels).flatten()
diags_df = pd.DataFrame(
    data=diags,
    columns=[f"{similarity_metric} corr movie vs. resting-state func conn"],
)
diags_df["func conn estimates"] = diag_labels
diags_df["comparison"] = ["same subject" for i in range(len(diags_df))]
off_diags = np.array(off_diags).flatten()
off_diag_labels = np.array(off_diag_labels).flatten()
off_diags_df = pd.DataFrame(
    data=off_diags,
    columns=[f"{similarity_metric} corr movie vs. resting-state func conn"],
)
off_diags_df["func conn estimates"] = off_diag_labels
off_diags_df["comparison"] = [
    "cross subject" for i in range(len(off_diags_df))
]
df = pd.concat([diags_df, off_diags_df])
fig = plt.figure(figsize=(10, 10))
title = f""
sns.boxplot(
    df,
    orient="h",
    x=f"{similarity_metric} corr movie vs. resting-state func conn",
    y="func conn estimates",
    hue="comparison",
    fliersize=0,
)
plt.title(title)
plt.savefig(f"func_conns_corr_boxplot_gsc.png", bbox_inches="tight")
plt.close("all")
