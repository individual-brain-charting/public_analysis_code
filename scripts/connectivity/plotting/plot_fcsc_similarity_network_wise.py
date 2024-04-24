import os
import pandas as pd
import seaborn as sns
import numpy as np
from nilearn.connectome import vec_to_sym_matrix
from nilearn import datasets
from sklearn import preprocessing
from ibc_public.connectivity.utils_plot import (
    get_network_labels,
    get_lower_tri_heatmap,
    mean_connectivity,
)

sns.set_theme()
sns.set_style("whitegrid")
sns.set_context("talk")

### fc-sc similarity, network-wise matrix

cache = DATA_ROOT = "/storage/store2/work/haggarwa/"

labels_fmt = "network"
n_parcels = 400
if n_parcels == 400:
    fc_data_path = os.path.join(cache, "connectomes_400_comprcorr")
    sc_data_path = os.path.join(cache, "sc_data_native_new")
    output_dir = "fc-sc_similarity_networkwise_plots_compcorr"
elif n_parcels == 200:
    fc_data_path = os.path.join(cache, "connectomes_200_comprcorr")
    sc_data_path = os.path.join(cache, "sc_data_native_200")
    output_dir = "fc-sc_similarity_networkwise_plots_200_compcorr"

output_dir = os.path.join(DATA_ROOT, output_dir)
os.makedirs(output_dir, exist_ok=True)
fc_data = pd.read_pickle(fc_data_path)
sc_data = pd.read_pickle(sc_data_path)

# cov estimators
cov_estimators = ["Graphical-Lasso", "Ledoit-Wolf", "Unregularized"]
# connectivity measures for each cov estimator
measures = ["correlation", "partial correlation"]
# tasks
tasks = [
    "RestingState",
    "Raiders",
    "GoodBadUgly",
    "MonkeyKingdom",
    "Mario",
]

# get atlas for yeo network labels
atlas = datasets.fetch_atlas_schaefer_2018(
    data_dir=cache, resolution_mm=2, n_rois=n_parcels
)

fc_data = mean_connectivity(fc_data, tasks, cov_estimators, measures)
fc_data.reset_index(drop=True, inplace=True)
sc_data.reset_index(drop=True, inplace=True)

if labels_fmt == "hemi network":
    labels = get_network_labels(atlas)[0]
elif labels_fmt == "network":
    labels = get_network_labels(atlas)[1]

le = preprocessing.LabelEncoder()
encoded_labels = le.fit_transform(labels)

unique_labels = np.unique(encoded_labels)

results = []

for cov in cov_estimators:
    for measure in measures:
        for task in tasks:
            func = fc_data[fc_data["task"] == task]
            func = func[func["measure"] == cov + " " + measure]
            # get func and struc conn for each subject
            for sub in func["subject"].unique():
                sub_func = func[func["subject"] == sub]
                sub_func_mat = vec_to_sym_matrix(
                    sub_func["connectivity"].values[0],
                    diagonal=np.ones(n_parcels),
                )
                sub_func_mat[np.triu_indices_from(sub_func_mat)] = np.nan
                sub_struc = sc_data[sc_data["subject"] == sub]
                sub_struc_mat = vec_to_sym_matrix(
                    sub_struc["connectivity"].values[0],
                    diagonal=np.ones(n_parcels),
                )
                sub_struc_mat[np.triu_indices_from(sub_struc_mat)] = np.nan

                # create empty matrix for network pair correlations
                network_pair_corr = np.zeros(
                    (len(unique_labels), len(unique_labels))
                )
                print(f"\n\n{task} {sub} {cov} {measure}\n\n")
                # get the nodes indices for each network
                for network_i in unique_labels:
                    index_i = np.where(encoded_labels == network_i)[0]
                    # print(index_i)
                    for network_j in unique_labels:
                        index_j = np.where(encoded_labels == network_j)[0]
                        # print(index_j)
                        # func connectivity for network pair
                        sub_func_network = sub_func_mat[
                            np.ix_(index_i, index_j)
                        ]
                        sub_func_network = sub_func_network[
                            ~np.isnan(sub_func_network)
                        ].flatten()
                        # print(sub_func_network)
                        # struc connectivity for network pair
                        sub_struc_network = sub_struc_mat[
                            np.ix_(index_i, index_j)
                        ]
                        sub_struc_network = sub_struc_network[
                            ~np.isnan(sub_struc_network)
                        ].flatten()
                        # print(sub_struc_network)
                        # correlation between func and struc connectivity
                        corr = np.corrcoef(sub_struc_network, sub_func_network)
                        print(corr, f"{task} {sub} {cov} {measure}")
                        network_pair_corr[network_i][network_j] = corr[0][1]
                result = {
                    "corr": network_pair_corr,
                    "task": task,
                    "subject": sub,
                    "cov measure": cov + " " + measure,
                }
                results.append(result)

results = pd.DataFrame(results)

for _, row in results.iterrows():
    task = row["task"]
    sub = row["subject"]
    cov = row["cov measure"]
    corr = row["corr"]
    title = f"{task} {sub} {cov}"
    output = os.path.join(output_dir, f"{task}_{sub}_{cov}")
    get_lower_tri_heatmap(
        corr,
        figsize=(5, 5),
        cmap="hot_r",
        title=title,
        labels=le.inverse_transform(unique_labels),
        output=output,
        triu=True,
    )

# take a mean of the network wise correlations across subjects
# gives a network wise correlation matrix for each task and cov measure
fc_sc_corr_tasks = (
    results.groupby(["task", "cov measure"]).mean().reset_index()
)
for _, row in results.iterrows():
    task = row["task"]
    sub = row["subject"]
    cov = row["cov measure"]
    corr = row["corr"]
    title = f"{task} {sub} {cov}"
    output = os.path.join(output_dir, f"mean_{task}_{cov}")
    get_lower_tri_heatmap(
        corr,
        figsize=(5, 5),
        cmap="hot_r",
        title=title,
        labels=le.inverse_transform(unique_labels),
        output=output,
        triu=True,
    )

# get the mean of the network wise correlations across tasks
# gives a network wise correlation matrix for each cov measure and subject
fc_sc_corr_subjects = (
    results.groupby(["task", "cov measure"]).mean().reset_index()
)
for _, row in fc_sc_corr_subjects.iterrows():
    task = row["task"]
    sub = row["subject"]
    cov = row["cov measure"]
    corr = row["corr"]
    title = f"{task} {sub} {cov}"
    output = os.path.join(output_dir, f"mean_{sub}_{cov}")
    get_lower_tri_heatmap(
        corr,
        figsize=(5, 5),
        cmap="hot_r",
        title=title,
        labels=le.inverse_transform(unique_labels),
        output=output,
        triu=True,
    )

# get the mean of the network wise correlations across tasks and subjects
# gives a network wise correlation matrix for each cov measure
fc_sc_corr = results.groupby(["cov measure"]).mean().reset_index()
for _, row in fc_sc_corr.iterrows():
    task = row["task"]
    sub = row["subject"]
    cov = row["cov measure"]
    corr = row["corr"]
    title = f"{task} {sub} {cov}"
    output = os.path.join(output_dir, f"mean_{cov}")
    get_lower_tri_heatmap(
        corr,
        figsize=(5, 5),
        cmap="hot_r",
        title=title,
        labels=le.inverse_transform(unique_labels),
        output=output,
        triu=True,
    )
