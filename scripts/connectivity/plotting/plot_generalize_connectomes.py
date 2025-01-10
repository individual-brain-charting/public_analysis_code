import os
import pandas as pd
import seaborn as sns
import numpy as np
from nilearn.connectome import vec_to_sym_matrix
import matplotlib.pyplot as plt
from nilearn import datasets
from tqdm import tqdm
from ibc_public.connectivity.utils_plot import get_lower_tri_heatmap

sns.set_theme()
sns.set_style("whitegrid")
sns.set_context("talk")

### transfer IBC -> external connectivity matrices ###
# get atlas for yeo network labels
cache = "/storage/store2/work/haggarwa/"
DATA_ROOT = "/storage/store2/work/haggarwa/"
IBC_ROOT = os.path.join(
    DATA_ROOT, "ibc_sync_external_connectivity_20231206-110710"
)
external_ROOT = os.path.join(
    DATA_ROOT, "external_connectivity_20231205-142311"
)

atlas = datasets.fetch_atlas_schaefer_2018(
    data_dir=cache, resolution_mm=2, n_rois=200
)
networks = atlas["labels"].astype("U")
hemi_network_labels = []
for network in networks:
    components = network.split("_")
    hemi_network = "_".join(components[1:3])
    hemi_network_labels.append(hemi_network)
ticks = []
unique_labels = []
for i, label in enumerate(hemi_network_labels):
    if label != hemi_network_labels[i - 1]:
        ticks.append(i)
        unique_labels.append(label)
# cov estimators
cov_estimators = ["Graphical-Lasso", "Ledoit-Wolf", "Unregularized"]
# connectivity measures for each cov estimator
measures = ["correlation", "partial correlation"]
for dataset in ["ibc", "external"]:
    if dataset == "ibc":
        # load the data
        fc_data = pd.read_pickle(os.path.join(IBC_ROOT, "connectomes_200.pkl"))
        mats_dir = os.path.join(IBC_ROOT, "connectivity_matrices")
    elif dataset == "external":
        # load the data
        fc_data = pd.read_pickle(
            os.path.join(external_ROOT, "connectomes_200.pkl")
        )
        mats_dir = os.path.join(external_ROOT, "connectivity_matrices")
    _, uniq_idx = np.unique(hemi_network_labels, return_index=True)
    hemi_network_labels = np.array(hemi_network_labels)[np.sort(uniq_idx)]
    sns.set_context("notebook", font_scale=1.05)
    os.makedirs(mats_dir, exist_ok=True)
    for sub in tqdm(np.unique(fc_data["subject_ids"]), desc=dataset):
        for cov in cov_estimators:
            for measure in measures:
                try:
                    vector = np.mean(
                        np.vstack(
                            list(
                                fc_data[(fc_data["subject_ids"] == sub)][
                                    f"{cov} {measure}"
                                ]
                            )
                        ),
                        axis=0,
                    )
                    matrix = vec_to_sym_matrix(vector, diagonal=np.ones(200))
                except ValueError:
                    print(f"{sub} {cov} {measure} does not exist")
                    continue
                get_lower_tri_heatmap(
                    matrix,
                    title=f"{sub} {dataset}",
                    output=os.path.join(mats_dir, f"{sub}_{cov}_{measure}"),
                    ticks=ticks,
                    labels=hemi_network_labels,
                    grid=True,
                    diag=True,
                    triu=True,
                )
                plt.close("all")
