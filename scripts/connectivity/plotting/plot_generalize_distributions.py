import os
import pandas as pd
import seaborn as sns
import numpy as np
from nilearn.connectome import vec_to_sym_matrix
import matplotlib.pyplot as plt
from nilearn import datasets
from ibc_public.connectivity.utils_plot import get_lower_tri_heatmap

sns.set_theme()
sns.set_style("whitegrid")
sns.set_context("talk")

### transfer IBC -> external connectivity distribution plots ###
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

dist_dir = os.path.join(
    DATA_ROOT, "transfer_ibc_external_connectivity_distributions"
)
os.makedirs(dist_dir, exist_ok=True)
for cov in cov_estimators:
    for measure in measures:
        fig, ax = plt.subplots()
        for dataset in ["ibc", "external"]:
            if dataset == "ibc":
                # load the data
                fc_data = pd.read_pickle(
                    os.path.join(IBC_ROOT, "connectomes_200.pkl")
                )
                color = "blue"
            elif dataset == "external":
                # load the data
                fc_data = pd.read_pickle(
                    os.path.join(external_ROOT, "connectomes_200.pkl")
                )
                color = "red"
            try:
                vector = np.mean(
                    np.vstack(list(fc_data[f"{cov} {measure}"])),
                    axis=0,
                )
                matrix = vec_to_sym_matrix(vector, diagonal=np.ones(200))
            except ValueError:
                print(f"{cov} {measure} does not exist")
                continue
            get_lower_tri_heatmap(
                matrix,
                title=f"{dataset} {cov} {measure}",
                output=os.path.join(
                    dist_dir, f"mat_{dataset}_{cov}_{measure}"
                ),
                ticks=ticks,
                labels=unique_labels,
                grid=True,
                diag=True,
                triu=True,
            )
            ax.hist(
                vector,
                bins=500,
                density=True,
                log=True,
                label=dataset,
                color=color,
                alpha=0.5,
            )
            ax.axvline(
                np.mean(vector), linestyle="dashed", linewidth=1, color=color
            )
        ax.legend()
        ax.set_title(f"{cov} {measure}")
        plt.savefig(
            os.path.join(dist_dir, f"dist_{cov}_{measure}.png"),
            bbox_inches="tight",
        )
        plt.close(fig)
