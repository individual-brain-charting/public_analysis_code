import os
import pandas as pd
import seaborn as sns
import numpy as np
from nilearn.connectome import vec_to_sym_matrix
from nilearn.plotting import plot_connectome, view_connectome
import matplotlib.pyplot as plt
from nilearn import datasets
from tqdm import tqdm
from sklearn import preprocessing
from ibc_public.connectivity.utils_plot import get_lower_tri_heatmap

sns.set_theme()
sns.set_style("whitegrid")
sns.set_context("talk")


### functional connectivity plots
# get atlas for yeo network labels
n_parcels = 400
cache = DATA_ROOT = "/storage/store2/work/haggarwa/"
atlas = datasets.fetch_atlas_schaefer_2018(
    data_dir=cache, resolution_mm=2, n_rois=n_parcels
)
networks = atlas["labels"].astype("U")
hemi_network_labels = []
network_labels = []
rename_networks = {
    "Vis": "Visual",
    "Cont": "FrontPar",
    "SalVentAttn": "VentAttn",
}
for network in networks:
    components = network.split("_")
    components[2] = rename_networks.get(components[2], components[2])
    hemi_network = " ".join(components[1:3])
    hemi_network_labels.append(hemi_network)
    network_labels.append(components[2])
ticks = []
unique_hemi_network_labels = []
for i, label in enumerate(hemi_network_labels):
    if label != hemi_network_labels[i - 1]:
        ticks.append(i)
        unique_hemi_network_labels.append(label)

le = preprocessing.LabelEncoder()
encoded_labels = le.fit_transform(network_labels)
unique_encoded_labels = np.unique(encoded_labels)
# cov estimators
cov_estimators = ["Graphical-Lasso", "Ledoit-Wolf", "Unregularized"]
# connectivity measures for each cov estimator
measures = ["correlation", "partial correlation"]
# what to classify
classify = ["Tasks", "Subjects", "Runs"]
# tasks
tasks = ["RestingState", "Raiders", "GoodBadUgly", "MonkeyKingdom", "Mario"]
# load the data
if n_parcels == 400:
    fc_data = pd.read_pickle(os.path.join(DATA_ROOT, "connectomes2"))
    coords_file = "Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.Centroid_RAS.csv"
    mats_dir = os.path.join(DATA_ROOT, "connectivity_matrices")
    brain_dir = os.path.join(DATA_ROOT, "brain_connectivity")
    html_dir = os.path.join(DATA_ROOT, "brain_connectivity_html")
elif n_parcels == 200:
    fc_data = pd.read_pickle(
        os.path.join(DATA_ROOT, "connectomes_200_parcels")
    )
    coords_file = "Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.Centroid_RAS.csv"
    mats_dir = os.path.join(DATA_ROOT, "connectivity_matrices_200")
    brain_dir = os.path.join(DATA_ROOT, "brain_connectivity_200")
    html_dir = os.path.join(DATA_ROOT, "brain_connectivity_html_200")
sns.set_context("notebook", font_scale=1.05)
os.makedirs(mats_dir, exist_ok=True)
os.makedirs(brain_dir, exist_ok=True)
os.makedirs(html_dir, exist_ok=True)
coords = pd.read_csv(
    os.path.join(
        cache,
        "schaefer_2018",
        coords_file,
    )
)[["R", "A", "S"]].to_numpy()
for sub in tqdm(np.unique(fc_data["subject_ids"])):
    for task in tasks:
        for cov in cov_estimators:
            for measure in measures:
                try:
                    matrix = vec_to_sym_matrix(
                        np.mean(
                            np.vstack(
                                list(
                                    fc_data[
                                        (fc_data["subject_ids"] == sub)
                                        & (fc_data["tasks"] == task)
                                    ][f"{cov} {measure}"]
                                )
                            ),
                            axis=0,
                        ),
                        diagonal=np.ones(n_parcels),
                    )
                except ValueError as e:
                    print(e)
                    print(f"{sub} {task} {cov} {measure} does not exist")
                    continue
                get_lower_tri_heatmap(
                    matrix,
                    title=f"{sub} {task}",
                    output=os.path.join(
                        mats_dir, f"{sub}_{task}_{cov}_{measure}"
                    ),
                    ticks=ticks,
                    labels=unique_hemi_network_labels,
                    grid=True,
                    diag=True,
                    triu=True,
                )

                # get network wise average connectivity
                network_pair_conns = np.zeros(
                    (len(unique_encoded_labels), len(unique_encoded_labels))
                )
                for network_i in unique_encoded_labels:
                    index_i = np.where(encoded_labels == network_i)[0]
                    for network_j in unique_encoded_labels:
                        index_j = np.where(encoded_labels == network_j)[0]
                        matrix[np.triu_indices_from(matrix)] = np.nan
                        network_pair_conn = np.nanmean(
                            matrix[np.ix_(index_i, index_j)]
                        )
                        network_pair_conns[network_i][
                            network_j
                        ] = network_pair_conn
                # plot network wise average connectivity
                get_lower_tri_heatmap(
                    network_pair_conns,
                    title=f"{sub} {task}",
                    output=os.path.join(
                        mats_dir, f"{sub}_{task}_{cov}_{measure}_networks"
                    ),
                    labels=le.inverse_transform(unique_encoded_labels),
                    triu=True,
                    cmap="hot_r",
                )

                # plot connectome on glass brain
                f = plt.figure(figsize=(9, 4))
                plot_connectome(
                    matrix,
                    coords,
                    edge_threshold="99.8%",
                    title=f"{sub} {task}",
                    node_size=25,
                    figure=f,
                    colorbar=True,
                    output_file=os.path.join(
                        brain_dir,
                        f"{sub}_{task}_{cov}_{measure}_connectome.png",
                    ),
                )
                threshold = np.percentile(matrix, 99.8)
                matrix_thresholded = np.where(matrix > threshold, matrix, 0)
                max_ = np.max(matrix)

                # plot connectome in 3D view in html
                three_d = view_connectome(
                    matrix,
                    coords,
                    # edge_threshold="99.8%",
                    symmetric_cmap=False,
                    title=f"{sub} {task}",
                )
                three_d.save_as_html(
                    os.path.join(
                        html_dir,
                        f"{sub}_{task}_{cov}_{measure}_connectome.html",
                    )
                )
                plt.close("all")


### mean functional connectivity plots
n_parcels = 400
cache = DATA_ROOT = "/storage/store2/work/haggarwa/"
atlas = datasets.fetch_atlas_schaefer_2018(
    data_dir=cache, resolution_mm=2, n_rois=n_parcels
)
networks = atlas["labels"].astype("U")
hemi_network_labels = []
network_labels = []
rename_networks = {
    "Vis": "Visual",
    "Cont": "FrontPar",
    "SalVentAttn": "VentAttn",
}
for network in networks:
    components = network.split("_")
    components[2] = rename_networks.get(components[2], components[2])
    hemi_network = " ".join(components[1:3])
    hemi_network_labels.append(hemi_network)
    network_labels.append(components[2])
ticks = []
unique_hemi_network_labels = []
for i, label in enumerate(hemi_network_labels):
    if label != hemi_network_labels[i - 1]:
        ticks.append(i)
        unique_hemi_network_labels.append(label)

le = preprocessing.LabelEncoder()
encoded_labels = le.fit_transform(network_labels)
unique_encoded_labels = np.unique(encoded_labels)
# cov estimators
cov_estimators = ["Graphical-Lasso", "Ledoit-Wolf", "Unregularized"]
# connectivity measures for each cov estimator
measures = ["correlation", "partial correlation"]
# what to classify
classify = ["Tasks", "Subjects", "Runs"]
# tasks
tasks = ["RestingState", "Raiders", "GoodBadUgly", "MonkeyKingdom", "Mario"]
# load the data
if n_parcels == 400:
    # fc_data = pd.read_pickle(os.path.join(DATA_ROOT, "connectomes2"))
    fc_data = pd.read_pickle(
        os.path.join(DATA_ROOT, "connectomes_400_comprcorr")
    )  # with compcorr
    coords_file = "Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.Centroid_RAS.csv"
    mats_dir = os.path.join(DATA_ROOT, "mean_connectivity_matrices_compcorr")
elif n_parcels == 200:
    # fc_data = pd.read_pickle(
    #     os.path.join(DATA_ROOT, "connectomes_200_parcels")
    # )
    fc_data = pd.read_pickle(
        os.path.join(DATA_ROOT, "connectomes_200_comprcorr")
    )  # with compcorr
    coords_file = "Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.Centroid_RAS.csv"
    mats_dir = os.path.join(
        DATA_ROOT, "mean_connectivity_matrices_200_compcorr"
    )
sns.set_context("notebook", font_scale=1.05)
os.makedirs(mats_dir, exist_ok=True)
coords = pd.read_csv(
    os.path.join(
        cache,
        "schaefer_2018",
        coords_file,
    )
)[["R", "A", "S"]].to_numpy()
for cov in cov_estimators:
    for measure in measures:
        try:
            matrix = vec_to_sym_matrix(
                np.mean(
                    np.vstack(list(fc_data[f"{cov} {measure}"])),
                    axis=0,
                ),
                diagonal=np.ones(n_parcels),
            )
        except ValueError as e:
            print(e)
            print(f"{sub} {task} {cov} {measure} does not exist")
            continue
        sns.set_context("notebook")
        get_lower_tri_heatmap(
            matrix,
            title=f"{cov} {measure}",
            output=os.path.join(mats_dir, f"full_{cov}_{measure}"),
            ticks=ticks,
            labels=unique_hemi_network_labels,
            grid=True,
            diag=True,
            triu=True,
        )

        # get network wise average connectivity
        network_pair_conns = np.zeros(
            (len(unique_encoded_labels), len(unique_encoded_labels))
        )
        for network_i in unique_encoded_labels:
            index_i = np.where(encoded_labels == network_i)[0]
            for network_j in unique_encoded_labels:
                index_j = np.where(encoded_labels == network_j)[0]
                matrix[np.triu_indices_from(matrix)] = np.nan
                network_pair_conn = np.nanmean(
                    matrix[np.ix_(index_i, index_j)]
                )
                network_pair_conns[network_i][network_j] = network_pair_conn
        # plot network wise average connectivity
        get_lower_tri_heatmap(
            network_pair_conns,
            figsize=(5, 5),
            title=f"{cov} {measure}",
            output=os.path.join(mats_dir, f"networks_{cov}_{measure}"),
            labels=le.inverse_transform(unique_encoded_labels),
            triu=True,
            cmap="viridis",
        )
        plt.close("all")


### structural connectivity plots ###
# get atlas for yeo network labels
cache = DATA_ROOT = "/storage/store2/work/haggarwa/"
atlas = datasets.fetch_atlas_schaefer_2018(
    data_dir=cache, resolution_mm=2, n_rois=400
)
networks = atlas["labels"].astype("U")
hemi_network_labels = []
for network in networks:
    components = network.split("_")
    hemi_network = "_".join(components[1:3])
    hemi_network_labels.append(hemi_network)
ticks = []
for i, label in enumerate(hemi_network_labels):
    if label != hemi_network_labels[i - 1]:
        ticks.append(i)
# load the data
sc_data = pd.read_pickle(os.path.join(DATA_ROOT, "sc_data_native_new"))
_, uniq_idx = np.unique(hemi_network_labels, return_index=True)
hemi_network_labels = np.array(hemi_network_labels)[np.sort(uniq_idx)]
sns.set_context("notebook", font_scale=1.05)
mats_dir = os.path.join(DATA_ROOT, "sc_matrices")
brain_dir = os.path.join(DATA_ROOT, "sc_glass_brain")
html_dir = os.path.join(DATA_ROOT, "sc_html")
os.makedirs(mats_dir, exist_ok=True)
os.makedirs(brain_dir, exist_ok=True)
os.makedirs(html_dir, exist_ok=True)
coords = pd.read_csv(
    os.path.join(
        cache,
        "schaefer_2018",
        "Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.Centroid_RAS.csv",
    )
)[["R", "A", "S"]].to_numpy()
for sub in tqdm(np.unique(sc_data["subject"])):
    try:
        matrix = vec_to_sym_matrix(
            np.mean(
                np.vstack(
                    list(sc_data[sc_data["subject"] == sub]["connectivity"])
                ),
                axis=0,
            ),
            diagonal=np.ones(400),
        )
    except ValueError:
        print(f"{sub} does not exist")
        continue
    # plot connectome as a matrix
    get_lower_tri_heatmap(
        matrix,
        title=f"{sub}",
        output=os.path.join(mats_dir, f"{sub}"),
        ticks=ticks,
        labels=hemi_network_labels,
        grid=True,
        diag=True,
        triu=True,
    )
    f = plt.figure(figsize=(9, 4))
    # plot connectome on glass brain
    plot_connectome(
        matrix,
        coords,
        edge_threshold="99.8%",
        title=f"{sub}",
        node_size=25,
        figure=f,
        colorbar=True,
        output_file=os.path.join(
            brain_dir,
            f"{sub}_connectome.png",
        ),
    )
    threshold = np.percentile(matrix, 99.8)
    matrix_thresholded = np.where(matrix > threshold, matrix, 0)
    max_ = np.max(matrix)
    # plot connectome in 3D view in html
    three_d = view_connectome(
        matrix,
        coords,
        edge_threshold="99.8%",
        symmetric_cmap=False,
        title=f"{sub}",
    )
    three_d.save_as_html(
        os.path.join(
            html_dir,
            f"{sub}_connectome.html",
        )
    )
    plt.close("all")
