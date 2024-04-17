"""This script fits classifiers to full data and
 plots the classifier coefficients"""

import os
import pandas as pd
import seaborn as sns
from nilearn import datasets
from joblib import Parallel, delayed
from sklearn import preprocessing
import numpy as np
from scipy.stats import mannwhitneyu
from ibc_public.connectivity.utils_plot import (
    fit_classifier,
    get_clas_cov_measure,
    plot_full_weight_matrix,
    plot_network_weight_matrix,
    get_network_labels,
    _average_over_networks,
)

sns.set_theme()
sns.set_style("whitegrid")
sns.set_context("talk")

### fit classifiers to get weights
DATA_ROOT = cache = "/storage/store2/work/haggarwa/"
n_parcels = 400
if n_parcels == 400:
    func_data_path = os.path.join(cache, "connectomes_400_comprcorr")
    output_dir = os.path.join(DATA_ROOT, "weights_compcorr")
elif n_parcels == 200:
    func_data_path = os.path.join(cache, "connectomes_200_comprcorr")
    output_dir = os.path.join(DATA_ROOT, "weights_200_compcorr")
os.makedirs(output_dir, exist_ok=True)
func_data = pd.read_pickle(func_data_path)
cov_estimators = ["Graphical-Lasso", "Ledoit-Wolf", "Unregularized"]
measures = ["correlation", "partial correlation"]
classify = ["Tasks", "Subjects", "Runs"]
x = Parallel(n_jobs=20, verbose=11)(
    delayed(fit_classifier)(
        clas, cov, measure, func_data, output_dir=output_dir
    )
    for clas, cov, measure in get_clas_cov_measure(
        classify, cov_estimators, measures
    )
)

### network pair SVC weight matrices
DATA_ROOT = cache = "/storage/store2/work/haggarwa/"
n_parcels = 400
if n_parcels == 400:
    weight_dir = os.path.join(DATA_ROOT, "weights_compcorr")
    output_dir = os.path.join(DATA_ROOT, "weight_plots_compcorr")
elif n_parcels == 200:
    weight_dir = os.path.join(DATA_ROOT, "weights_200_compcorr")
    output_dir = os.path.join(DATA_ROOT, "weight_plots_200_compcorr")
os.makedirs(output_dir, exist_ok=True)
# get atlas for yeo network labels
atlas = datasets.fetch_atlas_schaefer_2018(
    data_dir=cache, resolution_mm=2, n_rois=n_parcels
)
# cov estimators
cov_estimators = ["Graphical-Lasso", "Ledoit-Wolf", "Unregularized"]
# connectivity measures for each cov estimator
measures = ["correlation", "partial correlation"]
# what to classify
classify = ["Tasks", "Subjects", "Runs"]

x = Parallel(n_jobs=20, verbose=11)(
    delayed(plot_full_weight_matrix)(
        clas,
        cov,
        measure,
        atlas,
        transform="l2",
        output_dir=output_dir,
        fontsize=15,
        weight_dir=weight_dir,
        n_parcels=n_parcels,
    )
    for clas, cov, measure in get_clas_cov_measure(
        classify, cov_estimators, measures
    )
)
x = Parallel(n_jobs=20, verbose=11)(
    delayed(plot_network_weight_matrix)(
        clas,
        cov,
        measure,
        atlas,
        labels_fmt="network",
        transform="l2",
        output_dir=output_dir,
        fontsize=15,
        weight_dir=weight_dir,
        n_parcels=n_parcels,
    )
    for clas, cov, measure in get_clas_cov_measure(
        classify, cov_estimators, measures
    )
)

### compare within network and between network weights
DATA_ROOT = cache = "/storage/store2/work/haggarwa/"
n_parcels = 400
if n_parcels == 400:
    weight_dir = os.path.join(DATA_ROOT, "weights_compcorr")
elif n_parcels == 200:
    weight_dir = os.path.join(DATA_ROOT, "weights_200_compcorr")
# get atlas for yeo network labels
atlas = datasets.fetch_atlas_schaefer_2018(
    data_dir=cache, resolution_mm=2, n_rois=n_parcels
)
# cov estimators
cov_estimators = ["Graphical-Lasso", "Ledoit-Wolf", "Unregularized"]
# connectivity measures for each cov estimator
measures = ["correlation", "partial correlation"]
# what to classify
classify = ["Tasks", "Subjects", "Runs"]

labels = get_network_labels(atlas)[1]


le = preprocessing.LabelEncoder()
encoded_labels = le.fit_transform(labels)
transform = "l2"
for clas, cov, measure in get_clas_cov_measure(
    classify, cov_estimators, measures
):
    unique_labels = np.unique(encoded_labels)
    network_weights = _average_over_networks(
        encoded_labels,
        unique_labels,
        clas,
        cov,
        measure,
        transform,
        weight_dir,
        n_parcels,
    )

    within_network_weights = np.diag(network_weights)
    # keep only lower triangle
    between_network_weights = network_weights[
        np.tril_indices_from(network_weights, k=-1)
    ]
    _, within_greater = mannwhitneyu(
        within_network_weights, between_network_weights, alternative="greater"
    )
    _, between_greater = mannwhitneyu(
        between_network_weights, within_network_weights, alternative="greater"
    )
    _, two_sided = mannwhitneyu(
        within_network_weights, between_network_weights
    )

    print(
        "*** Network weights ***\n",
        f"Classifying {clas} with {cov} {measure}:\n Within network > ",
        f"between network: {within_greater:.2e}\n Between network > ",
        f"within network: {between_greater:.2e}\n",
        f"Two-sided: {two_sided:.2e}\n",
    )
