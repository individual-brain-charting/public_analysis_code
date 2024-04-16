"""This script fits classifiers to full data and
 plots the classifier coefficients"""

import os
import pandas as pd
import seaborn as sns
from nilearn import datasets
from joblib import Parallel, delayed
from ibc_public.connectivity.utils_plot import (
    fit_classifier,
    get_clas_cov_measure,
    plot_full_weight_matrix,
    plot_network_weight_matrix,
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
