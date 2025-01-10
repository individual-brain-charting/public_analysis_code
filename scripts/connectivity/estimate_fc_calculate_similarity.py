"""This script estimates functional connectivity (if needed) and then
 calculates the similarity between functional connectivity
 matrices from different tasks and structural connectivity"""

import os
import time
import pandas as pd
from nilearn import datasets
from joblib import Parallel, delayed
from ibc_public.connectivity.utils_similarity import (
    mean_connectivity,
    get_similarity,
)
from ibc_public.connectivity.utils_fc_estimation import (
    get_connectomes,
    get_time_series,
)

cache = DATA_ROOT = "/storage/store2/work/haggarwa/"
output_dir = f"fc_similarity_{time.strftime('%Y%m%d-%H%M%S')}"
output_dir = os.path.join(DATA_ROOT, output_dir)
os.makedirs(output_dir, exist_ok=True)
calculate_connectivity = False
n_parcels = 400
if n_parcels == 400:
    fc_data_path = os.path.join(cache, "connectomes_400_comprcorr")
    sc_data_path = os.path.join(cache, "sc_data_native_new")
elif n_parcels == 200:
    fc_data_path = os.path.join(cache, "connectomes_200_comprcorr")
    sc_data_path = os.path.join(cache, "sc_data_native_200")
# number of jobs to run in parallel
n_jobs = 50
# tasks
tasks = [
    "RestingState",
    "Raiders",
    "GoodBadUgly",
    "MonkeyKingdom",
    "Mario",
]
# cov estimators
cov_estimators = ["Graphical-Lasso", "Ledoit-Wolf", "Unregularized"]
# connectivity measures for each cov estimator
measures = ["correlation", "partial correlation"]

task_pairs = [
    ("RestingState", "Raiders"),
    ("RestingState", "GoodBadUgly"),
    ("RestingState", "MonkeyKingdom"),
    ("RestingState", "Mario"),
    ("Raiders", "GoodBadUgly"),
    ("Raiders", "MonkeyKingdom"),
    ("GoodBadUgly", "MonkeyKingdom"),
    ("Raiders", "Mario"),
    ("GoodBadUgly", "Mario"),
    ("MonkeyKingdom", "Mario"),
    ("RestingState", "SC"),
    ("Raiders", "SC"),
    ("GoodBadUgly", "SC"),
    ("MonkeyKingdom", "SC"),
    ("Mario", "SC"),
]


def all_combinations(task_pairs, cov_estimators, measures):
    """generator to yield all combinations of task pairs, cov estimators, to
    parallelize the similarity calculation for each combination"""
    for task_pair in task_pairs:
        for cov in cov_estimators:
            for measure in measures:
                yield task_pair, cov, measure


if calculate_connectivity:
    # get the atlas
    atlas = datasets.fetch_atlas_schaefer_2018(
        data_dir=cache, resolution_mm=2, n_rois=n_parcels
    )
    # use the atlas to extract time series for each task in parallel
    # get_time_series returns a dataframe with
    # the time series for each task, consisting of runs x subjects
    print("Time series extraction...")
    data = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(get_time_series)(task, atlas, cache) for task in tasks
    )
    # concatenate all the dataframes so we have a single dataframe
    # with the time series from all tasks
    data = pd.concat(data)
    # estimate the connectivity matrices for each cov estimator in parallel
    # get_connectomes returns a dataframe with two columns each corresponding
    # to the partial correlation and correlation connectome from each cov
    # estimator
    print("Connectivity estimation...")
    data = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(get_connectomes)(cov, data, n_jobs) for cov in cov_estimators
    )
    # concatenate the dataframes so we have a single dataframe
    # with the connectomes from all cov estimators
    cols_to_use = data[0].columns.difference(data[1].columns)
    data = pd.concat([data[1], data[0][cols_to_use]], axis=1)
    data.reset_index(inplace=True, drop=True)
else:
    data = pd.read_pickle(fc_data_path)
    sc_data = pd.read_pickle(sc_data_path)
all_connectivity = mean_connectivity(data, tasks, cov_estimators, measures)
all_connectivity = pd.concat([all_connectivity, sc_data], axis=0)

results = Parallel(n_jobs=n_jobs, verbose=2, backend="loky")(
    delayed(get_similarity)(all_connectivity, task_pair, cov, measure)
    for task_pair, cov, measure in all_combinations(
        task_pairs, cov_estimators, measures
    )
)

results = [item for sublist in results for item in sublist]
results = pd.DataFrame(results)
results.to_pickle(os.path.join(output_dir, "results.pkl"))
