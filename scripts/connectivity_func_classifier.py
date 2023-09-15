"""Pipeline to perform functional connectivity classification over runs, subjects and tasks."""

import os
import time

import pandas as pd
import seaborn as sns
from ibc_public import utils_connectivity as fc
from nilearn import datasets

from joblib import Parallel, delayed

sns.set_theme(context="talk", style="whitegrid")

# number of jobs to run in parallel
n_jobs = 15
# cache and root output directory
cache = DATA_ROOT = "/storage/store/work/haggarwa/"
output_dir = f"fc_classification_{time.strftime('%Y%m%d-%H%M%S')}"
output_dir = os.path.join(DATA_ROOT, output_dir)
os.makedirs(output_dir, exist_ok=True)

# connectivity calculation parameters
calculate_connectivity = True
fc_data_path = os.path.join(cache, "connectomes")

# cross-validation splits
cv_splits = 50
# we will use the resting state and all the movie-watching sessions
tasks = ["RestingState", "Raiders", "GoodBadUgly", "MonkeyKingdom", "Mario"]
# cov estimators
cov_estimators = ["GLC", "LedoitWolf"]
# connectivity measures for each cov estimator
measures = ["correlation", "partial correlation"]
# connectivity measures
connectivity_measures = []
for cov in cov_estimators:
    for measure in measures:
        connectivity_measures.append(cov + " " + measure)
# what to classify
classify = ["Runs", "Subjects", "Tasks"]


# generator to get all the combinations of classification
# for later running all cases in parallel
def all_combinations(classify, tasks, connectivity_measures):
    # dictionary to map the classification to the tasks
    # when classifying by runs or subjects, we classify runs or subjects within each task
    # when classifying by tasks, we classify between two tasks
    # in this case, RestingState vs. each movie-watching task
    tasks_ = {
        "Runs": tasks,
        "Subjects": tasks,
        "Tasks": [
            ["RestingState", "Raiders"],
            ["RestingState", "GoodBadUgly"],
            ["RestingState", "MonkeyKingdom"],
            ["RestingState", "Mario"],
            ["Raiders", "GoodBadUgly"],
            ["Raiders", "MonkeyKingdom"],
            ["GoodBadUgly", "MonkeyKingdom"],
            ["Raiders", "Mario"],
            ["GoodBadUgly", "Mario"],
            ["MonkeyKingdom", "Mario"],
        ],
    }
    for classes in classify:
        for task in tasks_[classes]:
            for connectivity_measure in connectivity_measures:
                yield classes, task, connectivity_measure


if calculate_connectivity == True:
    # get the atlas
    atlas = datasets.fetch_atlas_schaefer_2018(
        data_dir=cache, resolution_mm=2, n_rois=400
    )
    # use the atlas to extract time series for each task in parallel
    # get_time_series returns a dataframe with the time series for each task, consisting of runs x subjects
    print("Time series extraction...")
    data = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(fc.get_time_series)(task, atlas, cache) for task in tasks
    )
    # concatenate all the dataframes so we have a single dataframe with the time series from all tasks
    data = pd.concat(data)
    # estimate the connectivity matrices for each cov estimator in parallel
    # get_connectomes returns a dataframe with two columns each corresponding to the partial correlation and correlation connectome from each cov estimator
    print("Connectivity estimation...")
    data = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(fc.get_connectomes)(cov, data, n_jobs)
        for cov in cov_estimators
    )
    # concatenate the dataframes so we have a single dataframe with the connectomes from all cov estimators
    cols_to_use = data[0].columns.difference(data[1].columns)
    data = pd.concat([data[1], data[0][cols_to_use]], axis=1)
    data.reset_index(inplace=True, drop=True)
else:
    data = pd.read_pickle(fc_data_path)

# run classification for all combinations of classification, task and connectivity measure in parallel
# do_cross_validation returns a dataframe with the results of the cross validation for each case
print("Cross validation...")
all_results = Parallel(n_jobs=n_jobs, verbose=2, backend="loky")(
    delayed(fc.do_cross_validation)(
        classes, task, cv_splits, connectivity_measure, data, output_dir
    )
    for classes, task, connectivity_measure in all_combinations(
        classify, tasks, connectivity_measures
    )
)
print("Saving results...")
all_results = pd.concat(all_results)
# calculate chance level
all_results = fc.chance_level(all_results)
# save the results
all_results.to_csv(os.path.join(output_dir, "all_results.csv"))

print("Plotting results...")
fc.do_plots(all_results, output_dir)
print("Done!")
