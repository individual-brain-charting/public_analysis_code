"""Pipeline to perform functional connectivity classification over runs,
 subjects and tasks for the HCP dataset

Not used anywhere but kept for future use if needed. 
"""

import os
import time

import pandas as pd
import seaborn as sns
from ibc_public import utils_connectivity as fc
from nilearn import datasets
from tqdm import tqdm
from joblib import Parallel, delayed, dump
from sklearn.svm import LinearSVC
import numpy as np

sns.set_theme(context="talk", style="whitegrid")

# number of jobs to run in parallel
n_jobs = 30
# cache and root output directory
cache = DATA_ROOT = "/storage/store2/work/haggarwa/"
output_dir = f"fc_hcp_{time.strftime('%Y%m%d-%H%M%S')}"
output_dir = os.path.join(DATA_ROOT, output_dir)
os.makedirs(output_dir, exist_ok=True)

# connectivity calculation parameters
calculate_connectivity = True
fc_data_path = os.path.join(output_dir, "connectomes_hcp")

# CLASSIFICATION PARAMETERS
do_classify = False
# cross-validation splits
cv_splits = 50

# TRAIN CLASSIFIERS
train_classifiers = True

# we will use the resting state and all the task sessions
tasks = [
    "REST",
    "EMOTION",
    "GAMBLING",
    "LANGUAGE",
    "MOTOR",
    "RELATIONAL",
    "SOCIAL",
    "WM",
]
# cov estimators
cov_estimators = ["Graphical-Lasso", "Ledoit-Wolf", "Unregularized"]
# connectivity measures for each cov estimator
measures = ["correlation", "partial correlation"]
# connectivity measures
connectivity_measures = []
for cov in cov_estimators:
    for measure in measures:
        connectivity_measures.append(cov + " " + measure)

if calculate_connectivity:
    # get the atlas
    atlas = datasets.fetch_atlas_schaefer_2018(
        data_dir=cache, resolution_mm=2, n_rois=400
    )
    # use the atlas to extract time series for each task in parallel
    # get_time_series returns a dataframe with the time series for each task, consisting of runs x subjects
    print("Time series extraction...")
    data = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(fc.get_time_series)(task, atlas, cache, dataset="hcp")
        for task in tasks
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
    common_cols = ["time_series", "subject_ids", "run_labels", "tasks"]
    data_ts = data[0][common_cols]
    for df in data:
        df.drop(columns=common_cols, inplace=True)
    data.append(data_ts)
    data = pd.concat(data, axis=1)
    data.reset_index(inplace=True, drop=True)
    # save the data
    data.to_pickle(fc_data_path)
else:
    data = pd.read_pickle(fc_data_path)

# run classification for all combinations of classification, task and connectivity measure in parallel
# do_cross_validation returns a dataframe with the results of the cross validation for each case
if do_classify:
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
            "Runs": [tasks],
            "Subjects": [tasks],
            "Tasks": [tasks],
        }
        for classes in classify:
            for task in tasks_[classes]:
                for connectivity_measure in connectivity_measures:
                    yield classes, task, connectivity_measure

    print("Cross validation...")
    all_results = Parallel(n_jobs=n_jobs, verbose=11, backend="loky")(
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
    all_results.to_pickle(os.path.join(output_dir, "all_results.pkl"))

### fit classifiers to get weights ###
if train_classifiers:
    trained_dir = os.path.join(output_dir, "hcp_trained")
    os.makedirs(trained_dir, exist_ok=True)
    data = pd.read_pickle(fc_data_path)
    classify = ["Tasks", "Subjects"]
    for clas in tqdm(classify, desc="Classify", position=0):
        if clas == "Tasks":
            group_name = "subject_ids"
            groups = np.unique(data[group_name].to_numpy(dtype=object))
        elif clas == "Subjects":
            group_name = "tasks"
            groups = np.unique(data[group_name].to_numpy(dtype=object))
        for group in groups:
            data_group = data[data[group_name] == group]
            for cov in tqdm(cov_estimators, desc="Estimator", position=1):
                for measure in measures:
                    classifier_file = os.path.join(
                        trained_dir, f"{clas}_{cov}_{measure}_{group}_clf.pkl"
                    )
                    if os.path.exists(classifier_file):
                        print(f"skipping {cov} {measure}, already done")
                        continue
                    else:
                        if clas == "Tasks":
                            classes = data_group["tasks"].to_numpy(
                                dtype=object
                            )
                        elif clas == "Subjects":
                            classes = data_group["subject_ids"].to_numpy(
                                dtype=object
                            )
                        data_group = np.array(
                            data_group[f"{cov} {measure}"].values.tolist()
                        )
                        classifier = LinearSVC(
                            max_iter=100000, dual="auto"
                        ).fit(data_group, classes)
                        dump(classifier, classifier_file)

# print("Plotting results...")
# fc.do_plots(all_results, output_dir)
# print("Done!")
