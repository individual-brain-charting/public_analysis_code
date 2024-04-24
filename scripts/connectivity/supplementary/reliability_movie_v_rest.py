from ibc_public.utils_connectivity import (
    _get_tr,
    get_ses_modality,
    _get_confounds,
    _update_data,
    get_connectomes,
)
from ibc_public.utils_data import DERIVATIVES
from glob import glob
import os
from nilearn.maskers import NiftiLabelsMasker
from nilearn.image import high_variance_confounds
from nilearn import datasets
import itertools
import pandas as pd
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.stats import mannwhitneyu

HCP_ROOT = "/storage/store/data/HCP900"


def _select_runs(task, run_files):
    print("all runs", run_files)
    run_labels = [os.path.basename(run).split("_")[-2] for run in run_files]
    run_nums_files = [
        (run_label.split("-")[-1], run)
        for run_label, run in zip(run_labels, run_files)
    ]
    run_files = []
    for run_num, run_file in run_nums_files:
        if task == "Raiders":
            if run_num in ["11", "12", "13"]:
                run_files.append(run_file)
        elif task == "GoodBadUgly":
            if run_num in ["19", "20", "21"]:
                run_files.append(run_file)
        else:
            run_files.append(run_file)
    print("selected runs", run_files)
    return run_files


def _get_niftis(task, subject, session, dataset="ibc"):
    if dataset == "ibc":
        _run_files = glob(
            os.path.join(
                DERIVATIVES,
                subject,
                session,
                "func",
                f"wrdc*{task}*.nii.gz",
            )
        )
        _run_files = _select_runs(task, _run_files)
        run_labels = []
        run_files = []
        for run in _run_files:
            run_label = os.path.basename(run).split("_")[-2]
            run_labels.append(run_label)
            run_files.append(run)
    elif dataset == "hcp":
        run_files = glob(
            os.path.join(
                HCP_ROOT,
                subject,
                "MNINonLinear",
                "Results",
                session,
                f"{session}.nii.gz",
            )
        )
        run_labels = []
        for run in run_files:
            direction = session.split("_")[2]
            if task == "REST":
                rest_ses = session.split("_")[1]
                if direction == "LR" and rest_ses == "REST1":
                    run_label = "run-01"
                elif direction == "RL" and rest_ses == "REST1":
                    run_label = "run-02"
                elif direction == "LR" and rest_ses == "REST2":
                    run_label = "run-03"
                elif direction == "RL" and rest_ses == "REST2":
                    run_label = "run-04"
            else:
                if direction == "LR":
                    run_label = "run-01"
                elif direction == "RL":
                    run_label = "run-02"
            run_labels.append(run_label)

    return run_files, run_labels


def get_time_series(task, atlas, cache, dataset="ibc"):
    """Use NiftiLabelsMasker to extract time series from nifti files.

    Parameters
    ----------
    tasks : list
        List of tasks to extract time series from.
    atlas : atlas object
        Atlas to use for extracting time series.
    cache : str
        Path to cache directory.

    Returns
    -------
    pandas DataFrame
        DataFrame containing the time series, subject ids, run labels, and
        tasks.
    """
    data = {
        "time_series": [],
        "subject_ids": [],
        "run_labels": [],
        "tasks": [],
    }
    repetition_time = _get_tr(task, dataset)
    print(f"Getting time series for {task}...")
    masker = NiftiLabelsMasker(
        labels_img=atlas.maps,
        standardize=True,
        low_pass=0.2,
        high_pass=0.01,
        t_r=repetition_time,
        verbose=0,
        # memory=Memory(location=cache),
        memory_level=0,
        n_jobs=1,
    ).fit()
    subject_sessions, _ = get_ses_modality(task, dataset)
    if dataset == "hcp":
        subject_sessions = dict(itertools.islice(subject_sessions.items(), 50))
    all_time_series = []
    subject_ids = []
    run_labels_ = []
    for subject, sessions in tqdm(
        subject_sessions.items(), desc=task, total=len(subject_sessions)
    ):
        for session in sorted(sessions):
            runs, run_labels = _get_niftis(task, subject, session, dataset)
            for run, run_label in zip(runs, run_labels):
                confounds = _get_confounds(
                    task, run_label, subject, session, dataset
                )
                compcor_confounds = high_variance_confounds(run)
                confounds = np.hstack(
                    (np.loadtxt(confounds), compcor_confounds)
                )
                time_series = masker.transform(run, confounds=confounds)
                all_time_series.append(time_series)
                subject_ids.append(subject)
                run_labels_.append(run_label)

    tasks_ = [task for _ in range(len(all_time_series))]

    data = _update_data(
        data, all_time_series, subject_ids, run_labels_, tasks_
    )
    return pd.DataFrame(data)


if __name__ == "__main__":
    #### INPUTS
    # number of jobs to run in parallel
    n_jobs = 10
    # number of parcels
    n_parcels = 200  # or 400
    # number of splits for cross validation
    n_splits = 50
    # do within each task or across all tasks
    within_task = True
    # connectivity calculation parameters
    calculate_connectivity = False
    # we will use the resting state and all the movie-watching sessions
    tasks = [
        "Raiders",
        "GoodBadUgly",
    ]
    # cov estimators
    cov_estimators = ["Graphical-Lasso", "Ledoit-Wolf", "Unregularized"]
    # connectivity measures for each cov estimator
    measures = ["correlation", "partial correlation"]
    # what to classify
    classify = ["Runs", "Subjects", "Tasks"]

    #### SETUP
    # cache and root output directory
    cache = DATA_ROOT = "/storage/store2/work/haggarwa/"
    output_dir = f"reliability_{n_parcels}"
    output_dir = os.path.join(DATA_ROOT, output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # connectivity data path
    if n_parcels == 400:
        # without compcorr
        # fc_data_path = os.path.join(cache, "connectomes2")
        # with compcorr
        first_three_data_path = os.path.join(
            cache, "connectomes_400_comprcorr"
        )
    elif n_parcels == 200:
        # without compcorr
        # fc_data_path = os.path.join(cache, "connectomes_200_parcels")
        # with compcorr
        first_three_data_path = os.path.join(
            cache, "connectomes_200_comprcorr"
        )

    ### CALCULATE CONNECTIVITY IF NOT ALREADY CALCULATED
    if calculate_connectivity:
        # get the atlas
        atlas = datasets.fetch_atlas_schaefer_2018(
            data_dir=cache, resolution_mm=2, n_rois=n_parcels
        )
        # use the atlas to extract time series for each task in parallel
        # get_time_series returns a dataframe with the time series for
        # each task, consisting of runs x subjects
        print("Time series extraction...")
        data = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(get_time_series)(task, atlas, cache) for task in tasks
        )
        # concatenate all the dataframes so we have a single dataframe with the
        # time series from all tasks
        data = pd.concat(data)
        # estimate the connectivity matrices for each cov estimator in parallel
        # get_connectomes returns a dataframe with two columns each
        # corresponding to the partial correlation and correlation connectome
        # from each cov estimator
        print("Connectivity estimation...")
        data = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(get_connectomes)(cov, data, n_jobs)
            for cov in cov_estimators
        )
        # concatenate the dataframes so we have a single dataframe with the
        # connectomes from all cov estimators
        common_cols = ["time_series", "subject_ids", "run_labels", "tasks"]
        data_ts = data[0][common_cols]
        for df in data:
            df.drop(columns=common_cols, inplace=True)
        data.append(data_ts)
        data = pd.concat(data, axis=1)
        data.reset_index(inplace=True, drop=True)
        # save the data
        data.to_pickle(
            os.path.join(output_dir, f"connectomes_{n_parcels}_comprcorr")
        )
    #### LOAD CONNECTIVITY IF ALREADY CALCULATED
    else:
        data = pd.read_pickle(
            os.path.join(output_dir, f"connectomes_{n_parcels}_comprcorr")
        )
    # relabel runs to have the same labels as the first three runs (that they
    # are repetitions of)
    data["run_labels"] = data["run_labels"].replace(
        {
            "run-11": "run-01",
            "run-12": "run-02",
            "run-13": "run-03",
            "run-19": "run-01",
            "run-20": "run-02",
            "run-21": "run-03",
        },
    )

    ### load precomputed connectomes
    first_three_runs = pd.read_pickle(first_three_data_path)
    # select only Raiders, GoodBadUgly
    first_three_runs_movies = first_three_runs[
        first_three_runs["tasks"].isin(["Raiders", "GoodBadUgly"])
    ]
    # select first three runs of Raiders and GoodBadUgly
    first_three_runs_movies = first_three_runs_movies[
        first_three_runs_movies["run_labels"].isin(
            ["run-01", "run-02", "run-03"]
        )
    ]
    # select all RestingState runs
    rest_runs = first_three_runs[
        first_three_runs["tasks"].isin(["RestingState"])
    ]
    # concatenate the first three runs of Raiders and GoodBadUgly with all
    # RestingState runs
    first_three_runs = pd.concat([first_three_runs_movies, rest_runs])

    # concatenate with the last three runs
    all_runs = pd.concat([first_three_runs, data])

    # runs labels according to task
    task_run_labels = {
        "Raiders": ["run-01", "run-02", "run-03"],
        "GoodBadUgly": ["run-01", "run-02", "run-03"],
        "RestingState": ["dir-ap", "dir-pa"],
    }

    # concatenate estimator and measure names
    connectivity_measures = []
    for cov in cov_estimators:
        for measure in measures:
            connectivity_measures.append(cov + " " + measure)

    connectivity_measures.append("time_series")

    corrs = []
    for task in all_runs["tasks"].unique():
        for subject in all_runs["subject_ids"].unique():
            for run in task_run_labels[task]:
                for measure in connectivity_measures:
                    reps = all_runs[
                        (all_runs["subject_ids"] == subject)
                        & (all_runs["run_labels"] == run)
                        & (all_runs["tasks"] == task)
                    ][measure].reset_index(drop=True)
                    print(reps)
                    try:
                        rep_1 = reps[0]
                        rep_2 = reps[1]
                    except KeyError:
                        print(
                            f"\n\n*** {run} not found for {subject} in {task}"
                            " ***\n\n"
                        )
                        continue
                    # keep only rep1-rep2 correlation
                    if measure == "time_series":
                        try:
                            corr = np.corrcoef(rep_1.T, rep_2.T)
                            corr = corr[:n_parcels, n_parcels:]
                            corr = np.mean(np.diag(corr))
                        except ValueError:
                            print(
                                f"\n\n*** Different {run} lengths for {subject}"
                                f" in {task} ***\n\n"
                            )
                            corr = np.nan
                    else:
                        rep_1 = np.array(rep_1)
                        rep_2 = np.array(rep_2)
                        # rep_1 = np.diag(vec_to_sym_matrix(rep_1))
                        # rep_2 = np.diag(vec_to_sym_matrix(rep_2))
                        corr = np.corrcoef(rep_1, rep_2)
                        corr = corr[0, 1]
                    row = {
                        "subject_id": subject,
                        "run_label": run,
                        "task": task,
                        "correlation": corr,
                        "measure": measure,
                    }
                    corrs.append(row)

    corrs = pd.DataFrame(corrs)
    corrs.to_pickle(os.path.join(output_dir, f"corrs_full_mat_{n_parcels}"))

    p_vals = []

    for measure in connectivity_measures:
        corrs_measure = corrs[corrs["measure"] == measure]
        rest = corrs_measure[corrs_measure["task"] == "RestingState"]
        raiders = corrs_measure[corrs_measure["task"] == "Raiders"]
        gbu = corrs_measure[corrs_measure["task"] == "GoodBadUgly"]
        rest_v_raiders = mannwhitneyu(
            rest["correlation"], raiders["correlation"], nan_policy="omit"
        )
        rest_v_gbu = mannwhitneyu(
            rest["correlation"], gbu["correlation"], nan_policy="omit"
        )
        p_vals.append(
            {
                "comp": "RestingState vs Raiders",
                "p_val": rest_v_raiders[1],
                "measure": measure,
            }
        )
        p_vals.append(
            {
                "comp": "RestingState vs GoodBadUgly",
                "p_val": rest_v_gbu[1],
                "measure": measure,
            }
        )

    p_vals = pd.DataFrame(p_vals)
    p_vals.to_pickle(os.path.join(output_dir, f"p_vals_{n_parcels}"))
