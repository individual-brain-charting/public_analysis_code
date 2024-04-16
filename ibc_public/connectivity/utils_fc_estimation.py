"""Utility functions for functional connectivity estimation"""

import os
from glob import glob
import numpy as np
import pandas as pd
from ibc_public.utils_data import DERIVATIVES, get_subject_session
from nilearn.connectome import sym_matrix_to_vec
from nilearn.image import high_variance_confounds
from nilearn.maskers import NiftiLabelsMasker
from sklearn.base import clone
from sklearn.covariance import (
    GraphicalLassoCV,
    GraphicalLasso,
    LedoitWolf,
    EmpiricalCovariance,
    empirical_covariance,
    shrunk_covariance,
)
from tqdm import tqdm
import itertools

HCP_ROOT = "/storage/store/data/HCP900"


def _get_tr(task, dataset="ibc"):
    """Get repetition time for the given task and dataset

    Parameters
    ----------
    task : str
        Name of the task. Could be "RestingState", "GoodBadUgly", "Raiders",
        "MonkeyKingdom", "Mario" if dataset is "ibc". If dataset is "hcp",
         all tasks have a repetition time of 0.72.
    dataset : str, optional
        Which dataset to use, by default "ibc", could also be "hcp"

    Returns
    -------
    float or int
        Repetition time for the given task

    Raises
    ------
    ValueError
        If the task is not recognized
    """
    if dataset == "ibc":
        if task == "RestingState":
            repetition_time = 0.76
        elif task in ["GoodBadUgly", "Raiders", "MonkeyKingdom", "Mario"]:
            repetition_time = 2
        else:
            raise ValueError(f"Unknown task {task}")
    elif dataset == "hcp":
        repetition_time = 0.72

    return repetition_time


def _get_niftis(task, subject, session, dataset="ibc"):
    """Get nifti files of preprocessed BOLD data for the given task, subject,
    session and dataset

    Parameters
    ----------
    task : str
        Name of the task
    subject : str
        subject id
    session : str
        session number
    dataset : str, optional
        which dataset to use, by default "ibc", could also be "hcp"

    Returns
    -------
    list, list
        List of paths to nifti files, list of run labels for each nifti file
        eg. "run-01", "run-02", etc.
    """
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
        run_labels = []
        run_files = []
        for run in _run_files:
            run_label = os.path.basename(run).split("_")[-2]
            run_num = run_label.split("-")[-1]
            # skip repeats of run-01, run-02, run-03 done at the end of
            # the sessions in Raiders and GoodBadUgly
            if task == "Raiders" and int(run_num) > 10:
                continue
            elif task == "GoodBadUgly" and int(run_num) > 18:
                continue
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


def _get_confounds(task, run_label, subject, session, dataset="ibc"):
    """Get confounds file for the given task, run number, subject, session
    and dataset.

    Parameters
    ----------
    task : str
        Name of the task
    run_label : str
        Run label of the nifti file
    subject : str
        subject id
    session : str
        session number
    dataset : str, optional
        name of the dataset, by default "ibc". Could also be "hcp"

    Returns
    -------
    str
        Path to the confounds file
    """
    if dataset == "ibc":
        return glob(
            os.path.join(
                DERIVATIVES,
                subject,
                session,
                "func",
                f"rp*{task}*{run_label}_bold*",
            )
        )[0]
    elif dataset == "hcp":
        return glob(
            os.path.join(
                HCP_ROOT,
                subject,
                "MNINonLinear",
                "Results",
                session,
                "Movement_Regressors_dt.txt",
            )
        )[0]


def _update_data(data, all_time_series, subject_ids, run_labels, tasks):
    """Update the data dictionary with the new time series, subject ids,
    run labels"""
    data["time_series"].extend(all_time_series)
    data["subject_ids"].extend(subject_ids)
    data["run_labels"].extend(run_labels)
    data["tasks"].extend(tasks)

    return data


def _find_hcp_subjects(session_names):
    """Find HCP subjects with the given session names"""
    # load csv file with subject ids and task availability
    df = pd.read_csv(os.path.join(HCP_ROOT, "unrestricted_hcp_s900.csv"))
    df = df[df["3T_Full_MR_Compl"] == "True"]
    subs = list(df["Subject"].astype(str))
    sub_ses = {}
    for sub in subs:
        sub_ses[sub] = session_names

    return sub_ses


def get_ses_modality(task, dataset="ibc"):
    """Get session numbers and modality for given task

    Parameters
    ----------
    task : str
        name of the task
    dataset : str
        name of the dataset, can be ibc or hcp

    Returns
    -------
    sub_ses : dict
        dictionary with subject as key and session number as value
    modality : str
        modality of the task
    """
    if dataset == "ibc":
        if task == "GoodBadUgly":
            # session names with movie task data
            session_names = ["BBT1", "BBT2", "BBT3"]
        elif task == "MonkeyKingdom":
            # session names with movie task data
            session_names = ["monkey_kingdom"]
        elif task == "Raiders":
            # session names with movie task data
            session_names = ["raiders1", "raiders2"]
        elif task == "RestingState":
            # session names with RestingState state task data
            session_names = ["mtt1", "mtt2"]
        elif task == "DWI":
            # session names with diffusion data
            session_names = ["anat1"]
        elif task == "Mario":
            # session names with mario gameplay data
            session_names = ["mario1"]
        # get session numbers for each subject
        # returns a list of tuples with subject and session number
        subject_sessions = sorted(get_subject_session(session_names))
        # convert the tuples to a dictionary with subject as key and session
        # number as value
        sub_ses = {}
        # for dwi, with anat1 as session_name, get_subject_session returns
        # wrong session number for sub-01 and sub-15
        # setting it to ses-12 for these subjects
        if task == "DWI":
            modality = "structural"
            sub_ses = {
                subject_session[0]: "ses-12"
                if subject_session[0] in ["sub-01", "sub-15"]
                else subject_session[1]
                for subject_session in subject_sessions
            }
        else:
            # for fMRI tasks, for one of the movies, ses no. 13 pops up for
            # sub-11 and sub-12, so skipping that
            modality = "functional"
            for subject_session in subject_sessions:
                if (
                    subject_session[0] in ["sub-11", "sub-12"]
                    and subject_session[1] == "ses-13"
                ):
                    continue
                # initialize a subject as key and an empty list as the value
                # and populate the list with session numbers
                # try-except block is used to avoid overwriting the list
                # for subject
                try:
                    sub_ses[subject_session[0]]
                except KeyError:
                    sub_ses[subject_session[0]] = []
                sub_ses[subject_session[0]].append(subject_session[1])

    elif dataset == "hcp":
        if task == "REST":
            # session names with RestingState state task data
            session_names = ["rfMRI_REST1_LR", "rfMRI_REST2_RL"]
        else:
            # session names with diffusion data
            session_names = [f"tfMRI_{task}_LR", f"tfMRI_{task}_RL"]
        modality = "functional"

        # create dictionary with subject as key and session number as value
        sub_ses = _find_hcp_subjects(session_names)

    return sub_ses, modality


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
    dataset : str, optional
        Name of the dataset, by default "ibc". Could also be "hcp".

    Returns
    -------
    pandas DataFrame
        DataFrame containing the time series, subject ids, run labels,
        and tasks.
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
                print(task)
                print(subject, session)
                print(
                    glob(
                        os.path.join(
                            DERIVATIVES,
                            subject,
                            session,
                            "func",
                            f"rp*{task}*{run_label}_bold*",
                        )
                    )
                )
                print(run_label)

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


def calculate_connectivity(X, cov_estimator):
    """Fit given covariance estimator to data and return correlation
     and partial correlation.

    Parameters
    ----------
    X : numpy array
        Time series data.
    cov_estimator : sklearn estimator
        Covariance estimator to fit to data.

    Returns
    -------
    tuple of numpy arrays
        First array is the correlation matrix, second array is the partial
    """
    # get the connectivity measure
    cov_estimator_ = clone(cov_estimator)
    try:
        cv = cov_estimator_.fit(X)
    except FloatingPointError as error:
        if isinstance(cov_estimator_, GraphicalLassoCV):
            print(
                "Caught a FloatingPointError, ",
                "shrinking covariance beforehand...",
            )
            X = empirical_covariance(X, assume_centered=True)
            X = shrunk_covariance(X, shrinkage=1)
            cov_estimator_ = GraphicalLasso(
                alpha=0.52, verbose=0, mode="cd", covariance="precomputed"
            )
            cv = cov_estimator_.fit(X)
        else:
            raise error
    cv_correlation = sym_matrix_to_vec(cv.covariance_, discard_diagonal=True)
    cv_partial = sym_matrix_to_vec(-cv.precision_, discard_diagonal=True)

    return (cv_correlation, cv_partial)


def get_connectomes(cov, data, n_jobs):
    """Wrapper function to calculate connectomes using different covariance
    estimators. Selects appropriate covariance estimator based on the
    given string and adds the connectomes to the given data dataframe."""
    # covariance estimator
    if cov == "Graphical-Lasso":
        cov_estimator = GraphicalLassoCV(
            verbose=11, n_jobs=n_jobs, assume_centered=True
        )
    elif cov == "Ledoit-Wolf":
        cov_estimator = LedoitWolf(assume_centered=True)
    elif cov == "Unregularized":
        cov_estimator = EmpiricalCovariance(assume_centered=True)
    time_series = data["time_series"].tolist()
    connectomes = []
    for ts in tqdm(time_series, desc=cov, leave=True):
        connectome = calculate_connectivity(ts, cov_estimator)
        connectomes.append(connectome)
    correlation = np.asarray([connectome[0] for connectome in connectomes])
    partial_correlation = np.asarray(
        [connectome[1] for connectome in connectomes]
    )
    data[f"{cov} correlation"] = correlation.tolist()
    data[f"{cov} partial correlation"] = partial_correlation.tolist()

    return data
