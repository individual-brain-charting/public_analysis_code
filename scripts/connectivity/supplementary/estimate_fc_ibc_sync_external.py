"""Pipeline to calculate functional connectivity for IBC GBU runs synced 
with external GBU data (from Mantini et al. 2012)"""

import os
import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from nilearn.image import index_img
from nilearn.maskers import NiftiLabelsMasker
from nilearn import datasets
from ibc_public import utils_connectivity as fc
from sklearn.covariance import (
    GraphicalLassoCV,
    LedoitWolf,
    EmpiricalCovariance,
)
from sklearn.base import clone
from nilearn.connectome import sym_matrix_to_vec


def subject_run(task, dataset):
    """Get combinations of subject and run labels for a given task.
    Used to parallelize time series extraction."""
    subject_sessions, _ = fc.get_ses_modality(task, dataset)
    for subject, sessions in subject_sessions.items():
        for session in sorted(sessions):
            niftis, run_labels = fc._get_niftis(
                task, subject, session, dataset
            )
            for nifti, run_label in zip(niftis, run_labels):
                if run_label in ["run-03", "run-04", "run-05"]:
                    confound = fc._get_confounds(
                        run_label, subject, session, dataset
                    )
                    yield subject, nifti, run_label, confound
                else:
                    continue


def slice_nifti_confound(nifti, confound, run):
    """Slice nifti and confounds"""
    if run == "run-03":
        end = 304 - 34
    elif run == "run-04":
        end = 304 - 46
    elif run == "run-05":
        end = 304 - 68
    img = index_img(nifti, slice(0, end))
    confound = np.loadtxt(confound)[:end, :]
    return img, confound


def get_time_series(
    atlas, subject, nifti, run_label, confound, task, do_slice=True
):
    """Get time series from external data"""
    data = {
        "time_series": [],
        "subject_ids": [],
        "run_labels": [],
        "tasks": [],
    }
    repetition_time = 2
    print(f"Getting time series for {subject}, {run_label} ...")
    masker = NiftiLabelsMasker(
        labels_img=atlas.maps,
        standardize=True,
        low_pass=0.2,
        high_pass=0.01,
        t_r=repetition_time,
        verbose=0,
        memory_level=0,
        n_jobs=20,
    ).fit()
    if do_slice:
        nifti, confound = slice_nifti_confound(nifti, confound, run_label)
    time_series = masker.transform(nifti, confounds=confound)
    data["time_series"] = time_series
    data["subject_ids"] = subject
    data["run_labels"] = run_label
    data["tasks"] = task

    return data


def get_ts_cov(time_series, cov_estimators):
    """Get time series and covariance estimator combinations. Used to
    parallelize connectivity estimation."""
    for ts in time_series:
        for cov in cov_estimators:
            yield ts, cov


def calculate_connectivity(X, cov_estimator):
    """Fit given covariance estimator to data and return correlation and
    partial correlation.

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
    cv = cov_estimator_.fit(X)
    cv_correlation = sym_matrix_to_vec(cv.covariance_, discard_diagonal=True)
    cv_partial = sym_matrix_to_vec(-cv.precision_, discard_diagonal=True)

    return (cv_correlation, cv_partial)


def get_connectomes(cov, ts):
    # covariance estimator
    if cov == "Graphical-Lasso":
        cov_estimator = GraphicalLassoCV(
            verbose=0,
            n_jobs=1,
            assume_centered=True,
        )
    elif cov == "Ledoit-Wolf":
        cov_estimator = LedoitWolf(assume_centered=True)
    elif cov == "Unregularized":
        cov_estimator = EmpiricalCovariance(assume_centered=True)
    print(
        f"\nCalculating {cov} connectivity for {ts['subject_ids']},"
        " {ts['run_labels']}..."
    )
    correlation, partial_correlation = calculate_connectivity(
        ts["time_series"], cov_estimator
    )
    ts[f"{cov} correlation"] = correlation
    ts[f"{cov} partial correlation"] = partial_correlation

    return ts


if __name__ == "__main__":
    n_jobs = 10
    # output root directory
    cache = DATA_ROOT = "/storage/store2/work/haggarwa/"
    # output directory
    output_dir = os.path.join(
        DATA_ROOT,
        f"ibc_sync_external_connectivity_{time.strftime('%Y%m%d-%H%M%S')}",
    )
    # output file path
    fc_data_path = os.path.join(output_dir, "connectomes_200.pkl")
    os.makedirs(output_dir, exist_ok=True)
    # get the atlas
    atlas = datasets.fetch_atlas_schaefer_2018(
        data_dir=cache, resolution_mm=2, n_rois=200
    )
    task = "GoodBadUgly"
    dataset = "ibc"
    # cov estimators
    cov_estimators = ["Ledoit-Wolf", "Unregularized", "Graphical-Lasso"]
    # connectivity measures for each cov estimator
    measures = ["correlation", "partial correlation"]

    print("Time series extraction...")
    time_series = Parallel(n_jobs=n_jobs, verbose=11)(
        delayed(get_time_series)(
            atlas, subject, nifti, run_label, confound, task, do_slice=True
        )
        for subject, nifti, run_label, confound in subject_run(task, dataset)
    )
    pd.DataFrame(time_series).to_pickle(
        os.path.join(output_dir, "time_series.pkl")
    )

    print("\nConnectivity estimation...")
    connectomes = Parallel(n_jobs=n_jobs, verbose=11)(
        delayed(get_connectomes)(cov, ts)
        for ts, cov in get_ts_cov(time_series, cov_estimators)
    )

    connectomes_dups = pd.DataFrame(connectomes)
    connectomes = connectomes_dups.drop_duplicates(
        subset=["subject_ids", "run_labels", "tasks"]
    ).reset_index(drop=True)
    cnt = 0
    for cov in cov_estimators:
        for measure in measures:
            ids = (
                connectomes_dups[f"{cov} {measure}"]
                .drop_duplicates()
                .dropna()
                .index
            )
            if cnt == 0:
                ts = np.concatenate(
                    connectomes_dups.loc[ids]["time_series"].values
                )
            else:
                assert np.array_equal(
                    ts,
                    np.concatenate(
                        connectomes_dups.loc[ids]["time_series"].values
                    ),
                )
                ts = np.concatenate(
                    connectomes_dups.loc[ids]["time_series"].values
                )
            clean_col = connectomes_dups[f"{cov} {measure}"].dropna()
            connectomes[f"{cov} {measure}"] = clean_col.reset_index(drop=True)
            cnt += 1

    connectomes.to_pickle(fc_data_path)
