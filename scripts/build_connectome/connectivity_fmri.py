"""
This script computes:
1) pearson correlation
2) correlations and partial correlations from GraphicalLassoCV estimator
3) correlations and partial correlations GroupSparseCovarianceCV estimator 

Each representing functional connectivity from movie (Raiders of the Lost Ark) 
and Resting state task for ROIs from Schaefer 2018 atlas

See: https://nilearn.github.io/stable/connectivity/functional_connectomes.html
and https://nilearn.github.io/stable/connectivity/connectome_extraction.html
"""


import os
from glob import glob
import numpy as np
from sklearn.covariance import GraphicalLassoCV
from nilearn import datasets
from nilearn.maskers import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure
from nilearn.connectome import GroupSparseCovarianceCV
from nilearn.image import high_variance_confounds
from ibc_public.utils_data import get_subject_session, DERIVATIVES

# cache directory
cache = "/storage/store/work/haggarwa/"

# we will use the movie (raiders) and resting state data
tasks = ["task-Raiders", "task-RestingState"]

# get atlas
atlas = datasets.fetch_atlas_schaefer_2018(
    data_dir=cache, resolution_mm=2, n_rois=400
)
# give atlas a custom name
atlas["name"] = "schaefer400"

# define all 3 connectivity estimators
# 1. pearson's correlation between time series from each ROI
#    calculated for each run (from each session), each subject. Separately
#    returns a 400x400 matrix for each run, each subject
correlation_measure = ConnectivityMeasure(kind="correlation")
# 2. covariance matrix from GraphicalLassoCV
#    calculated over all runs for each subject
#    returns a 400x400 matrix for each subject
glc = GraphicalLassoCV()
# 3. covariance matrix from GroupSparseCovarianceCV
#    calculated over all runs for all subjects
#    returns a 400x400xN matrix where N is the number of subjects
#    saved separately for each subject
gsc = GroupSparseCovarianceCV(verbose=2, n_jobs=6)
# iterate over movie and task-RestingState state task data
for task in tasks:
    if task == "task-Raiders":
        # session names with movie task data
        session_names = ["raiders1", "raiders2"]
        repetition_time = 2
    elif task == "task-RestingState":
        # session names with task-RestingState state task data
        session_names = ["mtt1", "mtt2"]
        repetition_time = 0.76
    # get session numbers for each subject
    subject_sessions = sorted(get_subject_session(session_names))
    sub_ses = {}
    for subject_session in subject_sessions:
        if (
            subject_session[0] in ["sub-11", "sub-12"]
            and subject_session[1] == "ses-13"
        ):
            continue
        try:
            sub_ses[subject_session[0]]
        except KeyError:
            sub_ses[subject_session[0]] = []
        sub_ses[subject_session[0]].append(subject_session[1])
    # define regions using the atlas
    masker = NiftiLabelsMasker(
        labels_img=atlas.maps,
        standardize=True,
        low_pass=0.2,
        high_pass=0.01,
        t_r=repetition_time,
        verbose=1,
        memory=cache,
    ).fit()
    all_ts_all_sub = []
    # iterate over subjects
    for sub, sess in sub_ses.items():
        all_time_series = []
        # iterate over sessions for each subject
        for ses in sess:
            runs = glob(
                os.path.join(DERIVATIVES, sub, ses, "func", f"wrdc*{task}")
            )
            # iterate over runs in each session for each subject
            for run in runs:
                # get run number string
                run_num = run.split("/")[-1].split("_")[-2]
                confounds = glob(
                    os.path.join(
                        DERIVATIVES, sub, ses, "func", f"rp*{run_num}_bold*"
                    )
                )[0]
                # setup tmp dir for saving figures and calculated matrices
                tmp_dir = os.path.join(
                    DERIVATIVES, sub, ses, "func", "connectivity_tmp"
                )
                if not os.path.exists(tmp_dir):
                    os.makedirs(tmp_dir)
                # directory with func data
                func_dir = os.path.join(DERIVATIVES, sub, ses, "func")
                # calculate high-variance confounds
                compcor = high_variance_confounds(run)
                # load confounds and append high-variance confounds
                confounds = np.hstack((np.loadtxt(confounds), compcor))
                # extract time series for atlas defined regions
                time_series = masker.transform(run, confounds=confounds)
                # calculate pearson correlation matrix as csv
                correlation_matrix = correlation_measure.fit_transform(
                    [time_series]
                )[0]
                # save pearson correlation matrix as csv
                corr = os.path.join(
                    func_dir,
                    (
                        f"{sub}_{ses}_{task}_{run_num}_{atlas.name}"
                        "_Pearsons_corr.csv"
                    ),
                )
                np.savetxt(corr, correlation_matrix, delimiter=",")
                # append time series from each run (from each session)
                # to a list for each subject
                # each element in the list is a 2D array with shape
                # (n_timepoints, n_regions)
                all_time_series.append(time_series)
        # concatenate time series over all runs (from each session) for
        # each subject
        # all_ts_per_sub is now a 2D array with shape
        # (n_timepoints * n_runs, n_regions)
        all_ts_per_sub = np.concatenate(all_time_series)
        # fit GraphicalLassoCV on concatenated time series
        glc.fit(all_ts_per_sub)
        # save GraphicalLassoCV inv cov matrix as csv
        part_corr = os.path.join(
            func_dir,
            f"{sub}_{task}_{atlas.name}_GraphicalLassoCV_partcorr.csv",
        )
        np.savetxt(part_corr, -glc.precision_, delimiter=",")
        # save GraphicalLassoCV cov matrix as csv
        corr = os.path.join(
            func_dir, f"{sub}_{task}_{atlas.name}_GraphicalLassoCV_corr.csv"
        )
        np.savetxt(corr, glc.covariance_, delimiter=",")
        # append the (n_timepoints * n_runs, n_regions) arrays for each subject
        # to a list
        all_ts_all_sub.append(all_ts_per_sub)
    # fit Group Sparse Cov est on this list of
    # (n_timepoints * n_runs, n_regions) dim matrices where each matrix
    # corresponds to one subject
    gsc.fit(all_ts_all_sub)
    for count, (sub, sess) in enumerate(sub_ses.items()):
        ses = sorted(sess)[1]
        part_corr = os.path.join(
            func_dir,
            f"{sub}_{task}_{atlas.name}_GroupSparseCovarianceCV_partcorr.csv",
        )
        np.savetxt(part_corr, -gsc.precisions_[:, :, count], delimiter=",")
        corr = os.path.join(
            func_dir,
            f"{sub}_{task}_{atlas.name}_GroupSparseCovarianceCV_corr.csv",
        )
        np.savetxt(corr, gsc.covariances_[:, :, count], delimiter=",")
