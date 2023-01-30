"""
This script computes pearson correlation and covariance and inverse covariances 
from GraphicalLassoCV and GroupSparseCovariance estimators (each representing 
functional connectivity) from movie (Raiders of the Lost Ark) and resting state 
fMRI for ROIs from a Schaefer 2018 atlas

See: https://nilearn.github.io/stable/connectivity/functional_connectomes.html
and https://nilearn.github.io/stable/connectivity/connectome_extraction.html
"""

import os
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
from sklearn.covariance import GraphicalLassoCV
from nilearn import datasets
from nilearn.maskers import NiftiLabelsMasker
from nilearn import plotting
from nilearn.connectome import ConnectivityMeasure
from nilearn.connectome import GroupSparseCovarianceCV
from ibc_public.utils_data import get_subject_session
from nilearn.image import high_variance_confounds

if 0:
    DATA_ROOT = "/neurospin/ibc/derivatives/"
    mem = "/neurospin/tmp/bthirion/"
else:
    DATA_ROOT = "/storage/store2/data/ibc/derivatives/"
    mem = "/storage/store/work/bthirion/"

fmris = ["movie", "resting"]

# get atlas
atlas = datasets.fetch_atlas_schaefer_2018(
    data_dir=mem, resolution_mm=2, n_rois=400
)
# give atlas a custom name
atlas["name"] = "schaefer400"

# define connectivity estimators
correlation_measure = ConnectivityMeasure(kind="correlation")
glc = GraphicalLassoCV()

for fmri in fmris:
    if fmri == "movie":
        # get session numbers for movie fmri
        subject_sessions = sorted(
            get_subject_session(["raiders1", "raiders2"])
        )
        sub_ses = dict(
            [
                (
                    subject_sessions[2 * i][0],
                    [
                        subject_sessions[2 * i][1],
                        subject_sessions[2 * i + 1][1],
                    ],
                )
                for i in range(len(subject_sessions) // 2)
            ]
        )

        # define regions using the atlas
        masker = NiftiLabelsMasker(
            labels_img=atlas.maps,
            standardize=True,
            low_pass=0.2,
            high_pass=0.01,
            t_r=2,
            verbose=1,
            memory=mem,
        ).fit()
    elif fmri == "resting":
        # get session numbers for restin fmri
        subject_sessions = sorted(get_subject_session(["mtt1", "mtt2"]))
        sub_ses = dict(
            [
                (
                    subject_sessions[2 * i][0],
                    [
                        subject_sessions[2 * i][1],
                        subject_sessions[2 * i + 1][1],
                    ],
                )
                for i in range(len(subject_sessions) // 2)
            ]
        )

        # define regions using the atlas
        masker = NiftiLabelsMasker(
            labels_img=atlas.maps,
            standardize=True,
            low_pass=0.2,
            high_pass=0.01,
            t_r=0.76,
            verbose=1,
            memory=mem,
        ).fit()

    all_ts_all_sub = []
    for sub, sess in sub_ses.items():
        all_time_series = []
        for ses in sess:
            if fmri == "movie":
                runs = glob(os.path.join(DATA_ROOT, sub, ses, "func", "wrdc*"))
            elif fmri == "resting":
                runs = glob(
                    os.path.join(DATA_ROOT, sub, ses, "func", "wrdc*Resting*")
                )
            for run in runs:
                # get run number string
                run_num = run.split("/")[-1].split("_")[-2]
                confounds = glob(
                    os.path.join(
                        DATA_ROOT, sub, ses, "func", f"rp*{run_num}_bold*"
                    )
                )[0]
                # setup tmp dir for saving figures
                tmp_dir = os.path.join(
                    DATA_ROOT, sub, ses, "func", "connectivity_tmp"
                )
                if not os.path.exists(tmp_dir):
                    os.makedirs(tmp_dir)

                # add high-variance confounds
                compcor = high_variance_confounds(run)
                confounds = np.hstack((np.loadtxt(confounds), compcor))

                # extract time series for those regions
                time_series = masker.transform(run, confounds=confounds)
                all_time_series.append(time_series)

                # save pearson correlation matrix as csv
                correlation_matrix = correlation_measure.fit_transform(
                    [time_series]
                )[0]
                corr = os.path.join(
                    tmp_dir,
                    (f"{atlas.name}_corr_{sub}_{ses}_{run_num}.csv"),
                )
                np.savetxt(corr, correlation_matrix, delimiter=",")
                # plot heatmap and save fig
                fig = plt.figure(figsize=(10, 10))
                plotting.plot_matrix(
                    correlation_matrix,
                    labels=atlas.labels,
                    figure=fig,
                    vmax=1,
                    vmin=-1,
                    title="Covariance",
                )
                corr_fig = os.path.join(
                    tmp_dir,
                    f"{atlas.name}_corr_{sub}_{ses}_{run_num}.png",
                )
                fig.savefig(corr_fig, bbox_inches="tight")

                plt.close("all")

        all_ts_per_sub = np.concatenate(all_time_series)
        glc.fit(all_ts_per_sub)
        # save correlation and partial correlation matrices as csv
        part_corr = os.path.join(
            tmp_dir, f"{atlas.name}_part_corr_{sub}_all_gsc.csv"
        )
        np.savetxt(part_corr, -gsc.precisions_, delimiter=",")
        corr = os.path.join(tmp_dir, f"{atlas.name}_corr_{sub}_all_gsc.csv")
        np.savetxt(corr, gsc.covariances_, delimiter=",")

        # plot part_corr heatmap
        fig = plt.figure(figsize=(10, 10))
        plotting.plot_matrix(
            -gsc.precisions_,
            labels=atlas.labels,
            figure=fig,
            vmax=1,
            vmin=-1,
            title="Sparse inverse covariance",
        )
        part_corr_fig = os.path.join(
            tmp_dir, f"{atlas.name}_part_corr_{sub}_all.png"
        )
        fig.savefig(part_corr_fig, bbox_inches="tight")
        # plot corr heatmap
        fig = plt.figure(figsize=(10, 10))
        plotting.plot_matrix(
            gsc.covariances_,
            labels=atlas.labels,
            figure=fig,
            vmax=1,
            vmin=-1,
            title="Covariance with sparse inverse",
        )
        corr_fig = os.path.join(tmp_dir, f"{atlas.name}_corr_{sub}_all.png")
        fig.savefig(corr_fig, bbox_inches="tight")
        plt.close("all")

        # append n_ts x 400 parcels matrix for each subject to a list for 
        # Group Sparse Cov estimation
        all_ts_all_sub.append(all_ts_per_sub)

    # define gsc estimator with parallel workers
    gsc = GroupSparseCovarianceCV(verbose=2, n_jobs=6)
    # fit Group Sparse Cov est on the list of n_ts x 400 parcels matrices where 
    # each matrix corresponds to one subject
    gsc.fit(all_ts_all_sub)
    # save cov and inv cov estimates as csv
    count = 0
    for sub, sess in sub_ses.items():
        ses = sorted(sess)[1]
        part_corr = os.path.join(
            tmp_dir, f"{atlas.name}_part_corr_{sub}_all_gsc.csv"
        )
        np.savetxt(part_corr, -gsc.precisions_[:, :, count], delimiter=",")
        corr = os.path.join(tmp_dir, f"{atlas.name}_corr_{sub}_all_gsc.csv")
        np.savetxt(corr, gsc.covariances_[:, :, count], delimiter=",")
        count += 1
