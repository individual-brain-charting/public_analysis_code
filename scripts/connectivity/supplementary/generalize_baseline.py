"""CV withing IBC and external GBU datasets to get a baseline performance for
 later generalization tests between the two datasets."""

import pandas as pd
import numpy as np
import os
from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import (
    cross_val_score,
    StratifiedGroupKFold,
)
from matplotlib import pyplot as plt
import seaborn as sns

cache = DATA_ROOT = "/storage/store2/work/haggarwa/"
output_dir = os.path.join(DATA_ROOT, "transfer_classifier")
os.makedirs(output_dir, exist_ok=True)

# cov estimators
cov_estimators = ["Ledoit-Wolf", "Unregularized", "Graphical-Lasso"]
# connectivity measures for each cov estimator
measures = ["correlation", "partial correlation"]


for do_hist_equ in [False, True]:
    # load connectomes for external GBU
    external_connectomes = pd.read_pickle(
        os.path.join(
            DATA_ROOT,
            "external_connectivity_full_length",
            "connectomes_200.pkl",
        )
    )
    # load connectomes for IBC GBU
    connectomes = pd.read_pickle(
        os.path.join(
            DATA_ROOT,
            "ibc_sync_external_connectivity_full_length",
            "connectomes_200.pkl",
        )
    )

    # rename run labels to match across datasets
    connectomes["run_labels"].replace("run-03", "1", inplace=True)
    connectomes["run_labels"].replace("run-04", "2", inplace=True)
    connectomes["run_labels"].replace("run-05", "3", inplace=True)

    external_connectomes["run_labels"].replace("run-01", "1", inplace=True)
    external_connectomes["run_labels"].replace("run-02", "2", inplace=True)
    external_connectomes["run_labels"].replace("run-03", "3", inplace=True)

    classify = ["Runs"]

    ### cv on external GBU ###
    print("\n\ncv on external GBU")
    results = []
    for clas in classify:
        for cov in cov_estimators:
            for measure in measures:
                # train
                classes = external_connectomes["run_labels"].to_numpy(
                    dtype=object
                )
                groups = external_connectomes["subject_ids"].to_numpy(
                    dtype=object
                )
                unique_groups = np.unique(groups)
                data = np.array(
                    external_connectomes[f"{cov} {measure}"].values.tolist()
                )
                classifier = LinearSVC(max_iter=1000000, dual="auto")
                dummy = DummyClassifier(strategy="most_frequent")

                # score
                accuracy = cross_val_score(
                    classifier,
                    data,
                    classes,
                    groups=groups,
                    n_jobs=len(unique_groups),
                    cv=StratifiedGroupKFold(
                        shuffle=True,
                        n_splits=len(unique_groups),
                        random_state=0,
                    ),
                    scoring="accuracy",
                )
                dummy_accuracy = cross_val_score(
                    dummy,
                    data,
                    classes,
                    groups=groups,
                    n_jobs=len(unique_groups),
                    cv=StratifiedGroupKFold(
                        shuffle=True,
                        n_splits=len(unique_groups),
                        random_state=0,
                    ),
                    scoring="accuracy",
                )
                # save results
                result = {
                    "accuracy": np.mean(accuracy),
                    "dummy_accuracy": np.mean(dummy_accuracy),
                    "cov measure": f"{cov} {measure}",
                }
                results.append(result)
                print(
                    f"{cov} {measure}: {np.mean(accuracy):.3f} |"
                    f" {np.mean(dummy_accuracy):.3f}"
                )
        results = pd.DataFrame(results)
        sns.barplot(results, y="cov measure", x="accuracy", orient="h")
        plt.axvline(np.mean(dummy_accuracy), color="k", linestyle="--")
        plt_file = os.path.join(output_dir, f"{clas}.png")
        title = "CV on external"

        plt.xlim(0, 1.05)
        plt.title(title)
        plt.savefig(
            plt_file,
            bbox_inches="tight",
        )
        plt.close()

    ### cv on IBC GBU ###
    print("\n\ncv on IBC GBU")
    results = []
    for clas in classify:
        for cov in cov_estimators:
            for measure in measures:
                # train
                classes = connectomes["run_labels"].to_numpy(dtype=object)
                groups = connectomes["subject_ids"].to_numpy(dtype=object)
                unique_groups = np.unique(groups)
                data = np.array(
                    connectomes[f"{cov} {measure}"].values.tolist()
                )
                classifier = LinearSVC(max_iter=1000000, dual="auto")
                dummy = DummyClassifier(strategy="most_frequent")

                # score
                accuracy = cross_val_score(
                    classifier,
                    data,
                    classes,
                    groups=groups,
                    n_jobs=len(unique_groups),
                    cv=StratifiedGroupKFold(
                        shuffle=True,
                        n_splits=len(unique_groups),
                        random_state=0,
                    ),
                    scoring="accuracy",
                )
                dummy_accuracy = cross_val_score(
                    dummy,
                    data,
                    classes,
                    groups=groups,
                    n_jobs=len(unique_groups),
                    cv=StratifiedGroupKFold(
                        shuffle=True,
                        n_splits=len(unique_groups),
                        random_state=0,
                    ),
                    scoring="accuracy",
                )
                # save results
                result = {
                    "accuracy": np.mean(accuracy),
                    "dummy_accuracy": np.mean(dummy_accuracy),
                    "cov measure": f"{cov} {measure}",
                }
                results.append(result)
                print(
                    f"{cov} {measure}: {np.mean(accuracy):.3f} |"
                    f" {np.mean(dummy_accuracy):.3f}"
                )
        results = pd.DataFrame(results)
        sns.barplot(results, y="cov measure", x="accuracy", orient="h")
        plt.axvline(np.mean(dummy_accuracy), color="k", linestyle="--")
        plt_file = os.path.join(output_dir, f"{clas}.png")
        title = "CV on IBC"
        plt.xlim(0, 1.05)
        plt.title(title)
        plt.savefig(
            plt_file,
            bbox_inches="tight",
        )
        plt.close()
