"""This script trains a classifier on external GBU data (from Mantini et al.
 2012) and tests on IBC GBU"""
import pandas as pd
import numpy as np
import os
from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    accuracy_score,
    confusion_matrix,
)
from skimage import exposure
from matplotlib import pyplot as plt
import seaborn as sns
import time
from sklearn.preprocessing import RobustScaler, StandardScaler
from ibc_public.connectivity.utils_plot import wrap_labels

#### train on external test on IBC GBU ###
cache = DATA_ROOT = "/storage/store2/work/haggarwa/"

output_dir = os.path.join(
    DATA_ROOT,
    f"fc_external_to_ibc_classification_{time.strftime('%Y%m%d-%H%M%S')}",
)
os.makedirs(output_dir, exist_ok=True)

# cov estimators
cov_estimators = ["Unregularized", "Ledoit-Wolf", "Graphical-Lasso"]
# connectivity measures for each cov estimator
measures = ["correlation", "partial correlation"]


# what are we classifying
classify = ["Runs"]

# use which cross validation scheme
cv_scheme = "GroupShuffleSplit"  # "GroupShuffleSplit"

# evaluation metrics
scoring = [
    "accuracy",
    "balanced_accuracy",
    "roc_auc_ovr_weighted",
    "roc_auc_ovr",
    "roc_auc_ovo_weighted",
    "roc_auc_ovo",
]

for scaling_type in [
    "no_scaling",
    "robust_scaled",
    "standard_scaled",
    "hist_eq",
]:
    # load connectomes for external GBU
    external_connectomes = pd.read_pickle(
        os.path.join(
            DATA_ROOT,
            "external_connectivity_20240125-104121",
            "connectomes_200_compcorr.pkl",
        )
    )

    # load connectomes for IBC GBU
    IBC_connectomes = pd.read_pickle(
        os.path.join(
            DATA_ROOT,
            "connectomes_200_comprcorr",
        )
    )

    IBC_connectomes = IBC_connectomes[
        IBC_connectomes["tasks"] == "GoodBadUgly"
    ]
    IBC_connectomes = IBC_connectomes[
        IBC_connectomes["run_labels"].isin(["run-03", "run-04", "run-05"])
    ]
    IBC_connectomes.reset_index(inplace=True, drop=True)
    # rename run labels to match across datasets
    IBC_connectomes["run_labels"].replace("run-03", "1", inplace=True)
    IBC_connectomes["run_labels"].replace("run-04", "2", inplace=True)
    IBC_connectomes["run_labels"].replace("run-05", "3", inplace=True)

    external_connectomes["run_labels"].replace("run-01", "1", inplace=True)
    external_connectomes["run_labels"].replace("run-02", "2", inplace=True)
    external_connectomes["run_labels"].replace("run-03", "3", inplace=True)

    classify = ["Runs"]
    results = []
    for cov in cov_estimators:
        for measure in measures:
            # ibc connectomes
            ibc_fc = np.array(IBC_connectomes[f"{cov} {measure}"].tolist())
            # external connectomes
            external_fc = np.array(
                external_connectomes[f"{cov} {measure}"].tolist()
            )
            if scaling_type == "robust_scaled":
                print("\n\n With RobustScaler")
                # do robust scaling for each array
                ibc_fc = RobustScaler(unit_variance=True).fit_transform(ibc_fc)
                external_fc = RobustScaler(unit_variance=True).fit_transform(
                    external_fc
                )

            elif scaling_type == "standard_scaled":
                print("\n\n With StandardScaler")
                # do robust scaling for each array
                ibc_fc = StandardScaler().fit_transform(ibc_fc)
                external_fc = StandardScaler().fit_transform(external_fc)
            elif scaling_type == "hist_eq":
                print("\n\n With Histogram Equalization")
                # do robust scaling for each array
                ibc_fc = exposure.equalize_hist(ibc_fc)
                external_fc = exposure.equalize_hist(external_fc)
            else:
                print("\n\n Without Scaling")

            ### train on external GBU, test on IBC GBU ###
            print("\n\ntrain on external GBU, test on IBC GBU")

            for clas in classify:
                # train
                classes = external_connectomes["run_labels"].to_numpy(
                    dtype=object
                )
                data = external_fc.copy()
                classifier = LinearSVC(max_iter=1000, dual=False).fit(
                    data, classes
                )
                dummy = DummyClassifier(strategy="most_frequent").fit(
                    data, classes
                )

                # test
                predictions = classifier.predict(ibc_fc)
                dummy_predictions = dummy.predict(ibc_fc)

                # score
                accuracy = accuracy_score(
                    IBC_connectomes["run_labels"], predictions
                )
                balanced_accuracy = balanced_accuracy_score(
                    IBC_connectomes["run_labels"], predictions
                )
                dummy_accuracy = accuracy_score(
                    IBC_connectomes["run_labels"], dummy_predictions
                )
                dummy_balanced_accuracy = balanced_accuracy_score(
                    IBC_connectomes["run_labels"], dummy_predictions
                )
                conf_mat = confusion_matrix(
                    IBC_connectomes["run_labels"], predictions
                )

                # save results
                result = {
                    "accuracy": accuracy,
                    "balanced_accuracy": balanced_accuracy,
                    "dummy_accuracy": dummy_accuracy,
                    "dummy_balanced_accuracy": dummy_balanced_accuracy,
                    "conf_mat": conf_mat,
                    "cov measure": f"{cov} {measure}",
                    "classifying": clas,
                    "task_label": "GoodBadUgly",
                }
                results.append(result)
                print(
                    f"{cov} {measure}: {accuracy:.3f} | {dummy_accuracy:.3f}"
                )

    results = pd.DataFrame(results)

    results.to_pickle(
        os.path.join(output_dir, f"all_results_{scaling_type}.pkl")
    )

### plot ###
sns.set_theme()
sns.set_style("whitegrid")
sns.set_context("talk")


for scaling_type in [
    "no_scaling",
    "robust_scaled",
    "standard_scaled",
    "hist_eq",
]:
    results = pd.read_pickle(
        os.path.join(output_dir, f"all_results_{scaling_type}.pkl")
    )
    results[results.select_dtypes(include=["number"]).columns] *= 100

    for clas in classify:
        df_ = results[results["classifying"] == clas]
        for how_many in ["all", "three"]:
            if how_many == "all":
                order = [
                    "Unregularized correlation",
                    "Unregularized partial correlation",
                    "Ledoit-Wolf correlation",
                    "Ledoit-Wolf partial correlation",
                    "Graphical-Lasso correlation",
                    "Graphical-Lasso partial correlation",
                ]
            elif how_many == "three":
                order = [
                    "Unregularized correlation",
                    "Ledoit-Wolf correlation",
                    "Graphical-Lasso partial correlation",
                ]
            if clas == "Tasks":
                eval_metric1 = "balanced_accuracy"
                eval_metric2 = "dummy_balanced_accuracy"
                legend_cutoff = 9
                palette_init = 1
                xlabel = "Accuracy"
                title = ""
            elif clas == "Runs":
                eval_metric1 = "balanced_accuracy"
                eval_metric2 = "dummy_balanced_accuracy"
                legend_cutoff = 0
                palette_init = 2
                xlabel = "Accuracy"
                title = ""
            else:
                eval_metric1 = "balanced_accuracy"
                eval_metric2 = "dummy_balanced_accuracy"
                legend_cutoff = 4
                palette_init = 0
                xlabel = "Accuracy"
                title = "Within task"
            bar_label_color = "white"
            bar_label_weight = "bold"
            ax_score = sns.barplot(
                y="cov measure",
                x=eval_metric1,
                data=df_,
                orient="h",
                hue="task_label",
                palette=sns.color_palette("tab20c")[5:],
                order=order,
                # errwidth=1,
            )

            wrap_labels(ax_score, 20)
            for i, container in enumerate(ax_score.containers):
                plt.bar_label(
                    container,
                    fmt="%.1f",
                    label_type="edge",
                    fontsize="x-small",
                    padding=-40,
                    weight=bar_label_weight,
                    color=bar_label_color,
                )
            ax_chance = sns.barplot(
                y="cov measure",
                x=eval_metric2,
                data=df_,
                orient="h",
                hue="task_label",
                palette=sns.color_palette("pastel")[7:],
                order=order,
                ci=None,
            )
            plt.xlabel(xlabel)
            plt.ylabel("FC measure")
            plt.title(title)
            legend = plt.legend(
                framealpha=0, loc="center left", bbox_to_anchor=(1, 0.5)
            )
            # remove legend repetition for chance level
            for i, (text, handle) in enumerate(
                zip(legend.texts, legend.legend_handles)
            ):
                if i > legend_cutoff:
                    text.set_text("Chance-level")

            fig = plt.gcf()
            if how_many == "three":
                fig.set_size_inches(6, 2.5)
            plt.savefig(
                os.path.join(
                    output_dir,
                    f"{clas}_classification_{scaling_type}_{how_many}.png",
                ),
                bbox_inches="tight",
            )
            plt.savefig(
                os.path.join(
                    output_dir,
                    f"{clas}_classification_{scaling_type}_{how_many}.svg",
                ),
                bbox_inches="tight",
            )
            plt.close()
