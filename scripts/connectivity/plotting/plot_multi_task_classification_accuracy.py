import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ibc_public.connectivity.utils_plot import wrap_labels

sns.set_theme()
sns.set_style("whitegrid")
sns.set_context("talk")

### pooled or multi task classification accuracies ###
cache = DATA_ROOT = "/storage/store/work/haggarwa/"
DATA_ROOT2 = "/storage/store2/work/haggarwa/"

n_parcels = 400
if n_parcels == 400:
    # with compcorr and fixed resting state confounds
    results_dir = "fc_acrosstask_classification_400_20240118-143947"
    results_pkl = os.path.join(DATA_ROOT2, results_dir, "all_results.pkl")
    output_dir = "classification_plots_compcorr"
elif n_parcels == 200:
    # with compcorr and fixed resting state confounds
    results_dir = "fc_acrosstask_classification_200_20240117-185001"
    results_pkl = os.path.join(DATA_ROOT2, results_dir, "all_results.pkl")
    output_dir = "classification_plots_200_compcorr"
output_dir = os.path.join(DATA_ROOT2, output_dir)
os.makedirs(output_dir, exist_ok=True)
df = pd.read_pickle(results_pkl)

# cov estimators
cov_estimators = ["Graphical-Lasso", "Ledoit-Wolf", "Unregularized"]
# connectivity measures for each cov estimator
measures = ["correlation", "partial correlation"]
# what to classify
classify = ["Tasks", "Subjects", "Runs"]

df[df.select_dtypes(include=["number"]).columns] *= 100

for clas in classify:
    df_ = df[df["classes"] == clas]
    df_.reset_index(inplace=True, drop=True)

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
        ax_score = sns.barplot(
            y="connectivity",
            x="balanced_accuracy",
            data=df_,
            orient="h",
            palette=sns.color_palette()[0:1],
            order=order,
            facecolor=(0.4, 0.4, 0.4, 1),
        )
        for i in ax_score.containers:
            plt.bar_label(
                i,
                fmt="%.1f",
                label_type="edge",
                fontsize="x-small",
                padding=-45,
                weight="bold",
                color="white",
            )
        wrap_labels(ax_score, 20)
        sns.barplot(
            y="connectivity",
            x="dummy_balanced_accuracy",
            data=df_,
            orient="h",
            palette=sns.color_palette("pastel")[0:1],
            order=order,
            ci=None,
            facecolor=(0.8, 0.8, 0.8, 1),
        )
        plt.xlabel("Accuracy")
        plt.ylabel("FC measure")
        fig = plt.gcf()
        if how_many == "three":
            fig.set_size_inches(6, 2.5)
        plt.savefig(
            os.path.join(output_dir, f"{clas}_classification_{how_many}.png"),
            bbox_inches="tight",
        )
        plt.savefig(
            os.path.join(output_dir, f"{clas}_classification_{how_many}.svg"),
            bbox_inches="tight",
        )
        plt.close("all")
