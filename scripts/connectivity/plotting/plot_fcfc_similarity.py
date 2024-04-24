import os
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from ibc_public.connectivity.utils_plot import get_lower_tri_heatmap


sns.set_theme()
sns.set_style("whitegrid")
sns.set_context("talk")

### fc-fc similarity, sub-spec matrices
# cov estimators
cov_estimators = ["Graphical-Lasso", "Ledoit-Wolf", "Unregularized"]
# connectivity measures for each cov estimator
measures = ["correlation", "partial correlation"]
# what to classify
classify = ["Tasks", "Subjects", "Runs"]
# tasks
tasks = ["RestingState", "Raiders", "GoodBadUgly", "MonkeyKingdom", "Mario"]
# load the data
DATA_ROOT = "/storage/store2/work/haggarwa/"
n_parcels = 400
if n_parcels == 400:
    results_dir = "fc_similarity_20240411-155121"  # adding back the overall mean similarity
elif n_parcels == 200:
    results_dir = "fc_similarity_20240411-155035"  # adding back the overall mean similarity
similarity_data = pd.read_pickle(
    os.path.join(DATA_ROOT, results_dir, "results.pkl")
)
# create matrices showing similarity between tasks
if n_parcels == 400:
    output_dir = os.path.join(DATA_ROOT, "similarity_plots_compcorr")
elif n_parcels == 200:
    output_dir = os.path.join(DATA_ROOT, "similarity_plots_200_compcorr")
os.makedirs(output_dir, exist_ok=True)
for cov in cov_estimators:
    for measure in measures:
        similarity_values = np.zeros((len(tasks), len(tasks)))
        similarity_tasks = np.zeros((len(tasks), len(tasks)), dtype=object)
        for i, task1 in enumerate(tasks):
            for j, task2 in enumerate(tasks):
                if i == j:
                    similarity = 1
                elif i > j:
                    similarity = similarity_values[j][i]
                else:
                    mask = (
                        (similarity_data["task1"] == task1)
                        & (similarity_data["task2"] == task2)
                        & (similarity_data["measure"] == f"{cov} {measure}")
                        & (similarity_data["centering"] == "uncentered")
                    )
                    matrix = similarity_data[mask]["matrix"].to_numpy()[0]
                    similarity = np.mean(matrix)
                similarity_values[i][j] = similarity
                similarity_tasks[i][j] = (task1, task2)
        with sns.plotting_context("notebook"):
            get_lower_tri_heatmap(
                similarity_values,
                # cmap="Reds",
                figsize=(5, 5),
                labels=tasks,
                output=os.path.join(output_dir, f"similarity_{cov}_{measure}"),
                triu=True,
                diag=True,
                tril=False,
                title=f"{cov} {measure}",
                fontsize=15,
            )
# create matrices showing subject-specificity of tasks
if n_parcels == 400:
    output_dir = os.path.join(DATA_ROOT, "subspec_plots_compcorr")
elif n_parcels == 200:
    output_dir = os.path.join(DATA_ROOT, "subspec_plots_200_compcorr")
os.makedirs(output_dir, exist_ok=True)
diffss = []
simss = []
for cov in cov_estimators:
    for measure in measures:
        similarity_values = np.zeros((len(tasks), len(tasks)))
        similarity_type = np.zeros((len(tasks), len(tasks)), dtype=object)
        diffs = []
        sims = []
        for centering in ["uncentered", "centered"]:
            for i, task1 in enumerate(tasks):
                for j, task2 in enumerate(tasks):
                    if i < j:
                        mask = (
                            (similarity_data["task1"] == task1)
                            & (similarity_data["task2"] == task2)
                            & (
                                similarity_data["measure"]
                                == f"{cov} {measure}"
                            )
                            & (similarity_data["centering"] == centering)
                        )
                        matrix = similarity_data[mask]["matrix"].to_numpy()[0]
                        n_subs = len(
                            similarity_data[mask]["kept_subjects"].to_numpy()[
                                0
                            ]
                        )
                        matrix = matrix.reshape((n_subs, n_subs))
                        same_sub = np.mean(np.diagonal(matrix, offset=0))
                        upper_tri = np.triu(matrix, k=1).flatten()
                        lower_tri = np.tril(matrix, k=-1).flatten()
                        cross_sub = np.concatenate((upper_tri, lower_tri))
                        cross_sub = np.mean(cross_sub)
                        similarity_values[i][j] = same_sub
                        similarity_values[j][i] = cross_sub
                        similarity_type[i][j] = ""
                        similarity_type[j][i] = ""
                        diffs.append(same_sub - cross_sub)
                        sims.extend(matrix.flatten())
                    elif i == j:
                        similarity_values[i][j] = np.nan
                        similarity_type[i][j] = ""
                    else:
                        continue
                    similarity_type[1][3] = "Within-subject\nsimilarity"
                    similarity_type[3][1] = "Across-subject\nsimilarity"
            similarity_type = pd.DataFrame(similarity_type)
            similarity_type = similarity_type.astype("str")
            similarity_annot = similarity_values.round(2)
            similarity_annot = similarity_annot.astype("str")
            # similarity_annot[1][3] = "Within\nsubs"
            # similarity_annot[3][1] = "Across\nsubs"
            with sns.plotting_context("notebook"):
                get_lower_tri_heatmap(
                    similarity_values,
                    figsize=(5, 5),
                    # cmap="RdBu_r",
                    annot=similarity_annot,
                    labels=tasks,
                    output=os.path.join(
                        output_dir, f"subspec_{cov}_{measure}_{centering}"
                    ),
                    title=f"{cov} {measure}",
                    fontsize=15,
                )
            if centering == "centered":
                print(
                    f"{cov} {measure}: {np.mean(diffs):.2f} +/- {np.std(diffs):.2f}"
                )
                print(f"Av sim {cov} {measure}: {np.nanmean(sims):.2f}")
                if (
                    cov == "Graphical-Lasso"
                    and measure == "partial correlation"
                ):
                    diffss.append(diffs)
                    simss.append(sims)
                elif cov == "Ledoit-Wolf" and measure == "correlation":
                    diffss.append(diffs)
                    simss.append(sims)
                elif cov == "Unregularized" and measure == "correlation":
                    diffss.append(diffs)
                    simss.append(sims)
                else:
                    continue
print(
    "\n\n***Testing whether difference between within sub similarity and "
    "across sub similarity is greater in Graphical-Lasso partial corr than "
    "corr measures***"
)
print(
    "Graphical-Lasso partial corr > Ledoit-Wolf corr\n",
    mannwhitneyu(
        diffss[0],
        diffss[1],
        alternative="greater",
    ),
)
print(
    "Graphical-Lasso partial corr > Unregularized corr\n",
    mannwhitneyu(
        diffss[0],
        diffss[2],
        alternative="greater",
    ),
)
print(
    "\n\n***Testing whether similarity values for Graphical-Lasso partial "
    "corr are greater than for corr measures***"
)
print(
    "Graphical-Lasso partial corr > Ledoit-Wolf corr\n",
    mannwhitneyu(
        simss[0],
        simss[1],
        alternative="greater",
    ),
)
print(
    "Graphical-Lasso partial corr > Unregularized corr\n",
    mannwhitneyu(
        simss[0],
        simss[2],
        alternative="greater",
    ),
)


#### comparison between distributions of fc-fc similarity values for 400 vs.
# 200 parcels
# cov estimators
cov_estimators = ["Graphical-Lasso", "Ledoit-Wolf", "Unregularized"]
# connectivity measures for each cov estimator
measures = ["correlation", "partial correlation"]
# what to classify
classify = ["Tasks", "Subjects", "Runs"]
# tasks
tasks = ["RestingState", "Raiders", "GoodBadUgly", "MonkeyKingdom", "Mario"]
# load the data
DATA_ROOT = "/storage/store2/work/haggarwa/"
# colors
colors = ["red", "blue"]
# output directory
output_dir = os.path.join(
    DATA_ROOT, "200V400_fcfc_similarity_comparison_compcorr"
)
os.makedirs(output_dir, exist_ok=True)

for cov in cov_estimators:
    for measure in measures:
        fig, ax = plt.subplots(figsize=(5, 5))
        means = []
        values = []
        for n_parcels, color in zip([200, 400], colors):
            if n_parcels == 400:
                # adding back the overall mean similarity
                results_dir = "fc_similarity_20240411-155121"
            elif n_parcels == 200:
                # adding back the overall mean similarity
                results_dir = "fc_similarity_20240411-155035"
            similarity_data = pd.read_pickle(
                os.path.join(DATA_ROOT, results_dir, "results.pkl")
            )
            mask = (similarity_data["measure"] == f"{cov} {measure}") & (
                similarity_data["centering"] == "centered"
            )
            matrix = similarity_data[mask]["matrix"].to_numpy()[0]
            n_subs = len(similarity_data[mask]["kept_subjects"].to_numpy()[0])
            matrix = matrix.reshape((n_subs, n_subs))
            upper_tri = np.triu(matrix, k=1).flatten()
            lower_tri = np.tril(matrix, k=-1).flatten()
            cross_sub = np.concatenate((upper_tri, lower_tri))
            values.append(cross_sub)
            threshold = np.percentile(cross_sub, 1)
            cross_sub_thresh = cross_sub[(cross_sub > threshold)]
            ax.hist(
                cross_sub_thresh.flatten(),
                bins=50,
                label=f"{n_parcels} regions",
                color=color,
                alpha=0.5,
                density=True,
            )
            means.append(np.mean(cross_sub_thresh.flatten()))
        MWU_test = mannwhitneyu(values[0], values[1], alternative="greater")
        ax.annotate(
            f"MWU test\n200 > 400 regions:\np = {MWU_test[1]:.2e}",
            xy=(0.57, 0.83),
            xycoords="axes fraction",
            bbox={"fc": "0.8"},
            fontsize=12,
        )
        ax.axvline(
            means[0],
            color=colors[0],
            linestyle="--",
        )
        ax.axvline(
            means[1],
            color="k",
            linestyle="--",
            label="mean",
        )
        ax.axvline(
            means[1],
            color=colors[1],
            linestyle="--",
        )
        plt.legend(framealpha=0, loc="center left", bbox_to_anchor=(1, 0.5))
        ax.set_title(f"{cov} {measure}")
        plt.xlabel("FC-FC Similarity")
        plt.savefig(
            os.path.join(output_dir, f"{cov}_{measure}_hist.png"),
            bbox_inches="tight",
        )
        plt.savefig(
            os.path.join(output_dir, f"{cov}_{measure}_hist.svg"),
            transparent=True,
            bbox_inches="tight",
        )
        plt.close()
        print(
            f"Testing whether 200 > 400 for {cov} {measure}\n",
            MWU_test,
        )
