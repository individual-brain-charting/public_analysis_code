import os
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from ibc_public.connectivity.utils_plot import insert_stats, wrap_labels

sns.set_theme()
sns.set_style("whitegrid")
sns.set_context("talk")


### fc-sc similarity, sub-spec
hatches = [None, "X", "\\", "/", "|"] * 8
# cov estimators
cov_estimators = ["Graphical-Lasso", "Ledoit-Wolf", "Unregularized"]
# connectivity measures for each cov estimator
measures = ["correlation", "partial correlation"]
# what to classify
classify = ["Tasks", "Subjects", "Runs"]
# tasks
tasks = [
    "RestingState",
    "Raiders",
    "GoodBadUgly",
    "MonkeyKingdom",
    "Mario",
]
# load the data
DATA_ROOT = "/storage/store2/work/haggarwa/"
n_parcels = 200
if n_parcels == 400:
    # adding back the overall mean similarity
    results_dir = "fc_similarity_20240411-155121"
    output_dir = os.path.join(DATA_ROOT, "fc-sc_subspec_plots_compcorr")
elif n_parcels == 200:
    # adding back the overall mean similarity
    results_dir = "fc_similarity_20240411-155035"
    output_dir = os.path.join(DATA_ROOT, "fc-sc_subspec_plots_200_compcorr")

similarity_data = pd.read_pickle(
    os.path.join(DATA_ROOT, results_dir, "results.pkl")
)
os.makedirs(output_dir, exist_ok=True)
similarity_data = similarity_data[similarity_data["task2"] == "SC"]

for centering in similarity_data["centering"].unique():
    for cov in cov_estimators:
        for measure in measures:
            df = similarity_data[
                (similarity_data["centering"] == centering)
                & (similarity_data["measure"] == f"{cov} {measure}")
            ]
            for test in ["t", "mwu"]:
                d = {
                    "Comparison": [],
                    "FC measure": [],
                    "Task vs. SC": [],
                    "Similarity": [],
                }
                p_values = {}
                for _, row in df.iterrows():
                    n_subs = len(row["kept_subjects"])
                    corr = row["matrix"].reshape(n_subs, n_subs)
                    same_sub = np.diagonal(corr, offset=0).tolist()
                    upper_tri = corr[np.triu_indices_from(corr, k=1)].tolist()
                    lower_tri = corr[np.tril_indices_from(corr, k=-1)].tolist()
                    cross_sub = upper_tri + lower_tri
                    d["Comparison"].extend(
                        [f"{row['comparison']} Within Subject"] * len(same_sub)
                        + [f"{row['comparison']} Across Subject"]
                        * len(cross_sub)
                    )
                    d["Task vs. SC"].extend(
                        [f"{row['comparison']}"]
                        * (len(same_sub) + len(cross_sub))
                    )
                    d["FC measure"].extend(
                        [row["measure"]] * (len(same_sub) + len(cross_sub))
                    )
                    d["Similarity"].extend(same_sub + cross_sub)
                    p_values[row["comparison"]] = row[f"p_value_{test}"]
                d = pd.DataFrame(d)
                hue_order = [
                    "RestingState vs. SC Across Subject",
                    "RestingState vs. SC Within Subject",
                    "Raiders vs. SC Across Subject",
                    "Raiders vs. SC Within Subject",
                    "GoodBadUgly vs. SC Across Subject",
                    "GoodBadUgly vs. SC Within Subject",
                    "MonkeyKingdom vs. SC Across Subject",
                    "MonkeyKingdom vs. SC Within Subject",
                    "Mario vs. SC Across Subject",
                    "Mario vs. SC Within Subject",
                ]
                tasks = [
                    "RestingState vs. SC",
                    "Raiders vs. SC",
                    "GoodBadUgly vs. SC",
                    "MonkeyKingdom vs. SC",
                    "Mario vs. SC",
                ]
                rest_colors = sns.color_palette("tab20c")[0]
                movie_1 = sns.color_palette("tab20c")[4]
                movie_2 = sns.color_palette("tab20c")[5]
                movie_3 = sns.color_palette("tab20c")[6]
                mario_colors = sns.color_palette("tab20c")[8]
                color_palette = list(
                    [rest_colors] * 2
                    + [movie_1] * 2
                    + [movie_2] * 2
                    + [movie_3] * 2
                    + [mario_colors] * 2
                )
                fig = plt.figure()
                ax1 = plt.subplot2grid((1, 15), (0, 0), colspan=12)
                ax2 = plt.subplot2grid((1, 15), (0, -3))
                sns.violinplot(
                    x="Similarity",
                    y="Comparison",
                    order=hue_order,
                    hue="Comparison",
                    hue_order=hue_order,
                    palette=color_palette,
                    orient="h",
                    data=d,
                    ax=ax1,
                    split=True,
                    inner=None,
                )
                violin_patches = []
                for i, patch in enumerate(ax1.get_children()):
                    if isinstance(patch, mpl.collections.PolyCollection):
                        violin_patches.append(patch)
                for i, patch in enumerate(violin_patches):
                    if i % 2 == 0:
                        # Loop over the bars
                        patch.set_hatch(hatches[1])
                        patch.set_edgecolor("k")

                legend_elements = [
                    mpl.patches.Patch(
                        facecolor="white",
                        edgecolor="k",
                        label="Across Subject",
                    ),
                    mpl.patches.Patch(
                        facecolor="white",
                        edgecolor="k",
                        label="Within Subject",
                    ),
                ]
                legend_elements[0].set_hatch(hatches[1])
                ax1.legend(
                    framealpha=0,
                    loc="center left",
                    bbox_to_anchor=(1.2, 0.5),
                    handles=legend_elements,
                )

                for i, task in enumerate(tasks):
                    index = abs((i - len(p_values)) - 1)
                    insert_stats(
                        ax2,
                        p_values[task],
                        d["Similarity"],
                        loc=[index + 0.2, index + 0.6],
                        x_n=len(p_values),
                    )
                ax1.set_yticks(np.arange(0, 10, 2) + 0.5, tasks)
                ax1.set_ylabel("Task vs. SC")
                ax1.set_xlabel("Similarity")
                plt.title(f"{cov} {measure}", loc="right", x=-1, y=1.05)
                plot_file = os.path.join(
                    output_dir,
                    f"{cov}_{measure}_{centering}_{test}_box.svg",
                )
                plot_file2 = os.path.join(
                    output_dir,
                    f"{cov}_{measure}_{centering}_{test}_box.png",
                )
                plt.savefig(plot_file, bbox_inches="tight", transparent=True)
                plt.savefig(plot_file2, bbox_inches="tight", transparent=False)
                plt.close()

### fc-sc similarity, barplots ###
hatches = [None, "X", "\\", "/", "|"] * 8
do_hatch = False
# cov estimators
cov_estimators = ["Graphical-Lasso", "Ledoit-Wolf", "Unregularized"]
# connectivity measures for each cov estimator
measures = ["correlation", "partial correlation"]
# what to classify
classify = ["Tasks", "Subjects", "Runs"]
# tasks
tasks = [
    "RestingState",
    "Raiders",
    "GoodBadUgly",
    "MonkeyKingdom",
    "Mario",
]
# load the data
DATA_ROOT = "/storage/store2/work/haggarwa/"
n_parcels = 400
if n_parcels == 400:
    # adding back the overall mean similarity
    results_dir = "fc_similarity_20240411-155121"
    output_dir = os.path.join(DATA_ROOT, "fc-sc_similarity_plots_compcorr")
elif n_parcels == 200:
    # adding back the overall mean similarity
    results_dir = "fc_similarity_20240411-155035"
    output_dir = os.path.join(DATA_ROOT, "fc-sc_similarity_plots_200_compcorr")

similarity_data = pd.read_pickle(
    os.path.join(DATA_ROOT, results_dir, "results.pkl")
)
os.makedirs(output_dir, exist_ok=True)
similarity_data = similarity_data[similarity_data["task2"] == "SC"]

for centering in similarity_data["centering"].unique():
    df = similarity_data[similarity_data["centering"] == centering]
    for how_many in ["all", "three"]:
        if how_many == "all":
            fc_measure_order = [
                "Unregularized correlation",
                "Unregularized partial correlation",
                "Ledoit-Wolf correlation",
                "Ledoit-Wolf partial correlation",
                "Graphical-Lasso correlation",
                "Graphical-Lasso partial correlation",
            ]
        elif how_many == "three":
            fc_measure_order = [
                "Unregularized correlation",
                "Ledoit-Wolf correlation",
                "Graphical-Lasso partial correlation",
            ]

        d = {"FC measure": [], "Similarity": [], "Comparison": []}
        for _, row in df.iterrows():
            corr = row["matrix"].tolist()
            d["Similarity"].extend(corr)
            d["FC measure"].extend([row["measure"]] * len(corr))
            d["Comparison"].extend([row["comparison"]] * len(corr))
        d = pd.DataFrame(d)
        fig, ax = plt.subplots()

        hue_order = [
            "RestingState vs. SC",
            "Raiders vs. SC",
            "GoodBadUgly vs. SC",
            "MonkeyKingdom vs. SC",
            "Mario vs. SC",
        ]
        name = "fc_sc"
        rest_colors = sns.color_palette("tab20c")[0]
        movie_colors = sns.color_palette("tab20c")[4:7]
        mario_colors = sns.color_palette("tab20c")[8]
        color_palette = [rest_colors] + movie_colors + [mario_colors]
        ax = sns.barplot(
            x="Similarity",
            y="FC measure",
            order=fc_measure_order,
            hue="Comparison",
            orient="h",
            hue_order=hue_order,
            palette=color_palette,
            data=d,
            ax=ax,
            # errorbar=None,
        )
        wrap_labels(ax, 20)
        for i, container in enumerate(ax.containers):
            plt.bar_label(
                container,
                fmt="%.2f",
                label_type="edge",
                fontsize="x-small",
                padding=-45,
                weight="bold",
                color="white",
            )
            if do_hatch:
                # Loop over the bars
                for thisbar in container.patches:
                    # Set a different hatch for each bar
                    thisbar.set_hatch(hatches[i])

        legend = ax.legend(
            framealpha=0, loc="center left", bbox_to_anchor=(1, 0.5)
        )
        if do_hatch:
            for i, handle in enumerate(legend.legend_handles):
                handle._hatch = hatches[i]

        plot_file = os.path.join(
            output_dir,
            f"similarity_{name}_{centering}_{how_many}.svg",
        )
        plot_file2 = os.path.join(
            output_dir,
            f"similarity_{name}_{centering}_{how_many}.png",
        )
        if how_many == "three":
            fig.set_size_inches(5, 5)
        else:
            fig.set_size_inches(5, 10)
        plt.savefig(plot_file, bbox_inches="tight", transparent=True)
        plt.savefig(plot_file2, bbox_inches="tight", transparent=False)
        plt.close()
