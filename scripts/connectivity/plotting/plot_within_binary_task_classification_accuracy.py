import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ibc_public.connectivity.utils_plot import wrap_labels

sns.set_theme()
sns.set_style("whitegrid")
sns.set_context("talk")

### within or binary task classification accuracies ###
hatches = [None, "X", "\\", "/", "|"] * 8

cache = DATA_ROOT = "/storage/store/work/haggarwa/"
DATA_ROOT2 = "/storage/store2/work/haggarwa/"
n_parcels = 400

do_hatch = False

if n_parcels == 400:
    # with compcorr and fixed resting state confounds
    results_dir = "fc_withintask_classification_400_20240120-154926"
    results_pkl = os.path.join(DATA_ROOT2, results_dir, "all_results.pkl")
    output_dir = "within_task_classification_plots_compcorr"
elif n_parcels == 200:
    # with compcorr and fixed resting state confounds
    results_dir = "fc_withintask_classification_200_20240118-143124"
    results_pkl = os.path.join(DATA_ROOT2, results_dir, "all_results.pkl")
    output_dir = "within_task_classification_plots_200_compcorr"
output_dir = os.path.join(DATA_ROOT2, output_dir)
os.makedirs(output_dir, exist_ok=True)
df = pd.read_pickle(results_pkl)
# cov estimators
cov_estimators = ["Graphical-Lasso", "Ledoit-Wolf", "Unregularized"]
# connectivity measures for each cov estimator
measures = ["correlation", "partial correlation"]
# what to classify
classify = ["Tasks", "Subjects", "Runs"]
# tasks
tasks = ["RestingState", "Raiders", "GoodBadUgly", "MonkeyKingdom", "Mario"]
movies = tasks[1:4]

df[df.select_dtypes(include=["number"]).columns] *= 100

for clas in classify:
    df_ = df[df["classes"] == clas]
    df_.reset_index(inplace=True, drop=True)
    if clas == "Runs":
        print(len(df_))
        print(df_["task_label"].unique())
        df_ = df_[df_["task_label"].isin(movies)]
        print(len(df_))
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
            eval_metric1 = "LinearSVC_auc"
            eval_metric2 = "Dummy_auc"
            legend_cutoff = 10
            palette_init = 1
            xlabel = "AUC"
            title = ""
            rest_colors = sns.color_palette("tab20c")[0:4]
            movie_colors = sns.color_palette("tab20c")[4:7]
            mario_colors = sns.color_palette("tab20c")[8:11]
            color_palette = rest_colors + movie_colors + mario_colors
            bar_label_color = "white"
            bar_label_weight = "bold"
        elif clas == "Runs":
            eval_metric1 = "balanced_accuracy"
            eval_metric2 = "dummy_balanced_accuracy"
            legend_cutoff = 3
            palette_init = 1
            xlabel = "Accuracy"
            title = ""
            movie_colors = sns.color_palette("tab20c")[4:7]
            color_palette = movie_colors
            bar_label_color = "white"
            bar_label_weight = "bold"
        else:
            eval_metric1 = "balanced_accuracy"
            eval_metric2 = "dummy_balanced_accuracy"
            legend_cutoff = 5
            palette_init = 0
            xlabel = "Accuracy"
            title = ""
            rest_colors = sns.color_palette("tab20c")[0]
            movie_colors = sns.color_palette("tab20c")[4:7]
            mario_colors = sns.color_palette("tab20c")[8]
            color_palette = [rest_colors] + movie_colors + [mario_colors]
            bar_label_color = "white"
            bar_label_weight = "bold"
        ax_score = sns.barplot(
            y="connectivity",
            x=eval_metric1,
            data=df_,
            orient="h",
            hue="task_label",
            palette=color_palette,
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
                padding=-45,
                weight=bar_label_weight,
                color=bar_label_color,
            )
            if do_hatch:
                # Loop over the bars
                for thisbar in container.patches:
                    # Set a different hatch for each bar
                    thisbar.set_hatch(hatches[i])
        ax_chance = sns.barplot(
            y="connectivity",
            x=eval_metric2,
            data=df_,
            orient="h",
            hue="task_label",
            palette=sns.color_palette("pastel")[7:],
            order=order,
            errorbar=None,
            facecolor=(0.8, 0.8, 0.8, 1),
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
                text.set_visible(False)
                handle.set_visible(False)
            else:
                if do_hatch:
                    handle._hatch = hatches[i]

            if i == legend_cutoff:
                text.set_text("Chance-level")

        legend.set_title("Task")
        fig = plt.gcf()
        if clas == "Tasks":
            fig.set_size_inches(6, 10)
        else:
            fig.set_size_inches(6, 6)

        plt.savefig(
            os.path.join(output_dir, f"{clas}_classification_{how_many}.png"),
            bbox_inches="tight",
        )
        plt.savefig(
            os.path.join(output_dir, f"{clas}_classification_{how_many}.svg"),
            bbox_inches="tight",
            transparent=True,
        )
        plt.close()
