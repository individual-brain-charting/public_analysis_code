import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ibc_public.connectivity.utils_plot import (
    wrap_labels,
    insert_stats_reliability,
)

sns.set_theme()
sns.set_style("whitegrid")
sns.set_context("talk")

### plot reliability

# cov estimators
cov_estimators = ["Graphical-Lasso", "Ledoit-Wolf", "Unregularized"]
# connectivity measures for each cov estimator
measures = ["correlation", "partial correlation"]
# tasks
tasks = [
    "RestingState",
    "Raiders",
    "GoodBadUgly",
]
# load the data
DATA_ROOT = "/storage/store2/work/haggarwa/"
n_parcels = 200
results_dir = f"reliability_{n_parcels}"
reliability_data = pd.read_pickle(
    os.path.join(DATA_ROOT, results_dir, f"corrs_full_mat_{n_parcels}")
)
p_values = pd.read_pickle(
    os.path.join(DATA_ROOT, results_dir, f"p_vals_{n_parcels}")
)
keep_only = [
    "Unregularized correlation",
    "Ledoit-Wolf correlation",
    "Graphical-Lasso partial correlation",
    "time_series",
]
hue_order = [
    "RestingState",
    "Raiders",
    "GoodBadUgly",
]
rest_colors = [sns.color_palette("tab20c")[0]]
movie_colors = sns.color_palette("tab20c")[4:7]
color_palette = rest_colors + movie_colors
fig = plt.figure()
ax1 = plt.subplot2grid((1, 20), (0, 0), colspan=15)
ax2 = plt.subplot2grid((1, 16), (0, -4))
ax3 = plt.subplot2grid((1, 18), (0, -2))
sns.boxplot(
    x="correlation",
    y="measure",
    hue="task",
    data=reliability_data,
    palette=color_palette,
    orient="h",
    order=keep_only,
    hue_order=hue_order,
    ax=ax1,
)
wrap_labels(ax1, 20)
legend = ax1.legend(framealpha=0, loc="center left", bbox_to_anchor=(1.4, 0.5))
for i, task in enumerate(keep_only):
    p_val = p_values[p_values["measure"] == task].reset_index(drop=True)
    index = abs((i - len(p_values)) - 1)
    for j in range(2):
        if p_val.loc[j]["comp"].split(" ")[2] == "Raiders":
            axis = ax2
            xoff_1 = 0.2
            xoff_2 = 0.1
        else:
            axis = ax3
            xoff_1 = 0.2
            xoff_2 = 0.3
        p = p_val.loc[j]["p_val"]

        insert_stats_reliability(
            axis,
            p,
            1.2,
            loc=[index - xoff_1, index + xoff_2],
            x_n=len(keep_only),
        )

ax1.set_xlabel("Reliability")
ax1.set_ylabel("Measure")
plot_file = os.path.join(
    results_dir,
    "reliability.svg",
)
plot_file2 = os.path.join(
    results_dir,
    "reliability.png",
)
plt.savefig(plot_file, bbox_inches="tight", transparent=True)
plt.savefig(plot_file2, bbox_inches="tight", transparent=False)
plt.close()
