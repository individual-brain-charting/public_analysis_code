import os
import pandas as pd
import seaborn as sns
import numpy as np
from nilearn.connectome import vec_to_sym_matrix
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.svm import LinearSVC
import textwrap


def wrap_labels(ax, width, break_long_words=False):
    labels = []
    for label in ax.get_yticklabels():
        text = label.get_text()
        labels.append(
            textwrap.fill(text, width=width, break_long_words=break_long_words)
        )
    ax.set_yticklabels(labels, rotation=0)


def get_lower_tri_heatmap(
    df,
    figsize=(11, 9),
    cmap="viridis",
    annot=False,
    title=None,
    ticks=None,
    labels=None,
    grid=False,
    output="matrix.png",
    triu=False,
    diag=False,
    tril=False,
    fontsize=20,
):
    mask = np.zeros_like(df)
    mask[np.triu_indices_from(mask)] = triu

    mask[np.tril_indices_from(mask)] = tril

    # Want diagonal elements as well
    mask[np.diag_indices_from(mask)] = diag

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=figsize)

    # Generate a custom diverging colormap
    # cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns_plot = sns.heatmap(
        df,
        mask=mask,
        cmap=cmap,
        # cmap=cmap,
        # vmax=0.008,
        # vmin=0.003,
        # center=0,
        square=True,
        linewidths=0,
        cbar_kws={"shrink": 0.5},
        ax=ax,
        annot=annot,
        fmt="",
        # annot_kws={
        #     "backgroundcolor": "white",
        #     "color": "black",
        #     "bbox": {
        #         "alpha": 0.5,
        #         "color": "white",
        #     },
        # },
    )
    if grid:
        ax.grid(grid, color="black", linewidth=0.5)
    else:
        ax.grid(grid)
    if labels is not None and ticks is None:
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=fontsize)
        ax.set_yticklabels(labels, rotation=0, fontsize=fontsize)
    elif labels is not None and ticks is not None:
        ax.set_xticks(ticks, labels, fontsize=fontsize)
        ax.set_yticks(ticks, labels, fontsize=fontsize)
        ax.tick_params(left=True, bottom=True)
    else:
        ax.set_xticklabels([], fontsize=fontsize)
        ax.set_yticklabels([], fontsize=fontsize)
    if title is not None:
        ax.set_title(title, fontsize=fontsize)

    # save to file
    fig = sns_plot.get_figure()
    fig.savefig(f"{output}.png", bbox_inches="tight")
    fig.savefig(f"{output}.svg", bbox_inches="tight", transparent=True)
    plt.close(fig)


def get_clas_cov_measure(classify, cov_estimators, measures):
    for clas in classify:
        for cov in cov_estimators:
            for measure in measures:
                yield clas, cov, measure


def fit_classifier(clas, cov, measure, func_data, output_dir):
    weights_file = os.path.join(
        output_dir, f"{clas}_{cov} {measure}_weights.npy"
    )
    if os.path.exists(weights_file):
        print(f"skipping {cov} {measure}, already done")
        pass
    else:
        if clas == "Tasks":
            classes = func_data["tasks"].to_numpy(dtype=object)
        elif clas == "Subjects":
            classes = func_data["subject_ids"].to_numpy(dtype=object)
        elif clas == "Runs":
            func_data["run_task"] = (
                func_data["run_labels"] + "_" + func_data["tasks"]
            )
            classes = func_data["run_task"].to_numpy(dtype=object)
        data = np.array(func_data[f"{cov} {measure}"].values.tolist())
        classifier = LinearSVC(max_iter=100000, dual="auto").fit(data, classes)
        np.save(weights_file, classifier.coef_)


def get_network_labels(atlas):
    networks = atlas["labels"].astype("U")
    hemi_network_labels = []
    network_labels = []
    rename_networks = {
        "Vis": "Visual",
        "Cont": "FrontPar",
        "SalVentAttn": "VentAttn",
    }
    for network in networks:
        components = network.split("_")
        components[2] = rename_networks.get(components[2], components[2])
        hemi_network = " ".join(components[1:3])
        hemi_network_labels.append(hemi_network)
        network_labels.append(components[2])

    return hemi_network_labels, network_labels


def _load_transform_weights(
    clas, cov, measure, transform, weight_dir, n_parcels
):
    try:
        weights = np.load(
            os.path.join(weight_dir, f"{clas}_{cov} {measure}_weights.npy")
        )
    except FileNotFoundError:
        print(f"skipping {clas} {cov} {measure}")
        raise FileNotFoundError

    if transform == "maxratio":
        weights = np.abs(weights)
        max_weights = np.max(weights, axis=0)
        mask = weights.max(axis=0, keepdims=True) == weights
        mean_other_values = np.mean(weights[~mask], axis=0)
        weights = max_weights / mean_other_values
    elif transform == "l2":
        weights = np.sqrt(np.sum(weights**2, axis=0))

    weight_mat = vec_to_sym_matrix(weights, diagonal=np.ones(n_parcels))

    return weight_mat


def _average_over_networks(
    encoded_labels,
    unique_labels,
    clas,
    cov,
    measure,
    transform,
    weight_dir,
    n_parcels,
):
    network_pair_weights = np.zeros((len(unique_labels), len(unique_labels)))
    # get network pair weights
    for network_i in unique_labels:
        index_i = np.where(encoded_labels == network_i)[0]
        for network_j in unique_labels:
            index_j = np.where(encoded_labels == network_j)[0]
            weight_mat = _load_transform_weights(
                clas, cov, measure, transform, weight_dir, n_parcels
            )
            weight_mat[np.triu_indices_from(weight_mat)] = np.nan
            network_pair_weight = np.nanmean(
                weight_mat[np.ix_(index_i, index_j)]
            )
            network_pair_weights[network_i][network_j] = network_pair_weight

    return network_pair_weights


def plot_network_weight_matrix(
    clas,
    cov,
    measure,
    atlas,
    output_dir,
    weight_dir,
    n_parcels,
    labels_fmt="hemi network",
    transform="maxratio",
    fontsize=20,
):
    if labels_fmt == "hemi network":
        labels = get_network_labels(atlas)[0]
    elif labels_fmt == "network":
        labels = get_network_labels(atlas)[1]

    le = preprocessing.LabelEncoder()
    encoded_labels = le.fit_transform(labels)

    unique_labels = np.unique(encoded_labels)
    network_pair_weights = _average_over_networks(
        encoded_labels,
        unique_labels,
        clas,
        cov,
        measure,
        transform,
        weight_dir,
        n_parcels,
    )
    # plot network wise average weights
    get_lower_tri_heatmap(
        network_pair_weights,
        figsize=(5, 5),
        cmap="viridis",
        labels=le.inverse_transform(unique_labels),
        output=os.path.join(
            output_dir, f"{clas}_{cov}_{measure}_network_weights"
        ),
        triu=True,
        title=f"{cov} {measure}",
        fontsize=fontsize,
    )


def plot_full_weight_matrix(
    clas,
    cov,
    measure,
    atlas,
    output_dir,
    weight_dir,
    n_parcels,
    transform="maxratio",
    fontsize=20,
):
    weight_mat = _load_transform_weights(
        clas,
        cov,
        measure,
        transform=transform,
        weight_dir=weight_dir,
        n_parcels=n_parcels,
    )

    hemi_network_labels = get_network_labels(atlas)[0]

    # get tick locations
    ticks = []
    for i, label in enumerate(hemi_network_labels):
        if label != hemi_network_labels[i - 1]:
            ticks.append(i)

    # keep unique labels
    _, uniq_idx = np.unique(hemi_network_labels, return_index=True)
    hemi_network_labels = np.array(hemi_network_labels)[np.sort(uniq_idx)]

    # plot full matrix
    get_lower_tri_heatmap(
        weight_mat,
        cmap="viridis",
        labels=hemi_network_labels,
        output=os.path.join(output_dir, f"{clas}_{cov}_{measure}_all_weights"),
        triu=True,
        diag=True,
        title=f"{cov} {measure}",
        ticks=ticks,
        grid=True,
        fontsize=fontsize,
    )


def insert_stats(ax, p_val, data, loc=[], h=0.15, y_offset=0, x_n=3):
    """
    Insert p-values from statistical tests into boxplots.
    """
    max_y = data.max()
    h = h / 100 * max_y
    y_offset = y_offset / 100 * max_y
    x1, x2 = loc[0], loc[1]
    y = max_y + h + y_offset
    ax.plot([y, y], [x1, x2], lw=2, c="0.25")
    if p_val < 0.0001:
        text = "****"
    if p_val < 0.001:
        text = "***"
    elif p_val < 0.01:
        text = "**"
    elif p_val < 0.05:
        text = "*"
    else:
        text = "ns"
    ax.text(
        y + 3.5,
        ((x1 + x2) * 0.5) - 0.15,
        f"{text}",
        ha="center",
        va="bottom",
        color="0.25",
    )
    ax.set_xticks([*range(0, x_n)])
    ax.axis("off")


def mean_connectivity(data, tasks, cov_estimators, measures):
    """Average connectivity across runs for each subject and task.

    Parameters
    ----------
    data : pandas dataframe
        a dataframe with flattened connectivity matrices with a
        column for each fc measure (created by joining covariance
        estimator and the measure with a space), a column for
        the task, and a column for the subject
    tasks : list
        a list of tasks to average connectivity across runs
    cov_estimators : list
        a list of covariance estimators
    measures : list
        a list of connectivity measures estimated by each covariance

    Returns
    -------
    pandas dataframe
        a dataframe with the average connectivity for each subject,
        task, and measure in long format
    """
    av_connectivity = []
    for task in tasks:
        task_data = data[data["tasks"] == task]
        task_subjects = task_data["subject_ids"].unique()
        for sub in task_subjects:
            df = task_data[task_data["subject_ids"] == sub]
            for cov in cov_estimators:
                for measure in measures:
                    connectivity = df[cov + " " + measure].tolist()
                    connectivity = np.array(connectivity)
                    connectivity = connectivity.mean(axis=0)
                    av_connectivity.append(
                        {
                            "task": task,
                            "subject": sub,
                            "connectivity": connectivity,
                            "measure": cov + " " + measure,
                        }
                    )

    return pd.DataFrame(av_connectivity)


def insert_stats_reliability(
    ax,
    p_val,
    data,
    loc=[],
    h=0.15,
    y_offset=0,
    x_n=3,
):
    """
    Insert p-values from statistical tests into boxplots.
    """
    h = h / 100 * data
    y_offset = y_offset / 100 * data
    x1, x2 = loc[0], loc[1]
    y = data + h + y_offset
    ax.plot([y, y], [x1, x2], lw=2, c="0.25")
    if p_val < 0.0001:
        text = "****"
    if p_val < 0.001:
        text = "***"
    elif p_val < 0.01:
        text = "**"
    elif p_val < 0.05:
        text = "*"
    else:
        text = "ns"
    ax.text(
        y + 3.5,
        ((x1 + x2) * 0.5) - 0.15,
        f"{text}",
        ha="center",
        va="bottom",
        color="0.25",
    )
    ax.set_xticks([*range(0, x_n)])
    ax.axis("off")
