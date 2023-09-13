import os
import time

import numpy as np
import pandas as pd
import seaborn as sns
from ibc_public import utils_connectivity as fc
from nilearn import datasets
from scipy import stats
import matplotlib.pyplot as plt

from joblib import Parallel, delayed

sns.set_theme(context="talk", style="whitegrid")


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


def similarity(connectivity_matrices, subjects, mean_center=True, z_transform=True):
    """Calculate pearson correlation between two connectivity matrices

    Parameters
    ----------
    connectivity_matrices : list
        a list where each element is a connectivity matrix
    subjects : list
        a list where each element is a list of subjects
    mean_center : bool
        whether to mean center the correlation matrix
    z_transform : bool
        whether to z-transform the correlation matrix

    Returns
    -------
    similarity_mat : pandas dataframe
        a dataframe with pearson correlation between pairs of subjects for
        given pair of connectivity matrices
    similarity_mat_centered : pandas dataframe
        a dataframe with pearson correlation between pairs of subjects for
        given pair of connectivity matrices after double mean centering
    """
    task1_conn = connectivity_matrices[0]
    task2_conn = connectivity_matrices[1]
    task1_subs = subjects[0]
    task2_subs = subjects[1]

    def corr2_coeff(A, B):
        # Rowwise mean of input arrays & subtract from input arrays themeselves
        A_mA = A - A.mean(1)[:, None]
        B_mB = B - B.mean(1)[:, None]

        # Sum of squares across rows
        ssA = (A_mA**2).sum(1)
        ssB = (B_mB**2).sum(1)

        # Finally get corr coeff
        return np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None], ssB[None]))

    similarity_mat = corr2_coeff(task1_conn, task2_conn)
    # similarity_mat = np.hsplit(np.vsplit(similarity_mat, 2)[0], 2)[1]
    if mean_center:
        similarity_mat_centered = similarity_mat - similarity_mat.mean(axis=0)
        similarity_mat_centered = (
            similarity_mat_centered.T - similarity_mat_centered.mean(axis=1)
        ).T

    if z_transform:
        similarity_mat_zscored = stats.zscore(similarity_mat, axis=0)
        similarity_mat_zscored = stats.zscore(similarity_mat_zscored, axis=1)

    similarity_mat = pd.DataFrame(
        similarity_mat, columns=task2_subs, index=task1_subs
    )
    similarity_mat_centered = pd.DataFrame(
        similarity_mat_centered, columns=task2_subs, index=task1_subs
    )
    similarity_mat_zscored = pd.DataFrame(similarity_mat_zscored, columns=task2_subs, index=task1_subs)

    return similarity_mat, similarity_mat_centered, similarity_mat_zscored


def symmetrize(a):
    """Symmetrize a matrix by dropping rows and columns that are not
    shared between the rows and columns.

    Parameters
    ----------
    a : pandas dataframe
        a non-symmetrical matrix in pandas dataframe format. Already symmetrical
        matrices will not be affected.

    Returns
    -------
    a : numpy array
        symmetrical matrix in numpy array format
    columns : numpy array
        columns of the symmetrical matrix
    mask : tuple
        a tuple of two boolean arrays indicating which indices of rows and columns
        that were kept in the symmetrization process
    """
    extra_sub1 = np.setdiff1d(a.index.values, a.columns.values)
    extra_sub2 = np.setdiff1d(a.columns.values, a.index.values)
    sub1_mask = np.isin(a.index.values, a.columns.values)
    sub2_mask = np.isin(a.columns.values, a.index.values)
    a = a.drop(index=extra_sub1, errors="ignore")
    a = a.drop(columns=extra_sub2, errors="ignore")
    return a.to_numpy(), a.columns.values, (sub1_mask, sub2_mask)


def samevcross_test(correlation):
    """Test whether the diagonal of a matrix is greater than the off-diagonal
    elements using a one-sided t-test.

    Parameters
    ----------
    correlation : numpy array
        a symmetrical matrix

    Returns
    -------
    p_value : float
        the p-value from the t-test
    """
    same_sub_corr = np.diagonal(correlation, offset=0).tolist()
    upper_tri = correlation[np.triu_indices_from(correlation, k=1)].tolist()
    lower_tri = correlation[np.tril_indices_from(correlation, k=-1)].tolist()
    cross_sub_corr = upper_tri + lower_tri
    t_test = stats.ttest_ind(
        same_sub_corr, cross_sub_corr, alternative="greater"
    )
    p_value = t_test[1]

    return p_value


def _mask(df, task, cov, measure):
    """Create a boolean mask for a dataframe by task, covariance estimator, and measure."""
    if task == "SC":
        return df["measure"] == "SC"
    else:
        return (df["task"] == task) & (df["measure"] == cov + " " + measure)


def _filter_connectivity(data, task, cov, measure):
    """Keep connectivity matrices by task, covariance estimator, and measure, by applying a boolean mask."""
    connectivity = data[_mask(data, task, cov, measure)][
        "connectivity"
    ].tolist()
    return np.array(connectivity)


def _get_subjects(data, task, cov, measure):
    """Keep subject labels for a given task, covariance estimator, and measure."""
    return data[_mask(data, task, cov, measure)]["subject"].tolist()


def get_similarity(all_connectivity, task_pair, cov, measure):
    """Get similarity between two tasks for a given covariance estimator and measure.

    Parameters
    ----------
    all_connectivity : pandas dataframe
        a dataframe with the average connectivity for each subject, task, and measure
    task_pair : tuple
        a tuple of two tasks to compare
    cov : str
        covariance estimator
    measure : str
        connectivity measure

    Returns
    -------
    result : dict
        a dictionary with the results of the similarity test
    """
    task1, task2 = task_pair
    task_pair_connectivity = []
    task_pair_subjects = []
    for task in task_pair:
        task_pair_connectivity.append(
            _filter_connectivity(all_connectivity, task, cov, measure)
        )
        task_pair_subjects.append(
            _get_subjects(all_connectivity, task, cov, measure)
        )
    similarity_mat, similarity_centered, similarity_z = similarity(
        [*task_pair_connectivity],
        [*task_pair_subjects],
    )
    result = []
    for i, matrix in enumerate([similarity_mat, similarity_centered, similarity_z]):
        matrix, kept_subs, kept_ind = symmetrize(matrix)
        p_value = samevcross_test(matrix)
        if i == 0:
            centering = "uncentered"
        elif i == 1:
            centering = "centered"
        else:
            centering = "z-scored"

        result_ = {
            "task1": task1,
            "task2": task2,
            "measure": cov + " " + measure,
            "centering": centering,
            "p_value": p_value,
            "matrix": matrix.flatten(),
            "kept_subjects": kept_subs,
            "comparison": task1 + " vs. " + task2,
        }
        result.append(result_)

    return result


def plot_barplot(stats_df, out_dir):
    barplot_dir = os.path.join(out_dir, "barplot")
    os.makedirs(barplot_dir, exist_ok=True)
    for centering in stats_df["centering"].unique():
        df = stats_df[stats_df["centering"] == centering]
        sc_fc_mask = df["comparison"].str.contains("SC")
        fc_fc = df[~sc_fc_mask]
        sc_sc = df[sc_fc_mask]
        fc_measure_order = [
            "LedoitWolf correlation",
            "LedoitWolf partial correlation",
            "GLC correlation",
            "GLC partial correlation",
        ]
        for i, df in enumerate([fc_fc, sc_sc]):
            d = {"FC measure": [], "Similarity": [], "Comparison": []}
            for _, row in df.iterrows():
                corr = row["matrix"].tolist()
                d["Similarity"].extend(corr)
                d["FC measure"].extend([row["measure"]] * len(corr))
                d["Comparison"].extend([row["comparison"]] * len(corr))
            d = pd.DataFrame(d)
            fig, ax = plt.subplots()
            if i == 0:
                hue_order = [
                    "RestingState vs. Raiders",
                    "RestingState vs. GoodBadUgly",
                    "RestingState vs. MonkeyKingdom",
                    "Raiders vs. GoodBadUgly",
                    "Raiders vs. MonkeyKingdom",
                    "GoodBadUgly vs. MonkeyKingdom",
                ]
                name = "fc_fc"
                color_palette = sns.color_palette()[1:]
            else:
                hue_order = [
                    "RestingState vs. SC",
                    "Raiders vs. SC",
                    "GoodBadUgly vs. SC",
                    "MonkeyKingdom vs. SC",
                ]
                name = "fc_sc"
                color_palette = sns.color_palette()
            sns.barplot(
                x="Similarity",
                y="FC measure",
                order=fc_measure_order,
                hue="Comparison",
                orient="h",
                hue_order=hue_order,
                palette=color_palette,
                data=d,
                ax=ax,
            )
            ax.legend(framealpha=0, loc="center left", bbox_to_anchor=(1, 0.5))
            plot_file = os.path.join(
                barplot_dir,
                f"similarity_{name}_{centering}.png",
            )
            ax.xaxis.set_tick_params(rotation=90)
            plt.savefig(plot_file, bbox_inches="tight", transparent=False)
            plt.close()


def insert_stats(ax, p_val, data, loc=[], h=0, y_offset=0, x_n=3):
    """
    Insert p-values from statistical tests into boxplots.
    """
    max_y = data.max()
    h = h / 100 * max_y
    y_offset = y_offset / 100 * max_y
    x1, x2 = loc[0], loc[1]
    y = max_y + h + y_offset
    ax.plot([y, y + h, y + h, y], [x1, x1, x2, x2], lw=1, c="0.25")
    if p_val < 0.0001:
        text = f"****"
    if p_val < 0.001:
        text = f"***"
    elif p_val < 0.01:
        text = f"**"
    elif p_val < 0.05:
        text = f"*"
    else:
        text = f"ns"
    ax.text(
        y + 2.5, ((x1 + x2) * 0.5)-0.15, text, ha="center", va="bottom", color="0.25"
    )
    ax.set_xticks([*range(0, x_n)])
    ax.axis("off")


def plot_boxplot(stats_df, out_dir):
    boxplot_dir = os.path.join(out_dir, "boxplot")
    os.makedirs(boxplot_dir, exist_ok=True)
    fc_measure_order = [
        "LedoitWolf correlation",
        "LedoitWolf partial correlation",
        "GLC correlation",
        "GLC partial correlation",
    ]
    for centering in stats_df["centering"].unique():
        for comparison in stats_df["comparison"].unique():
            df = stats_df[
                (stats_df["centering"] == centering)
                & (stats_df["comparison"] == comparison)
            ]
            d = {
                "Comparison": [],
                "FC measure": [],
                "Similarity": [],
            }
            p_values = []
            for _, row in df.iterrows():
                n_subs = len(row["kept_subjects"])
                corr = row["matrix"].reshape(n_subs, n_subs)
                same_sub = np.diagonal(corr, offset=0).tolist()
                upper_tri = corr[np.triu_indices_from(corr, k=1)].tolist()
                lower_tri = corr[np.tril_indices_from(corr, k=-1)].tolist()
                cross_sub = upper_tri + lower_tri
                d["Comparison"].extend(
                    ["Within Subject"] * len(same_sub)
                    + ["Across Subject"] * len(cross_sub)
                )
                d["FC measure"].extend(
                    [row["measure"]] * (len(same_sub) + len(cross_sub))
                )
                d["Similarity"].extend(same_sub + cross_sub)
                p_values.append((row["p_value"], row["measure"]))
            d = pd.DataFrame(d)
            color_map = {
                "RestingState vs. Raiders": 1,
                "RestingState vs. GoodBadUgly": 2,
                "RestingState vs. MonkeyKingdom": 3,
                "Raiders vs. GoodBadUgly": 4,
                "Raiders vs. MonkeyKingdom": 5,
                "GoodBadUgly vs. MonkeyKingdom": 6,
                "RestingState vs. SC": 0,
                "Raiders vs. SC": 1,
                "GoodBadUgly vs. SC": 2,
                "MonkeyKingdom vs. SC": 3,
            }
            color_palette = [
                sns.color_palette("pastel")[color_map[comparison]],
                sns.color_palette()[color_map[comparison]],
            ]
            fig = plt.figure()
            ax1 = plt.subplot2grid((1, 15), (0, 0), colspan=12)
            ax2 = plt.subplot2grid((1, 15), (0, -3))
            sns.boxplot(
                x="Similarity",
                y="FC measure",
                order=fc_measure_order,
                hue="Comparison",
                hue_order=["Across Subject", "Within Subject"],
                palette=color_palette,
                orient="h",
                data=d,
                ax=ax1,
                fliersize=0,
            )
            for i, p in enumerate(p_values):
                index = abs((i - len(p_values)) - 1)
                insert_stats(
                    ax2,
                    p[0],
                    d["Similarity"],
                    loc=[index + 0.2, index + 0.6],
                    x_n=len(p_values),
                )
            ax1.legend(
                framealpha=0, loc="center left", bbox_to_anchor=(1.2, 0.5)
            )
            ax1.xaxis.set_tick_params(rotation=90)
            plot_file = os.path.join(
                boxplot_dir,
                f"{comparison}_{centering}_box.png",
            )
            plt.savefig(plot_file, bbox_inches="tight", transparent=False)
            plt.close()


if __name__ == "__main__":
    cache = DATA_ROOT = "/storage/store/work/haggarwa/"
    output_dir = f"fc_similarity_{time.strftime('%Y%m%d-%H%M%S')}"
    output_dir = os.path.join(DATA_ROOT, output_dir)
    os.makedirs(output_dir, exist_ok=True)
    calculate_connectivity = False
    fc_data_path = os.path.join(cache, "connectomes")
    sc_data_path = os.path.join(cache, "sc_data")
    # number of jobs to run in parallel
    n_jobs = 15
    # tasks
    tasks = ["RestingState", "Raiders", "GoodBadUgly", "MonkeyKingdom"]
    # cov estimators
    cov_estimators = ["GLC", "LedoitWolf"]
    # connectivity measures for each cov estimator
    measures = ["correlation", "partial correlation"]

    task_pairs = [
        ("RestingState", "Raiders"),
        ("RestingState", "GoodBadUgly"),
        ("RestingState", "MonkeyKingdom"),
        ("Raiders", "GoodBadUgly"),
        ("Raiders", "MonkeyKingdom"),
        ("GoodBadUgly", "MonkeyKingdom"),
        ("RestingState", "SC"),
        ("Raiders", "SC"),
        ("GoodBadUgly", "SC"),
        ("MonkeyKingdom", "SC"),
    ]

    def all_combinations(task_pairs, cov_estimators, measures):
        for task_pair in task_pairs:
            for cov in cov_estimators:
                for measure in measures:
                    yield task_pair, cov, measure

    if calculate_connectivity == True:
        # get the atlas
        atlas = datasets.fetch_atlas_schaefer_2018(
            data_dir=cache, resolution_mm=2, n_rois=400
        )
        # use the atlas to extract time series for each task in parallel
        # get_time_series returns a dataframe with
        # the time series for each task, consisting of runs x subjects
        print("Time series extraction...")
        data = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(fc.get_time_series)(task, atlas, cache) for task in tasks
        )
        # concatenate all the dataframes so we have a single dataframe
        # with the time series from all tasks
        data = pd.concat(data)
        # estimate the connectivity matrices for each cov estimator in parallel
        # get_connectomes returns a dataframe with two columns each corresponding
        # to the partial correlation and correlation connectome from each cov estimator
        print("Connectivity estimation...")
        data = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(fc.get_connectomes)(cov, data, n_jobs)
            for cov in cov_estimators
        )
        # concatenate the dataframes so we have a single dataframe
        # with the connectomes from all cov estimators
        cols_to_use = data[0].columns.difference(data[1].columns)
        data = pd.concat([data[1], data[0][cols_to_use]], axis=1)
        data.reset_index(inplace=True, drop=True)
    else:
        data = pd.read_pickle(fc_data_path)
        sc_data = pd.read_pickle(sc_data_path)
    all_connectivity = mean_connectivity(data, tasks, cov_estimators, measures)
    all_connectivity = pd.concat([all_connectivity, sc_data], axis=0)

    results = Parallel(n_jobs=n_jobs, verbose=2, backend="multiprocessing")(
        delayed(get_similarity)(all_connectivity, task_pair, cov, measure)
        for task_pair, cov, measure in all_combinations(
            task_pairs, cov_estimators, measures
        )
    )

    results = [item for sublist in results for item in sublist]
    results = pd.DataFrame(results)
    results.to_pickle(os.path.join(output_dir, "results.pkl"))

    plot_barplot(results, output_dir)
    plot_boxplot(results, output_dir)
