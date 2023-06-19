from ibc_public.utils_connectivity import (
    get_ses_modality,
    get_all_subject_connectivity,
)
import pandas as pd
import numpy as np
from scipy import stats
from nilearn.connectome import vec_to_sym_matrix, sym_matrix_to_vec
import os
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import seaborn as sns

# calculated measures of functional connectivity
FUNC_CONN_MEASURES = [
    "Pearsons_corr",
    "GraphicalLassoCV_corr",
    "GraphicalLassoCV_partcorr",
    "GroupSparseCovarianceCV_corr",
    "GroupSparseCovarianceCV_partcorr",
]
# calculated measures of structural connectivity
STRUC_CONN_MEASURES = [
    "_connectome_schaefer400_MNI152_siftweighted",
]


def select_rois(connectivities, n_rois, selection_criterion):
    selected_connectivities = []
    for connectivity in connectivities:
        connectivity_mat = vec_to_sym_matrix(connectivity, np.ones(n_rois))
        if selection_criterion == "intra":
            lh = connectivity_mat[: n_rois // 2, : n_rois // 2]
            lh = sym_matrix_to_vec(lh, discard_diagonal=True)
            rh = connectivity_mat[n_rois // 2 :, n_rois // 2 :]
            rh = sym_matrix_to_vec(rh, discard_diagonal=True)
            lh_rh = np.concatenate((lh, rh))
        elif selection_criterion == "inter":
            lh_rh = connectivity_mat[: n_rois // 2, n_rois // 2 :]
            lh_rh = sym_matrix_to_vec(lh_rh, discard_diagonal=False)
        else:
            lh_rh = sym_matrix_to_vec(connectivity_mat, discard_diagonal=True)
        selected_connectivities.append(lh_rh)

    return selected_connectivities


def symmetrize(a):
    extra_sub1 = np.setdiff1d(a.index.values, a.columns.values)
    extra_sub2 = np.setdiff1d(a.columns.values, a.index.values)
    a = a.drop(index=extra_sub1, errors="ignore")
    a = a.drop(columns=extra_sub2, errors="ignore")
    return a.to_numpy(), a.columns.values


def calculate_pearson_corr(connectivity_matrices, subjects, mean_center=True):
    """Calculate pearson correlation between two connectivity matrices

    Parameters
    ----------
    connectivity_matrices : list
        a list where each element is a connectivity matrix
    subjects : list
        a list where each element is a list of subjects

    Returns
    -------
    similarity_df : pandas dataframe
        a dataframe with pearson correlation between two connectivity matrices
    """
    task1_conn = np.asarray(connectivity_matrices[0])
    task2_conn = np.asarray(connectivity_matrices[1])
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
        similarity_mat = similarity_mat_centered

    return pd.DataFrame(similarity_mat, columns=task2_subs, index=task1_subs)


def samevcross_test(correlation):
    same_sub_corr = np.diagonal(correlation, offset=0).tolist()
    upper_tri = correlation[np.triu_indices_from(correlation, k=1)].tolist()
    lower_tri = correlation[np.tril_indices_from(correlation, k=-1)].tolist()
    cross_sub_corr = upper_tri + lower_tri
    t_test = stats.ttest_ind(
        same_sub_corr, cross_sub_corr, alternative="greater"
    )
    p_value = t_test[1]

    return p_value


def get_subjects(task1_sub_ses, task2_sub_ses):
    subjects = [task1_sub_ses.keys(), task2_sub_ses.keys()]
    return subjects


def get_conn_measures(task1_modality, task2_modality):
    if task1_modality == task2_modality == "functional":
        task1_conn_measures = task2_conn_measures = FUNC_CONN_MEASURES
    else:
        if task1_modality == task2_modality == "structural":
            task1_conn_measures = task2_conn_measures = STRUC_CONN_MEASURES
        elif task1_modality == "functional":
            task1_conn_measures = FUNC_CONN_MEASURES
            task2_conn_measures = STRUC_CONN_MEASURES
        elif task1_modality == "structural":
            task1_conn_measures = STRUC_CONN_MEASURES
            task2_conn_measures = FUNC_CONN_MEASURES
        else:
            raise ValueError("Invalid task modality.")
    return task1_conn_measures, task2_conn_measures


def generate_name(task1, task2, task1_conn_measure, task2_conn_measure, rois):
    return f"{task1}_{task1_conn_measure}_{task2}_{task2_conn_measure}_{rois}-rois"


def get_subject_connectivity(sub_ses, modality, conn_measure, data_root):
    if conn_measure == "Pearsons_corr":
        average_runs = True
    else:
        average_runs = False
    return get_all_subject_connectivity(
        sub_ses, modality, conn_measure, data_root, average_runs
    )


def get_task_pair_stats(
    task1_conn_measure,
    task2_conn_measure,
    task1_sub_ses,
    task1_modality,
    task2_sub_ses,
    task2_modality,
    task1,
    task2,
    rois,
    data_root,
    n_rois,
    subjects,
    results,
):
    task1_conn = get_subject_connectivity(
        task1_sub_ses, task1_modality, task1_conn_measure, data_root
    )
    task1_conn = select_rois(task1_conn, n_rois, rois)
    task2_conn = get_subject_connectivity(
        task2_sub_ses, task2_modality, task2_conn_measure, data_root
    )
    task2_conn = select_rois(task2_conn, n_rois, rois)
    correlation = calculate_pearson_corr([task1_conn, task2_conn], subjects)
    correlation, kept_subs = symmetrize(correlation)
    p_value = samevcross_test(correlation)
    results["correlation"].append(correlation.flatten())
    results["p_value"].append(p_value)
    results["kept_subjects"].append(kept_subs)
    results["task1"].append(task1)
    results["task2"].append(task2)
    results["task1_modality"].append(task1_modality)
    results["task2_modality"].append(task2_modality)
    results["task1_measure"].append(task1_conn_measure)
    results["task2_measure"].append(task2_conn_measure)
    results["selected_rois"].append(rois)
    return results


def plot_correlation(stats_df, out_dir):
    correlation_dir = os.path.join(out_dir, "correlation")
    os.makedirs(correlation_dir, exist_ok=True)
    for _, row in stats_df.iterrows():
        n_subs = len(row["kept_subjects"])
        corr = row["correlation"].reshape(n_subs, n_subs)
        fig, ax = plt.subplots()
        pos = ax.matshow(corr, cmap=plt.cm.Blues)
        ax.set_xticks(np.arange(n_subs))
        ax.xaxis.tick_bottom()
        ax.set_yticks(np.arange(n_subs))
        ax.set_xticklabels(row["kept_subjects"], rotation=45)
        ax.set_yticklabels(row["kept_subjects"])
        ax.set_xlabel(row["task2_modality"])
        ax.set_ylabel(row["task1_modality"])
        fig.colorbar(pos, ax=ax)
        ax.set_title(
            f"Correlation between FC from {row['task1']}\nas {row['task1_measure']}\nand SC from {row['task2']}, {row['selected_rois']} rois"
        )
        plot_file = os.path.join(
            correlation_dir,
            f"{row['task1']}_{row['task1_measure']}_{row['task2']}_{row['selected_rois']}_corr.png",
        )
        plt.savefig(plot_file, bbox_inches="tight")
        plt.close()


def plot_boxplot(stats_df, out_dir):
    boxplot_dir = os.path.join(out_dir, "boxplot")
    os.makedirs(boxplot_dir, exist_ok=True)
    d = {"Comparison": [], "FC measure": [], "Correlation FC v SC": []}
    p_values = []
    for _, row in stats_df.iterrows():
        n_subs = len(row["kept_subjects"])
        corr = row["correlation"].reshape(n_subs, n_subs)
        same_sub = np.diagonal(corr, offset=0).tolist()
        upper_tri = corr[np.triu_indices_from(corr, k=1)].tolist()
        lower_tri = corr[np.tril_indices_from(corr, k=-1)].tolist()
        cross_sub = upper_tri + lower_tri
        d["Comparison"].extend(
            ["Within Subject"] * len(same_sub)
            + ["Across Subject"] * len(cross_sub)
        )
        d["FC measure"].extend(
            [row["task1_measure"]] * (len(same_sub) + len(cross_sub))
        )
        d["Correlation FC v SC"].extend(same_sub + cross_sub)
        p_values.append(row["p_value"])
    df = pd.DataFrame(d)
    fig, ax = plt.subplots()
    sns.boxplot(
        x="Correlation FC v SC",
        y="FC measure",
        hue="Comparison",
        data=df,
        ax=ax,
        orient="h",
        fliersize=0,
    )
    for i, p in enumerate(p_values):
        if p > 0.05:
            signif = "ns"
        elif p <= 0.0001:
            signif = "p ≤ 0.0001"
        elif p <= 0.001:
            signif = "p ≤ 0.001"
        elif p <= 0.01:
            signif = "p ≤ 0.01"
        elif p <= 0.05:
            signif = "p ≤ 0.05"
        ax.text(
            df["Correlation FC v SC"].min(),
            i,
            signif,
            fontsize=10,
            bbox=dict(facecolor="white", alpha=1),
        )
    ax.set_title(
        f"FC-SC correlation within vs across subjects, {row['selected_rois']} rois"
    )
    plot_file = os.path.join(
        boxplot_dir,
        f"{row['task1']}_{row['task2']}_{row['selected_rois']}_box.png",
    )
    plt.savefig(plot_file, bbox_inches="tight")
    plt.close()


def pipeline(task_pair, rois, data_root, out_dir, n_rois=400):
    task1, task2 = task_pair[0], task_pair[1]
    print(f"Calculating stats for {task1} vs {task2}, {rois} rois")
    task1_sub_ses, task1_modality = get_ses_modality(task1)
    task2_sub_ses, task2_modality = get_ses_modality(task2)
    subjects = get_subjects(task1_sub_ses, task2_sub_ses)
    task1_conn_measures, task2_conn_measures = get_conn_measures(
        task1_modality, task2_modality
    )
    results = {
        "correlation": [],
        "task1": [],
        "task2": [],
        "task1_modality": [],
        "task2_modality": [],
        "task1_measure": [],
        "task2_measure": [],
        "selected_rois": [],
        "p_value": [],
        "kept_subjects": [],
    }
    if task1_modality == task2_modality == "functional":
        for conn_measure in task1_conn_measures:
            task1_conn_measure = task2_conn_measure = conn_measure
            print(f"\tfor {task1_conn_measure}, {task2_conn_measure}")
            results = get_task_pair_stats(
                task1_conn_measure,
                task2_conn_measure,
                task1_sub_ses,
                task1_modality,
                task2_sub_ses,
                task2_modality,
                task1,
                task2,
                rois,
                data_root,
                n_rois,
                subjects,
                results,
            )
    else:
        for task1_conn_measure in task1_conn_measures:
            for task2_conn_measure in task2_conn_measures:
                print(f"\tfor {task1_conn_measure}, {task2_conn_measure}")
                results = get_task_pair_stats(
                    task1_conn_measure,
                    task2_conn_measure,
                    task1_sub_ses,
                    task1_modality,
                    task2_sub_ses,
                    task2_modality,
                    task1,
                    task2,
                    rois,
                    data_root,
                    n_rois,
                    subjects,
                    results,
                )

    stats_df = pd.DataFrame(results)
    plot_correlation(stats_df, out_dir)
    plot_boxplot(stats_df, out_dir)
    stats_file = os.path.join(out_dir, f"{task1}_{task2}_{rois}.csv")
    stats_df.to_csv(stats_file)
    print(f"Done {task1} vs {task2}, {rois} rois\n\n")
    return stats_df


if __name__ == "__main__":
    # cache directory
    cache = "/storage/store/work/haggarwa/"
    # output directory
    out_dir = os.path.join(cache, "func_v_struc")
    os.makedirs(out_dir, exist_ok=True)
    # task pairs
    task_pairs = [
        ("GoodBadUgly", "DWI"),
        ("MonkeyKingdom", "DWI"),
        ("RestingState", "DWI"),
        ("Raiders", "DWI"),
    ]

    # select rois
    rois = ["all", "intra", "inter"]

    all_results = Parallel(n_jobs=12, verbose=1, backend="multiprocessing")(
        delayed(pipeline)(task_pair, roi, cache, out_dir)
        for roi in rois
        for task_pair in task_pairs
    )

    all_results = pd.concat(all_results)
    all_results.to_csv(os.path.join(out_dir, "all_results.csv"))

    ## EXTRA PLOTS ##
    # plot boxplot for all functional sessions together
    # one plot for each roi selection
    boxplot_dir = os.path.join(out_dir, "overall_boxplot")
    os.makedirs(boxplot_dir, exist_ok=True)
    for roi in rois:
        all_results_roi = all_results[all_results["selected_rois"] == roi]
        d = {"Comparison": [], "FC measure": [], "Correlation FC v SC": []}
        for _, row in all_results_roi.iterrows():
            n_subs = len(row["kept_subjects"])
            corr = row["correlation"].reshape(n_subs, n_subs)
            same_sub = np.diagonal(corr, offset=0).tolist()
            upper_tri = corr[np.triu_indices_from(corr, k=1)].tolist()
            lower_tri = corr[np.tril_indices_from(corr, k=-1)].tolist()
            cross_sub = upper_tri + lower_tri
            d["Comparison"].extend(
                ["Within Subject"] * len(same_sub)
                + ["Across Subject"] * len(cross_sub)
            )
            d["FC measure"].extend(
                [row["task1_measure"]] * (len(same_sub) + len(cross_sub))
            )
            d["Correlation FC v SC"].extend(same_sub + cross_sub)
            # p_values.append(row["p_value"])
        df = pd.DataFrame(d)
        p_values = []
        for fc_measure in df["FC measure"].unique():
            df_fc = df[df["FC measure"] == fc_measure]
            same_sub = df_fc[df_fc["Comparison"] == "Within Subject"][
                "Correlation FC v SC"
            ].values
            cross_sub = df_fc[df_fc["Comparison"] == "Across Subject"][
                "Correlation FC v SC"
            ].values
            t_test = stats.ttest_ind(
                same_sub, cross_sub, alternative="greater"
            )
            p_values.append(t_test[1])
        fig, ax = plt.subplots()
        sns.boxplot(
            x="Correlation FC v SC",
            y="FC measure",
            hue="Comparison",
            data=df,
            ax=ax,
            orient="h",
            fliersize=0,
        )
        for i, p in enumerate(p_values):
            if p > 0.05:
                signif = "ns"
            elif p <= 0.0001:
                signif = "p ≤ 0.0001"
            elif p <= 0.001:
                signif = "p ≤ 0.001"
            elif p <= 0.01:
                signif = "p ≤ 0.01"
            elif p <= 0.05:
                signif = "p ≤ 0.05"
            ax.text(
                df["Correlation FC v SC"].min(),
                i,
                signif,
                fontsize=10,
                bbox=dict(facecolor="white", alpha=1),
            )
        ax.set_title(
            f"FC-SC correlation within vs across subjects, {row['selected_rois']} rois"
        )
        plot_file = os.path.join(
            boxplot_dir, f"withinvacross_FCSC_{row['selected_rois']}_box.png"
        )
        plt.savefig(plot_file, bbox_inches="tight")
        plt.close()

    # plot boxplot for FC-SC correlation but compare across different roi selections
    # one plot for within subject and one for across subject
    d = {
        "Comparison": [],
        "FC measure": [],
        "Correlation FC v SC": [],
        "Selected ROIs": [],
    }
    for _, row in all_results.iterrows():
        n_subs = len(row["kept_subjects"])
        corr = row["correlation"].reshape(n_subs, n_subs)
        same_sub = np.diagonal(corr, offset=0).tolist()
        upper_tri = corr[np.triu_indices_from(corr, k=1)].tolist()
        lower_tri = corr[np.tril_indices_from(corr, k=-1)].tolist()
        cross_sub = upper_tri + lower_tri
        d["Comparison"].extend(
            ["Within Subject"] * len(same_sub)
            + ["Across Subject"] * len(cross_sub)
        )
        d["FC measure"].extend(
            [row["task1_measure"]] * (len(same_sub) + len(cross_sub))
        )
        d["Correlation FC v SC"].extend(same_sub + cross_sub)
        d["Selected ROIs"].extend(
            [row["selected_rois"]] * (len(same_sub) + len(cross_sub))
        )
    df = pd.DataFrame(d)
    for comparison in df["Comparison"].unique():
        df_comp = df[df["Comparison"] == comparison]
        fig, ax = plt.subplots()
        sns.boxplot(
            x="Correlation FC v SC",
            y="FC measure",
            hue="Selected ROIs",
            data=df,
            ax=ax,
            orient="h",
            fliersize=0,
        )
        ax.set_title(
            f"FC-SC correlation {comparison} subjects, based on ROI selection"
        )
        plot_file = os.path.join(boxplot_dir, f"{comparison}_FCSC_box.png")
        plt.savefig(plot_file, bbox_inches="tight")
        plt.close()
