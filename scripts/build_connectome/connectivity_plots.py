"""
This script:
1) plots connectivity networks and matrices for each subject for each 
task-connectivity_measure combination
2) calculates and plots pearsons correlation matrices across subjects between 
given pairs of task-connectvity_measure combinations 
3) t-tests and plots box plots to compare same-subject and cross-subject 
correlations for each task-connectivity_measure combination
"""

import numpy as np
import matplotlib.pyplot as plt
from nilearn import datasets, plotting, connectome
import pandas as pd
import os
import seaborn as sns
from scipy import stats
from ibc_public.utils_data import get_subject_session
from glob import glob

sns.set_theme(context="talk", style="ticks")


def get_ses_modality(task):
    """Get session numbers and modality for given task

    Parameters
    ----------
    task : str
        name of the task

    Returns
    -------
    sub_ses : dict
        dictionary with subject as key and session number as value
    modality : str
        modality of the task
    """
    if task == "Raiders":
        # session names with movie task data
        session_names = ["raiders1", "raiders2"]
    elif task == "RestingState":
        # session names with RestingState state task data
        session_names = ["mtt1", "mtt2"]
    elif task == "DWI":
        # session names with diffusion data
        session_names = ["anat1"]
    # get session numbers for each subject
    subject_sessions = sorted(get_subject_session(session_names))
    sub_ses = {}
    if task in ["Raiders", "RestingState"]:
        modality = "functional"
        for subject_session in subject_sessions:
            if (
                subject_session[0] in ["sub-11", "sub-12"]
                and subject_session[1] == "ses-13"
            ):
                continue
            try:
                sub_ses[subject_session[0]]
            except KeyError:
                sub_ses[subject_session[0]] = []
            sub_ses[subject_session[0]].append(subject_session[1])
    elif task == "DWI":
        modality = "structural"
        sub_ses = {
            subject_session[0]: "ses-12"
            if subject_session[0] in ["sub-01", "sub-15"]
            else subject_session[1]
            for subject_session in subject_sessions
        }

    return sub_ses, modality


def plot_connectivity_network(
    connectivity_matrix, task, sub, ax_net, DATA_ROOT
):
    """Plot connectivity network

    Parameters
    ----------
    connectivity_matrix : np.array
        a numpy array with connectivity matrix
    labels : list
        a list of labels for the connectivity matrix
    title : str
        title of the plot
    plot_filename : str
        filename of the plot
    """

    coords = pd.read_csv(
        os.path.join(
            DATA_ROOT,
            "sub-04",
            "ses-08",
            "dwi",
            "tract2mni_tmp",
            "schaefer_2018",
            "ras_coords_400.csv",
        )
    )
    coords = coords[["R", "A", "S"]]
    coords = coords.to_numpy()
    edge_threshold = "99.5%" if task == "DWI" else "99.9%"
    plotting.plot_connectome(
        connectivity_matrix,
        coords,
        edge_threshold=edge_threshold,
        node_size=5,
        axes=ax_net,
        title=f"{sub}",
    )


def plot_connectivity_matrix(connectivity_matrix, task, sub, ax):
    """Plot connectivity matrix

    Parameters
    ----------
    connectivity_matrix : np.array
        a numpy array with connectivity matrix
    task : str
        name of the task
    conn_measure : str
        name of the connectivity measure
    sub : str
        name of the subject
    ax : matplotlib.axes._subplots.AxesSubplot
        matplotlib axes object to plot the connectivity matrix
    """
    # plot connectivity matrix
    if task == "DWI":
        connectivity_matrix = np.log(connectivity_matrix)
        vmax = 2.6
        vmin = -10.1
    else:
        vmax = 0.8
        vmin = -0.35
    plotting.plot_matrix(
        connectivity_matrix,
        title=f"{sub}",
        axes=ax,
        tri="lower",
        vmax=vmax,
        vmin=vmin,
    )


def get_all_subject_connectivity(
    task,
    conn_measure,
    DATA_ROOT,
    tmp_dir,
    atlas,
    combinations_plotted,
    skip_sub06=False,
    plot_network=True,
    plot_matrix=True,
):
    """Get all subject connectivity for given task and connectivity measure

    Parameters
    ----------
    task : str
        name of the task
    conn_measure : str
        name of the connectivity measure
    skip_sub06 : bool, optional
        whether to skip sub-06 (because normalisation of diffusion tracts fails
        for sub-06), by default False

    Returns
    -------
    np.array
        a numpy array with connectivity matrices for all subjects for the given
        task and connectivity measure
    """
    # get session numbers and modality for the given task
    task_sub_ses, task_modality = get_ses_modality(task)

    if task_modality == "functional":
        ses_mod_dir = "func"
    elif task_modality == "structural":
        ses_mod_dir = "dwi"

    if plot_network:
        fig_net, axs_net = plt.subplots(
            len(task_sub_ses.items()) // 3, 3, figsize=(36, 16)
        )
        net_plot_counter = 0
    if plot_matrix:
        fig_mat, axs_mat = plt.subplots(
            len(task_sub_ses.items()) // 4, 4, figsize=(16, 12)
        )
        mat_plot_counter = 0

    all_subject_connectivity = []

    for sub, sess in task_sub_ses.items():
        if skip_sub06 and sub == "sub-06":
            continue
        if conn_measure == "Pearsons_corr":
            sess = sorted(sess)
            all_pearsons = []
            all_pearsons_ = []
            for ses in sess:
                file_loc = os.path.join(DATA_ROOT, sub, ses, ses_mod_dir)
                runs = glob(os.path.join(file_loc, f"*{conn_measure}*"))
                for run in runs:
                    pearson_mat = pd.read_csv(run, names=atlas.labels)
                    pearson_mat_flat = connectome.sym_matrix_to_vec(
                        pearson_mat.to_numpy(), discard_diagonal=True
                    )
                    all_pearsons.append(pearson_mat_flat)
                    all_pearsons_.append(pearson_mat)
            subject_connectivity_ = np.mean(all_pearsons_, axis=0)
            if plot_network:
                ax_net = axs_net.flat[net_plot_counter]
                plot_connectivity_network(
                    subject_connectivity_, task, sub, ax_net, DATA_ROOT
                )
                net_plot_counter += 1
            if plot_matrix:
                ax_mat = axs_mat.flat[mat_plot_counter]
                plot_connectivity_matrix(
                    subject_connectivity_, task, sub, ax_mat
                )
                mat_plot_counter += 1
            subject_connectivity = np.mean(all_pearsons, axis=0)
        else:
            if task_modality == "functional":
                sess = sorted(sess)
                ses = sess[1]
            elif task_modality == "structural":
                ses = sess
            file_loc = os.path.join(DATA_ROOT, sub, ses, ses_mod_dir)
            subject_connectivity = glob(
                os.path.join(file_loc, f"*{conn_measure}*")
            )
            assert len(subject_connectivity) == 1
            subject_connectivity = pd.read_csv(
                subject_connectivity[0], names=atlas.labels
            )
            if plot_network:
                ax_net = axs_net.flat[net_plot_counter]
                plot_connectivity_network(
                    subject_connectivity, task, sub, ax_net, DATA_ROOT
                )
                net_plot_counter += 1
            if plot_matrix:
                ax_mat = axs_mat.flat[mat_plot_counter]
                plot_connectivity_matrix(
                    subject_connectivity, task, sub, ax_mat
                )
                mat_plot_counter += 1
            subject_connectivity = connectome.sym_matrix_to_vec(
                subject_connectivity.to_numpy(), discard_diagonal=True
            )
        all_subject_connectivity.append(subject_connectivity)

        if plot_network:
            fig_net.subplots_adjust(wspace=0, hspace=0)
            fig_net.suptitle(
                f"{task} {task_modality} connectivity networks,"
                f" measured as {conn_measure}"
            )
            if not os.path.exists("connectivity_networks"):
                os.makedirs("connectivity_networks")
            fig_net.savefig(
                f"connectivity_networks/{task}_{conn_measure}_networks.png",
                bbox_inches="tight",
            )
        if plot_matrix:
            fig_mat.subplots_adjust(wspace=0.4, hspace=0)
            fig_mat.suptitle(
                f"{task} {task_modality} connectivity matrices,"
                f" measured as {conn_measure}"
            )
            if not os.path.exists("connectivity_matrices"):
                os.makedirs("connectivity_matrices")
            fig_mat.savefig(
                f"connectivity_matrices/{task}_{conn_measure}_matrices.png",
                bbox_inches="tight",
            )

            combinations_plotted.append(f"{task}_{conn_measure}")

    return np.array(all_subject_connectivity), combinations_plotted


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
    task1_conn = connectivity_matrices[0]
    task2_conn = connectivity_matrices[1]
    task1_subs = subjects[0]
    task2_subs = subjects[1]
    similarity_mat = np.corrcoef(task1_conn, task2_conn)
    similarity_mat = np.hsplit(np.vsplit(similarity_mat, 2)[0], 2)[1]
    if mean_center:
        similarity_mat_centered = similarity_mat - similarity_mat.mean(axis=0)
        similarity_mat_centered = (
            similarity_mat_centered.T - similarity_mat_centered.mean(axis=1)
        ).T
        similarity_mat = similarity_mat_centered
    return pd.DataFrame(similarity_mat, columns=task1_subs, index=task2_subs)


def plot_correlation_matrix(
    matrix_as_df,
    fig_title,
    xlabel,
    ylabel,
    plot_filename,
    mean_center,
    fig_size=(10, 10),
):
    """Plot correlation matrix

    Parameters
    ----------
    matrix_as_df : pandas dataframe
        correlation matrix as a dataframe with subjects as index and columns
    fig_title : str
        figure title
    xlabel : str
        x-axis label
    ylabel : str
        y-axis label
    plot_filename : str
        name of the plot file
    fig_size : tuple, optional
        size of the figure, by default (10,10)
    """
    fig = plt.figure(figsize=fig_size)
    sns.heatmap(matrix_as_df, cmap="Spectral")
    plt.title(fig_title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    save_folder = "correration_matrices"
    if mean_center:
        save_folder += "_mean_centered"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    plt.savefig(f"{save_folder}/{plot_filename}.png", bbox_inches="tight")
    plt.close()


def save_all_info(
    similarity_df,
    d,
    task1,
    task2,
    task1_conn_measure,
    task2_conn_measure,
    task1_modality,
    task2_modality,
):
    """Update dictionary with every single correlation value in a long-form
    fashion (every row contains a single correlation in the correlation column),
    and the corresponding information about the tasks, the modality, the
    connectivity measure, and the comparison (same subject vs cross subject)

    Parameters
    ----------
    similarity_df : pandas dataframe
        correlation matrix as a dataframe with subjects as index and columns
    d : dict
        dictionary to update
    task1 : str
        name of the first task
    task2 : str
        name of the second task
    task1_conn_measure : str
        name of the connectivity measure used for the first task
    task2_conn_measure : str
        name of the connectivity measure used for the second task
    task1_modality : str
        name of the modality of the first task
    task2_modality : str
        name of the modality of the second task

    Returns
    -------
    d : dict
        updated dictionary
    """
    same_sub_corr = np.diagonal(similarity_df.values).tolist()
    cross_sub_corr = similarity_df.values[
        np.where(~np.eye(similarity_df.values.shape[0], dtype=bool))
    ].tolist()

    t_test = stats.ttest_ind(
        same_sub_corr, cross_sub_corr, alternative="greater"
    )
    p_value = t_test[0]

    for comparison in ["same subject", "cross subject"]:
        if comparison == "same subject":
            size = len(same_sub_corr)
            d["correlation"].extend(same_sub_corr)
        else:
            size = len(cross_sub_corr)
            d["correlation"].extend(cross_sub_corr)
        d["comparison"].extend([comparison for _ in range(size)])
        d["task1"].extend([task1 for _ in range(size)])
        d["task2"].extend([task2 for _ in range(size)])
        d["task1_measure"].extend([task1_conn_measure for _ in range(size)])
        d["task2_measure"].extend([task2_conn_measure for _ in range(size)])
        d["task1_modality"].extend([task1_modality for _ in range(size)])
        d["task2_modality"].extend([task2_modality for _ in range(size)])
        d["p_value"].extend([p_value for _ in range(size)])

    return d


def plot_summary_box(
    df,
    x,
    y,
    hue,
    xlabel,
    ylabel,
    fig_title,
    plot_filename,
    mean_center,
    fig_size=(10, 10),
):
    """Plot summary boxplot"""
    plt.figure(figsize=fig_size)
    sns.boxplot(df, orient="h", x=x, y=y, hue=hue, fliersize=0)
    if mean_center:
        xlabel = xlabel + " (double mean-centered)"
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(fig_title)
    save_folder = "boxplots"
    if mean_center:
        save_folder += "_mean_centered"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    plt.savefig(f"{save_folder}/{plot_filename}.png", bbox_inches="tight")
    plt.close("all")


def run_all_comparisons(
    DATA_ROOT,
    tmp_dir,
    task_pair,
    func_conn_measures,
    struc_conn_measures,
    atlas,
    d,
    mean_center,
):
    """Manages specific choices related to calculation of pearson correlation
    between two connectivity matrices, depending on the modality and the type
    of connectivity measure

    Parameters
    ----------
    DATA_ROOT : str
        path to the root directory of the data
    tmp_dir : str
        path to the temporary directory
    task_pair : tuple of str
        pair of tasks to be compared
    func_conn_measures : list of str
        calculated functional connectivity measures
    struc_conn_measures : list of str
        calculated structural connectivity measures
    atlas : sklearn.utils.Bunch
        Dictionary-like object, contains:
        - 'maps': `str`, path to nifti file containing the
        3D ~nibabel.nifti1.Nifti1Image (its shape is (182, 218, 182)). The
        values are consecutive integers between 0 and n_rois which can be
        interpreted as indices in the list of labels.
        - 'labels': numpy.ndarray of str, array containing the ROI labels
        including Yeo-network annotation.
        - 'description': `str`, short description of the atlas
          and some references.
    d : dict
        dictionary containing every single correlation value in a long-form
        fashion (every row contains a single correlation in the correlation
        column), and the corresponding information about the tasks,
        the modality, the connectivity measure, and the comparison (same
        subject vs cross subject)
    mean_center : bool
        whether to mean-center the correlation matrices

    Returns
    -------
    dictionary
        dictionary d defined above is updated with the new values
    """
    # get session numbers and modality for each task in the pair
    task1, task2 = task_pair[0], task_pair[1]
    task1_sub_ses, task1_modality = get_ses_modality(task1)
    task2_sub_ses, task2_modality = get_ses_modality(task2)
    combinations_plotted = []
    if task1_modality == task2_modality == "functional":
        conn_measures = func_conn_measures
        task1_conn_measures = task2_conn_measures = struc_conn_measures
        for conn_measure in conn_measures:
            task1_conn_measure = task2_conn_measure = conn_measure
            name = f"{task1}_{task2}_{conn_measure}"
            print(name)
            if f"{task1}_{conn_measure}" in combinations_plotted:
                plot_matrix = plot_network = False
            else:
                plot_matrix = plot_network = True
            # get all the connectivity matrices for each subject
            task1_conn, combinations_plotted = get_all_subject_connectivity(
                task1,
                conn_measure,
                DATA_ROOT,
                tmp_dir,
                atlas,
                combinations_plotted,
                plot_matrix=plot_matrix,
                plot_network=plot_network,
            )
            if f"{task2}_{conn_measure}" in combinations_plotted:
                plot_matrix = plot_network = False
            else:
                plot_matrix = plot_network = True
            task2_conn, combinations_plotted = get_all_subject_connectivity(
                task2,
                conn_measure,
                DATA_ROOT,
                tmp_dir,
                atlas,
                combinations_plotted,
                plot_matrix=plot_matrix,
                plot_network=plot_network,
            )
            subjects = [list(task1_sub_ses.keys()), list(task2_sub_ses.keys())]
            fig_title = (
                f"Correlation b/w {task1_modality} connectivities\n"
                f" measured from {task1} and {task2}\nas {conn_measure}"
            )
            fig_xlabel = f"{task2} {task2_conn_measure}"
            fig_ylabel = f"{task1} {task1_conn_measure}"
            fig_title = (
                f"Correlation b/w {task1_modality} connectivities\n"
                f" measured from {task1} and {task2}\nas {conn_measure}"
            )

            if mean_center:
                fig_title = "Double mean centered " + fig_title

            similarity_df = calculate_pearson_corr(
                [task1_conn, task2_conn],
                [list(task1_sub_ses.keys()), list(task2_sub_ses.keys())],
                mean_center,
            )
            plot_correlation_matrix(
                similarity_df,
                fig_title,
                fig_xlabel,
                fig_ylabel,
                name,
                mean_center,
            )
            d = save_all_info(
                similarity_df,
                d,
                task1,
                task2,
                task1_conn_measure,
                task2_conn_measure,
                task1_modality,
                task2_modality,
            )
            plt.close("all")
    else:
        if task1_modality == task2_modality == "structural":
            task1_conn_measures = task2_conn_measures = struc_conn_measures
        elif task1_modality == "functional":
            task1_conn_measures = func_conn_measures
            task2_conn_measures = struc_conn_measures
        elif task1_modality == "structural":
            task1_conn_measures = struc_conn_measures
            task2_conn_measures = func_conn_measures
        else:
            return "Should never reach here"

        skip_sub06 = True
        task1_sub_ses.pop("sub-06")
        task2_sub_ses.pop("sub-06")
        for task1_conn_measure in task1_conn_measures:
            if f"{task1}_{task1_conn_measure}" in combinations_plotted:
                plot_matrix = plot_network = False
            else:
                plot_matrix = plot_network = True
            task1_conn, combinations_plotted = get_all_subject_connectivity(
                task1,
                task1_conn_measure,
                DATA_ROOT,
                tmp_dir,
                atlas,
                combinations_plotted,
                skip_sub06=skip_sub06,
                plot_matrix=plot_matrix,
                plot_network=plot_network,
            )
            for task2_conn_measure in task2_conn_measures:
                name = (
                    f"{task1}_{task1_conn_measure}_"
                    f"{task2}_{task2_conn_measure}"
                )
                print(name)
                if f"{task1}_{task2_conn_measure}" in combinations_plotted:
                    plot_matrix = plot_network = False
                else:
                    plot_matrix = plot_network = True
                (
                    task2_conn,
                    combinations_plotted,
                ) = get_all_subject_connectivity(
                    task2,
                    task2_conn_measure,
                    DATA_ROOT,
                    tmp_dir,
                    atlas,
                    combinations_plotted,
                    skip_sub06,
                    plot_matrix=plot_matrix,
                    plot_network=plot_network,
                )

                fig_title = (
                    f"Correlation b/w {task1} {task1_modality}\n"
                    f"connectivity measured as {task1_conn_measure}\n"
                    f"and {task2} {task2_modality} connectivity measured\n"
                    f"as {task2_conn_measure}"
                )
                if mean_center:
                    fig_title = "Double mean centered " + fig_title
                fig_xlabel = f"{task2} {task2_conn_measure}"
                fig_ylabel = f"{task1} {task1_conn_measure}"

                similarity_df = calculate_pearson_corr(
                    [task1_conn, task2_conn],
                    [list(task1_sub_ses.keys()), list(task2_sub_ses.keys())],
                    mean_center,
                )
                plot_correlation_matrix(
                    similarity_df,
                    fig_title,
                    fig_xlabel,
                    fig_ylabel,
                    name,
                    mean_center,
                )
                d = save_all_info(
                    similarity_df,
                    d,
                    task1,
                    task2,
                    task1_conn_measure,
                    task2_conn_measure,
                    task1_modality,
                    task2_modality,
                )
                plt.close("all")

    return d


if __name__ == "__main__":
    #### Inputs ####
    # set data paths
    DATA_ROOT = "/data/parietal/store2/data/ibc/derivatives/"
    # cache directory
    cache = "/storage/store/work/haggarwa/"
    tmp_dir = "connectivity_tmp"
    # pairs of connectivity matrices to compare
    # pick between DWI, RestingState and Raiders
    task_pairs = [
        ("RestingState", "DWI"),
        ("Raiders", "DWI"),
        ("DWI", "DWI"),
        ("Raiders", "RestingState"),
        ("Raiders", "Raiders"),
        ("RestingState", "RestingState"),
    ]
    # calculated measures of functional connectivity
    func_conn_measures = [
        "Pearsons_corr",
        "GraphicalLassoCV_corr",
        "GraphicalLassoCV_partcorr",
        "GroupSparseCovarianceCV_corr",
        "GroupSparseCovarianceCV_partcorr",
    ]
    # calculated measures of structural connectivity
    struc_conn_measures = [
        "MNI152_siftweighted",
        "individual_siftweighted",
        "MNI152_nosift",
        "individual_nosift",
    ]
    # get atlas
    atlas = datasets.fetch_atlas_schaefer_2018(
        data_dir=cache, resolution_mm=2, n_rois=400
    )
    # give atlas a custom name
    atlas["name"] = "schaefer400"
    # dictionary to store all the results
    results = {
        k: []
        for k in [
            "task1",
            "task2",
            "correlation",
            "p_value",
            "comparison",
            "task1_measure",
            "task2_measure",
            "task1_modality",
            "task2_modality",
        ]
    }

    #### Run Comparisons ####
    # save results with and without mean-centering the correlations
    for mean_center in [True, False]:
        # dictionary to store all the results
        results = {
            k: []
            for k in [
                "task1",
                "task2",
                "correlation",
                "p_value",
                "comparison",
                "task1_measure",
                "task2_measure",
                "task1_modality",
                "task2_modality",
            ]
        }
        # loop over all pairs of tasks
        for task_pair in task_pairs:
            results = run_all_comparisons(
                DATA_ROOT,
                tmp_dir,
                task_pair,
                func_conn_measures,
                struc_conn_measures,
                atlas,
                results,
                mean_center,
            )
        # save results
        results_df = pd.DataFrame(results)
        correlation_file = "all_comparisons"
        if mean_center:
            correlation_file += "_mean_centered"
        results_df.to_csv(f"{correlation_file}.csv", index=False)
        # keep results for functional vs functional comparisons
        # between resting state and raiders
        results_df_funcvfunc = results_df[
            (results_df["task1_modality"] == "functional")
            & (results_df["task2_modality"] == "functional")
            & (results_df["task1"] != results_df["task2"])
        ]
        # results for structural vs structural comparisons
        results_df_strucvstruc = results_df[
            (results_df["task1_modality"] == "structural")
            & (results_df["task2_modality"] == "structural")
        ]
        # results for Raiders vs structural comparisons
        results_df_Raidersvstruc = results_df[
            (
                (results_df["task1_modality"] == "functional" & 
                 results_df["task1"] == results_df["Raiders"])
                & (results_df["task2_modality"] == "structural")
            )
            | (
                (results_df["task1_modality"] == "structural")
                & (results_df["task2_modality"] == "functional" & 
                 results_df["task1"] == results_df["Raiders"])
            )
        ]
        # results for RestingState vs structural comparisons
        results_df_RestingStatevstruc = results_df[
            (
                (results_df["task1_modality"] == "functional" & 
                 results_df["task1"] == results_df["RestingState"])
                & (results_df["task2_modality"] == "structural")
            )
            | (
                (results_df["task1_modality"] == "structural")
                & (results_df["task2_modality"] == "functional" & 
                 results_df["task1"] == results_df["RestingState"])
            )
        ]
        # results for functional vs structural comparisons
        results_df_funcvstruc = results_df[
            (
                (results_df["task1_modality"] == "functional")
                & (results_df["task2_modality"] == "structural")
            )
            | (
                (results_df["task1_modality"] == "structural")
                & (results_df["task2_modality"] == "functional")
            )
        ]
        if results_df["task1_modality"] == "functional":
            results_df_funcvstruc['task1'] = 'all'
        else:
            results_df_funcvstruc['task2'] = 'all'
        # plot the summary box plots for all the above subsets of results
        for results_df in [
            results_df_funcvfunc,
            results_df_strucvstruc,
            results_df_Raidersvstruc,
            results_df_RestingStatevstruc,
            results_df_funcvstruc,
        ]:
            name = (
                results_df.iloc[0]["task1"]
                + results_df.iloc[0]["task1_modality"]
                + "v"
                + results_df.iloc[0]["task2"]
                + results_df.iloc[0]["task2_modality"]
            )
            plot_summary_box(
                results_df,
                "correlation",
                "task1_measure",
                "comparison",
                "Pearson's correlation",
                "Functional connectivity measure",
                f"{results_df.iloc[0]['task1']} "
                f"{results_df.iloc[0]['task1_modality']}"
                f" vs {results_df.iloc[0]['task2']} "
                f"{results_df.iloc[0]['task2_modality']} connectivity",
                name,
                mean_center,
            )
