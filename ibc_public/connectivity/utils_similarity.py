"""Utility functions for calculating similarity between connectivity matrices.
"""
import numpy as np
import pandas as pd
from scipy import stats


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


def similarity(
    connectivity_matrices, subjects, mean_center=True, z_transform=True
):
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
        similarity_mat_centered = (
            similarity_mat_centered + similarity_mat.mean()
        )

    if z_transform:
        similarity_mat_zscored = stats.zscore(similarity_mat, axis=0)
        similarity_mat_zscored = stats.zscore(similarity_mat_zscored, axis=1)

    similarity_mat = pd.DataFrame(
        similarity_mat, columns=task2_subs, index=task1_subs
    )
    similarity_mat_centered = pd.DataFrame(
        similarity_mat_centered, columns=task2_subs, index=task1_subs
    )
    similarity_mat_zscored = pd.DataFrame(
        similarity_mat_zscored, columns=task2_subs, index=task1_subs
    )

    return similarity_mat, similarity_mat_centered, similarity_mat_zscored


def symmetrize(a):
    """Symmetrize a matrix by dropping rows and columns that are not
    shared between the rows and columns.

    Parameters
    ----------
    a : pandas dataframe
        a non-symmetrical matrix in pandas dataframe format. Already
        symmetrical matrices will not be affected.

    Returns
    -------
    a : numpy array
        symmetrical matrix in numpy array format
    columns : numpy array
        columns of the symmetrical matrix
    mask : tuple
        a tuple of two boolean arrays indicating which indices of rows
        and columns that were kept in the symmetrization process
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
    p_value_t = t_test[1]

    mwu_test = stats.mannwhitneyu(
        same_sub_corr, cross_sub_corr, alternative="greater"
    )
    p_value_mwu = mwu_test[1]

    return p_value_t, p_value_mwu


def _mask(df, task, cov, measure):
    """Create a boolean mask for a dataframe by task, covariance estimator,
    and measure."""
    if task == "SC":
        return df["measure"] == "SC"
    else:
        return (df["task"] == task) & (df["measure"] == cov + " " + measure)


def _filter_connectivity(data, task, cov, measure):
    """Keep connectivity matrices by task, covariance estimator, and measure,
    by applying a boolean mask."""
    connectivity = data[_mask(data, task, cov, measure)][
        "connectivity"
    ].tolist()
    return np.array(connectivity)


def _get_subjects(data, task, cov, measure):
    """Keep subject labels for a given task, covariance estimator,
    and measure."""
    return data[_mask(data, task, cov, measure)]["subject"].tolist()


def get_similarity(all_connectivity, task_pair, cov, measure):
    """Get similarity between two tasks for a given covariance estimator
     and measure.

    Parameters
    ----------
    all_connectivity : pandas dataframe
        a dataframe with the average connectivity for each subject, task,
        and measure
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
    for i, matrix in enumerate(
        [similarity_mat, similarity_centered, similarity_z]
    ):
        matrix, kept_subs, kept_ind = symmetrize(matrix)
        p_value_t, p_value_mwu = samevcross_test(matrix)
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
            "p_value_t": p_value_t,
            "p_value_mwu": p_value_mwu,
            "matrix": matrix.flatten(),
            "kept_subjects": kept_subs,
            "comparison": task1 + " vs. " + task2,
        }
        result.append(result_)

    return result
