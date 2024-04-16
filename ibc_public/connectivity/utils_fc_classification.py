"""Utility functions for connectivity classification."""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    balanced_accuracy_score,
)
from sklearn.model_selection import GroupShuffleSplit, LeavePGroupsOut
from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier
from tqdm import tqdm


def _update_results(
    results,
    accuracy,
    auc,
    predictions,
    truth,
    train,
    test,
    task_label,
    connectivity_measure,
    classify,
    dummy_auc,
    dummy_accuracy,
    dummy_predictions,
    balanced_accuracy,
    dummy_balanced_accuracy,
    # weights,
):
    """Update the results dictionary with the current classification results."""
    results["LinearSVC_accuracy"].append(accuracy)
    results["LinearSVC_auc"].append(auc)
    # store test labels and predictions
    results["LinearSVC_predicted_class"].append(predictions)
    results["true_class"].append(truth)
    # store the connectivity measure
    results["connectivity"].append(connectivity_measure)
    # store the train and test sets
    results["train_sets"].append(train)
    results["test_sets"].append(test)
    # store the task label
    results["task_label"].append(task_label)
    # extra columns for consistency with other results
    results["classes"].append(classify)
    if classify in ["Runs", "Tasks"]:
        grouping = "Subjects"
    elif classify == "Subjects":
        grouping = "Tasks"
    results["groups"].append(grouping)
    results["Dummy_auc"].append(dummy_auc)
    results["Dummy_accuracy"].append(dummy_accuracy)
    results["Dummy_predicted_class"].append(dummy_predictions)
    results["balanced_accuracy"].append(balanced_accuracy)
    results["dummy_balanced_accuracy"].append(dummy_balanced_accuracy)
    # results["weights"].append(weights)

    return results


def _select_data(data, classify, task, pooled_tasks):
    """Select the data based on the classification scheme."""
    if pooled_tasks:
        assert isinstance(task, list)
        task_label = " vs ".join(task)
        data = data[data["tasks"].isin(task)]
        if classify == "Tasks":
            classes = data["tasks"].to_numpy(dtype=object)
            groups = data["subject_ids"].to_numpy(dtype=object)
        elif classify == "Subjects":
            classes = data["subject_ids"].to_numpy(dtype=object)
            groups = data["tasks"].to_numpy(dtype=object)
        elif classify == "Runs":
            data["run_task"] = data["run_labels"] + "_" + data["tasks"]
            classes = data["run_task"].to_numpy(dtype=object)
            groups = data["subject_ids"].to_numpy(dtype=object)
    else:
        if classify in ["Runs", "Subjects"]:
            assert isinstance(task, str)
            task_label = task
            data = data[data["tasks"] == task]
            if classify == "Runs":
                classes = data["run_labels"].to_numpy(dtype=object)
                groups = data["subject_ids"].to_numpy(dtype=object)
            elif classify == "Subjects":
                classes = data["subject_ids"].to_numpy(dtype=object)
                groups = data["run_labels"].to_numpy(dtype=object)
        elif classify == "Tasks":
            assert isinstance(task, list)
            task_label = " vs ".join(task)
            data = data[data["tasks"].isin(task)]
            classes = data["tasks"].to_numpy(dtype=object)
            groups = data["subject_ids"].to_numpy(dtype=object)

    return data, task_label, classes, groups


def classify_connectivity(
    connectomes,
    classes,
    task_label,
    connectivity_measure,
    train,
    test,
    results,
    classify,
    pooled_tasks,
):
    """Classify the given connectomes using the given classes.

    Parameters
    ----------
    connectomes : numpy array
        Array containing the connectomes to classify. Each row is a vectorised connectome (upper triangle), each corresponding a run of a subject, meaning n_rows = n_runs * n_subjects for a task. When classify == "Runs" or "Subjects", the connectomes only correspond to one task, but when classify == "Task", we are classifying between two tasks. The number of columns is the number of edges in the connectome, meaning n_columns = n_parcels * (n_parcels - 1) / 2.
    classes : numpy array
        One dimensional, dtype = object numpy array containing the class name as str for each connectome, so n_rows = n_runs * n_subjects for the task involved as explained above. When classify == "Runs", the class names are runs, such as "run-01", "run-02" and so on. When classify == "Subjects", they are the subjects, such as "sub-01", "sub-02" and so on. And when classify == "Tasks", they are tasks, such as "RestingState", "Raiders", "GoodBadUgly", "MonkeyKingdom".
    task_label : str
        Name of the task currently being classified.
    connectivity_measure : str
        Connectivity measure currently being classified.
    train : numpy array
        Array containing the indices of the training set.
    test : numpy array
        Array containing the indices of the test set.
    results : dict
        Dictionary storing the results of the classification.
    classify : str
        What is being classified. Can be "Runs", "Tasks", or "Subjects".
    pooled_tasks : bool
        Whether the classification is between multiple tasks.

    Returns
    -------
    dict
        Dictionary storing the results of the classification.
    """
    # fit the classifier
    classifier = LinearSVC(max_iter=100000, dual="auto").fit(
        connectomes[train], classes[train]
    )
    # make predictions for the left-out test subjects
    predictions = classifier.predict(connectomes[test])
    accuracy = accuracy_score(classes[test], predictions)
    balanced_accuracy = balanced_accuracy_score(classes[test], predictions)
    # weights = classifier.coef_

    # fit a dummy classifier to get chance level
    dummy = DummyClassifier(strategy="stratified").fit(
        connectomes[train], classes[train]
    )
    dummy_predictions = dummy.predict(connectomes[test])
    dummy_accuracy = accuracy_score(classes[test], dummy_predictions)
    dummy_balanced_accuracy = balanced_accuracy_score(
        classes[test], dummy_predictions
    )

    print(
        f"\n{classify}, {connectivity_measure}, {task_label}: {accuracy:.2f} / {dummy_accuracy:.2f} ||| {balanced_accuracy:.2f} / {dummy_balanced_accuracy:.2f}"
    )

    if pooled_tasks:
        auc = 0
        dummy_auc = 0
    else:
        if classify in ["Runs", "Subjects"]:
            auc = 0
            dummy_auc = 0
        else:
            average = None
            multi_class = "raise"
            auc = roc_auc_score(
                classes[test],
                classifier.decision_function(connectomes[test]),
                average=average,
                multi_class=multi_class,
            )
            dummy_auc = roc_auc_score(
                classes[test],
                dummy.predict_proba(connectomes[test])[:, 1],
                average=average,
                multi_class=multi_class,
            )

    # store the scores for this cross-validation fold
    results = _update_results(
        results,
        accuracy,
        auc,
        predictions,
        classes[test],
        train,
        test,
        task_label,
        connectivity_measure,
        classify,
        dummy_auc,
        dummy_accuracy,
        dummy_predictions,
        # weights,
        balanced_accuracy,
        dummy_balanced_accuracy,
    )

    return results


def cross_validate(
    connectomes,
    classes,
    splits,
    task_label,
    connectivity_measure,
    results,
    classify,
    pooled_tasks,
):
    """Cross validate the given connectomes using the given classes.

    Parameters
    ----------
    connectomes : numpy array
        Array containing the connectomes to classify. Each row is a vectorised connectome (upper triangle), each corresponding a run of a subject, meaning n_rows = n_runs * n_subjects for a task. When classify == "Runs" or "Subjects", the connectomes only correspond to one task, but when classify == "Task", we are classifying between two tasks. The number of columns is the number of edges in the connectome, meaning n_columns = n_parcels * (n_parcels - 1) / 2.
    classes : numpy array
        One dimensional, dtype = object numpy array containing the class name as str for each connectome, so n_rows = n_runs * n_subjects for the task involved as explained above. When classify == "Runs", the class names are runs, such as "run-01", "run-02" and so on. When classify == "Subjects", they are the subjects, such as "sub-01", "sub-02" and so on. And when classify == "Tasks", they are tasks, such as "RestingState", "Raiders", "GoodBadUgly", "MonkeyKingdom".
    splits : list
        List containing tuples of train set and test set indices. Each tuple corresponds to a cross-validation split. Length of the list is the number of splits.
    task_label : str
        Name of the task currently being classified.
    connectivity_measure : str
        Connectivity measure currently being used for classification.
    results : dict
        Dictionary storing the results of the classification.
    classify : str
        What is being classified. Can be "Runs", "Tasks", or "Subjects".
    pooled_tasks : bool
        Whether the classification is between multiple tasks.

    Returns
    -------
    dict
        Dictionary storing the results of the classification.
    """
    print("Fitting classifier for each cross-validation split")
    # fit the classifier for each cross validation split in serial
    for train, test in tqdm(
        splits,
        desc=f"{classify}, {connectivity_measure}, {task_label}",
        leave=True,
    ):
        results = classify_connectivity(
            connectomes,
            classes,
            task_label,
            connectivity_measure,
            train,
            test,
            results,
            classify,
            pooled_tasks,
        )

    return results


def do_cross_validation(
    classify, task, cv_splits, connectivity_measure, data, root
):
    """Select cross-validation scheme, set-up output directory, plot train-test splits, run classification and then save the results.

    Parameters
    ----------
    classify : str
        What is being classified. Can be "Runs", "Tasks", or "Subjects".
    task : str
        Name of the task currently being classified.
    cv_splits : int
        Number of cross-validation splits.
    connectivity_measure : str
        Connectivity measure currently being used for classification.
    data : pandas DataFrame
        DataFrame containing the time series, subject ids, run labels, tasks and connectomes.
    root : str
        Path to root directory for storing results.

    Returns
    -------
    pandas DataFrame
        DataFrame containing the results of the classification.
    """
    cov = connectivity_measure.split(" ")[0]
    if isinstance(task, list) and len(task) > 2:
        pooled_tasks = True
    else:
        pooled_tasks = False

    if pooled_tasks:
        if classify == "Subjects":
            cv_test_size = 0.25
            cv = GroupShuffleSplit(
                n_splits=cv_splits, random_state=0, test_size=cv_test_size
            )
            cv_scheme = "GroupShuffleSplit"
        elif classify in ["Tasks", "Runs"]:
            cv = LeavePGroupsOut(5)
            cv_scheme = "LeavePGroupsOut"
    else:
        if classify in ["Runs", "Subjects"]:
            cv_test_size = 0.25
            cv = GroupShuffleSplit(
                n_splits=cv_splits, random_state=0, test_size=cv_test_size
            )
            cv_scheme = "GroupShuffleSplit"
        elif classify == "Tasks":
            cv = LeavePGroupsOut(5)
            cv_scheme = "LeavePGroupsOut"

    results_dir = os.path.join(
        root, f"{classify}_{cov}_{cv_scheme}_{cv_splits}"
    )
    os.makedirs(results_dir, exist_ok=True)

    # select data based current task and whatever is being classified
    # get classes and groups
    data, task_label, classes, groups = _select_data(
        data, classify, task, pooled_tasks
    )
    # pick specific connectome to classify based on Sconnectivity measure
    connectomes = np.array(data[connectivity_measure].values.tolist())
    # get train and test splits
    splits = [split for split in cv.split(connectomes, classes, groups)]
    # too many splits for Tasks from LPGO scheme, so randomly select some
    if pooled_tasks:
        if classify in ["Tasks", "Runs"]:
            rng = np.random.default_rng(1)
            select_splits = rng.choice(len(splits), cv_splits, replace=False)
            splits = [splits[i] for i in select_splits]
    else:
        if classify == "Tasks":
            rng = np.random.default_rng(1)
            select_splits = rng.choice(len(splits), cv_splits, replace=False)
            splits = [splits[i] for i in select_splits]
    # plot the train/test cross-validation splits
    _plot_cv_indices(
        splits,
        connectomes,
        classes,
        groups,
        results_dir,
        task_label,
        connectivity_measure,
    )
    # initialize lists to store results
    results = {
        "LinearSVC_accuracy": [],
        "LinearSVC_auc": [],
        "LinearSVC_predicted_class": [],
        "true_class": [],
        "connectivity": [],
        "classes": [],
        "groups": [],
        "train_sets": [],
        "test_sets": [],
        "task_label": [],
        "Dummy_auc": [],
        "Dummy_accuracy": [],
        "Dummy_predicted_class": [],
        "balanced_accuracy": [],
        "dummy_balanced_accuracy": [],
        # "weights": [],
    }
    results = cross_validate(
        connectomes,
        classes,
        splits,
        task_label,
        connectivity_measure,
        results,
        classify,
        pooled_tasks,
    )
    results_df = pd.DataFrame(results)
    # save the results
    results_file = os.path.join(
        results_dir,
        f"{task_label}_{connectivity_measure}_results.pkl",
    )
    results_df.to_pickle(results_file)
    # do_plots(results, results_dir, classify)

    return results_df


def _plot_cv_indices(
    splits,
    X,
    y,
    group,
    out_dir,
    task_label,
    connectivity_measure,
    lw=10,
):
    """Create a sample plot for indices of a cross-validation object."""
    fig, ax = plt.subplots()
    cmap_data = plt.cm.tab20
    cmap_cv = plt.cm.coolwarm
    n_splits = len(splits)
    _, y = np.unique(y, return_inverse=True)
    _, group = np.unique(group, return_inverse=True)
    # Generate the training/testing visualizations for each CV split
    for ii, (tr, tt) in enumerate(splits):
        # Fill in indices with the training/test groups
        indices = np.array([np.nan] * len(X))
        tt = tt.astype(int)
        tr = tr.astype(int)
        indices[tt] = 1
        indices[tr] = 0
        # Visualize the results
        ax.scatter(
            range(len(indices)),
            [ii + 0.5] * len(indices),
            c=indices,
            marker="_",
            lw=lw,
            cmap=cmap_cv,
            vmin=-0.2,
            vmax=1.2,
        )
    # Plot the data classes and groups at the end
    ax.scatter(
        range(len(X)),
        [ii + 1.5] * len(X),
        c=y,
        marker="_",
        lw=lw,
        cmap=cmap_data,
    )
    ax.scatter(
        range(len(X)),
        [ii + 2.5] * len(X),
        c=group,
        marker="_",
        lw=lw,
        cmap=cmap_data,
    )
    # Formatting
    yticklabels = [*range(n_splits)] + ["class", "group"]
    ax.set(
        yticks=np.arange(n_splits + 2) + 0.5,
        yticklabels=yticklabels,
        xlabel="Sample index",
        ylabel="CV iteration",
        ylim=[n_splits + 2.2, -0.2],
    )
    split_dir = os.path.join(out_dir, "test_train_splits")
    os.makedirs(split_dir, exist_ok=True)
    ax.set_title(f"Train/test splits for classifying {task_label}")
    plot_file = f"{task_label}_{connectivity_measure}_cv_indices.png"
    plot_file = os.path.join(split_dir, plot_file)
    fig.savefig(plot_file, bbox_inches="tight")
    plt.close(fig)
