import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.svm import LinearSVC
from ibc_public.utils_data import DERIVATIVES
import os
import pandas as pd
from joblib import Memory
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from glob import glob
from nilearn.maskers import NiftiLabelsMasker
from sklearn.base import clone
from nilearn.connectome import sym_matrix_to_vec
from tqdm import tqdm
from sklearn.covariance import GraphicalLassoCV, LedoitWolf
from sklearn.model_selection import LeavePGroupsOut, GroupShuffleSplit
from ibc_public.utils_data import get_subject_session


sns.set_theme(context="talk", style="whitegrid")


def _get_tr(task):
    if task == "RestingState":
        repetition_time = 0.76
    elif task in ["GoodBadUgly", "Raiders", "MonkeyKingdom"]:
        repetition_time = 2
    else:
        raise ValueError(f"Unknown task {task}")

    return repetition_time


def _get_niftis(task, subject, session, root=DERIVATIVES):
    _run_files = glob(
        os.path.join(
            DERIVATIVES,
            subject,
            session,
            "func",
            f"wrdc*{task}*.nii.gz",
        )
    )
    run_labels = []
    run_files = []
    for run in _run_files:
        run_label = os.path.basename(run).split("_")[-2]
        run_num = run_label.split("-")[-1]
        # skip repeats of run-01, run-02, run-03 done at the end of
        # the sessions in Raiders and GoodBadUgly
        if task == "Raiders" and int(run_num) > 10:
            continue
        elif task == "GoodBadUgly" and int(run_num) > 18:
            continue
        run_labels.append(run_label)
        run_files.append(run)

    return run_files, run_labels


def _get_confounds(run_num, subject, session, root=DERIVATIVES):
    return glob(
        os.path.join(
            DERIVATIVES,
            subject,
            session,
            "func",
            f"rp*{run_num}_bold*",
        )
    )[0]


def _update_data(data, all_time_series, subject_ids, run_labels, tasks):
    data["time_series"].extend(all_time_series)
    data["subject_ids"].extend(subject_ids)
    data["run_labels"].extend(run_labels)
    data["tasks"].extend(tasks)

    return data


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
):
    results["accuracy"].append(accuracy)
    results["auc"].append(auc)
    # store test labels and predictions
    results["predicted_class"].append(predictions)
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
        grouping = "Runs"
    results["groups"].append(grouping)

    return results


def _select_data(data, classify, task):
    if classify in ["Runs", "Subjects"]:
        assert type(task) == str
        task_label = task
        data = data[data["tasks"] == task]
        if classify == "Runs":
            classes = data["run_labels"].to_numpy(dtype=object)
            groups = data["subject_ids"].to_numpy(dtype=object)
        elif classify == "Subjects":
            classes = data["subject_ids"].to_numpy(dtype=object)
            groups = data["run_labels"].to_numpy(dtype=object)
    elif classify == "Tasks":
        assert type(task) == list
        task_label = " vs ".join(task)
        data = data[data["tasks"].isin(task)]
        classes = data["tasks"].to_numpy(dtype=object)
        groups = data["subject_ids"].to_numpy(dtype=object)

    return data, task_label, classes, groups


def _clean(series):
    series = series.split()
    l = []
    for i in series:
        l.append(i.strip("'[]"))
    series = np.asarray(l)
    return series


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
    if task == "GoodBadUgly":
        # session names with movie task data
        session_names = ["BBT1", "BBT2", "BBT3"]
    elif task == "MonkeyKingdom":
        # session names with movie task data
        session_names = ["monkey_kingdom"]
    elif task == "Raiders":
        # session names with movie task data
        session_names = ["raiders1", "raiders2"]
    elif task == "RestingState":
        # session names with RestingState state task data
        session_names = ["mtt1", "mtt2"]
    elif task == "DWI":
        # session names with diffusion data
        session_names = ["anat1"]
    # get session numbers for each subject
    # returns a list of tuples with subject and session number
    subject_sessions = sorted(get_subject_session(session_names))
    # convert the tuples to a dictionary with subject as key and session
    # number as value
    sub_ses = {}
    # for dwi, with anat1 as session_name, get_subject_session returns wrong
    # session number for sub-01 and sub-15
    # setting it to ses-12 for these subjects
    if task == "DWI":
        modality = "structural"
        sub_ses = {
            subject_session[0]: "ses-12"
            if subject_session[0] in ["sub-01", "sub-15"]
            else subject_session[1]
            for subject_session in subject_sessions
        }
    else:
        # for fMRI tasks, for one of the movies, ses no. 13 pops up for sub-11
        # and sub-12, so skipping that
        modality = "functional"
        for subject_session in subject_sessions:
            if (
                subject_session[0] in ["sub-11", "sub-12"]
                and subject_session[1] == "ses-13"
            ):
                continue
            # initialize a subject as key and an empty list as the value
            # and populate the list with session numbers
            # try-except block is used to avoid overwriting the list for subject
            try:
                sub_ses[subject_session[0]]
            except KeyError:
                sub_ses[subject_session[0]] = []
            sub_ses[subject_session[0]].append(subject_session[1])
    return sub_ses, modality


def get_time_series(task, atlas, cache):
    """Use NiftiLabelsMasker to extract time series from nifti files.

    Parameters
    ----------
    tasks : list
        List of tasks to extract time series from.
    atlas : atlas object
        Atlas to use for extracting time series.
    cache : str
        Path to cache directory.

    Returns
    -------
    pandas DataFrame
        DataFrame containing the time series, subject ids, run labels, and tasks.
    """
    data = {
        "time_series": [],
        "subject_ids": [],
        "run_labels": [],
        "tasks": [],
    }
    repetition_time = _get_tr(task)
    print(f"Getting time series for {task}...")
    masker = NiftiLabelsMasker(
        labels_img=atlas.maps,
        standardize=True,
        low_pass=0.2,
        high_pass=0.01,
        t_r=repetition_time,
        verbose=0,
        memory=Memory(location=cache),
        memory_level=3,
    ).fit()
    subject_sessions, _ = get_ses_modality(task)
    all_time_series = []
    subject_ids = []
    run_labels = []
    for subject, sessions in subject_sessions.items():
        for session in sorted(sessions):
            runs, run_labels_ = _get_niftis(task, subject, session)
            for run, run_label in zip(runs, run_labels_):
                confounds = _get_confounds(run_label, subject, session)
                time_series = masker.transform(run, confounds=confounds)
                all_time_series.append(time_series)
                subject_ids.append(subject)
                # Label dir-ap as run-01 and dir-pa as run-02
                if task == "RestingState":
                    if run_label == "dir-ap":
                        run_label = "run-01"
                    else:
                        run_label = "run-02"
                run_labels.append(run_label)
    tasks_ = [task for _ in range(len(all_time_series))]

    data = _update_data(data, all_time_series, subject_ids, run_labels, tasks_)

    return pd.DataFrame(data)


def calculate_connectivity(X, cov_estimator):
    """Fit given covariance estimator to data and return correlation and partial correlation.

    Parameters
    ----------
    X : numpy array
        Time series data.
    cov_estimator : sklearn estimator
        Covariance estimator to fit to data.

    Returns
    -------
    tuple of numpy arrays
        First array is the correlation matrix, second array is the partial
    """
    # get the connectivity measure
    cov_estimator_ = clone(cov_estimator)
    cv = cov_estimator_.fit(X)
    cv_correlation = sym_matrix_to_vec(cv.covariance_, discard_diagonal=True)
    cv_partial = sym_matrix_to_vec(-cv.precision_, discard_diagonal=True)

    return (cv_correlation, cv_partial)


def get_connectomes(cov, data, n_jobs):
    print(f"[{cov} cov estimator]")
    # covariance estimator
    if cov == "GLC":
        cov_estimator = GraphicalLassoCV(verbose=0, n_jobs=n_jobs)
    elif cov == "LedoitWolf":
        cov_estimator = LedoitWolf()
    time_series = data["time_series"].tolist()
    connectomes = []
    for ts in tqdm(time_series):
        connectome = calculate_connectivity(ts, cov_estimator)
        connectomes.append(connectome)
    correlation = np.asarray([connectome[0] for connectome in connectomes])
    partial_correlation = np.asarray(
        [connectome[1] for connectome in connectomes]
    )
    data[f"{cov} correlation"] = correlation.tolist()
    data[f"{cov} partial correlation"] = partial_correlation.tolist()

    return data


def classify_connectivity(
    connectomes,
    classes,
    task_label,
    connectivity_measure,
    train,
    test,
    results,
    classify,
):
    """Classify the given connectomes using the given classes.

    Parameters
    ----------
    connectomes : numpy array
        Array containing the connectomes to classify. Each row is a vectorised connectome.
    classes : numpy array
        Array containing the classes for each connectome.
    tasks : list
        list of tasks currently being classified.
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

    Returns
    -------
    dict
        Dictionary storing the results of the classification.
    """
    # fit the classifier
    classifier = LinearSVC().fit(connectomes[train], classes[train])
    # make predictions for the left-out test subjects
    predictions = classifier.predict(connectomes[test])
    accuracy = accuracy_score(classes[test], predictions)
    if classify in ["Runs", "Subjects"]:
        auc = 0
    else:
        auc = roc_auc_score(
            classes[test], classifier.decision_function(connectomes[test])
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
):
    """Cross validate the given connectomes using the given classes.

    Parameters
    ----------
    connectomes : numpy array
        Array containing the connectomes to classify. Each row is a vectorised connectome.
    classes : numpy array
        Array containing the class for each connectome.
    splits : list
        List containing tuples of train set and test set indices. Each tuple corresponds to a cross-validation split.
    task_label : str
        Name of the task currently being classified.
    connectivity_measure : str
        Connectivity measure currently being used for classification.
    results : dict
        Dictionary storing the results of the classification.
    classify : str
        What is being classified. Can be "Runs", "Tasks", or "Subjects".

    Returns
    -------
    dict
        Dictionary storing the results of the classification.
    """
    print("Fitting classifier for each cross-validation split")
    # fit the classifier for each cross validation split in serial
    for train, test in tqdm(splits):
        results = classify_connectivity(
            connectomes,
            classes,
            task_label,
            connectivity_measure,
            train,
            test,
            results,
            classify,
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
    print(f"[{classify} classification], [{connectivity_measure}], [{task}]")

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
    data, task_label, classes, groups = _select_data(data, classify, task)
    # pick specific connectome to classify based on Sconnectivity measure
    connectomes = np.array(data[connectivity_measure].values.tolist())
    # get train and test splits
    splits = [split for split in cv.split(connectomes, classes, groups)]
    # too many splits for Tasks from LPGO scheme, so randomly select some
    if classify == "Tasks":
        rng = np.random.default_rng(1)
        select_splits = rng.choice(len(splits), cv_splits, replace=False)
        splits = [splits[i] for i in select_splits]
    # plot the train/test cross-validation splits
    plot_cv_indices(
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
        "accuracy": [],
        "auc": [],
        "predicted_class": [],
        "true_class": [],
        "connectivity": [],
        "classes": [],
        "groups": [],
        "train_sets": [],
        "test_sets": [],
        "task_label": [],
    }
    results = cross_validate(
        connectomes,
        classes,
        splits,
        task_label,
        connectivity_measure,
        results,
        classify,
    )

    results_df = pd.DataFrame(results)
    # save the results
    results_file = os.path.join(
        results_dir,
        f"{task_label}_{connectivity_measure}_results.csv",
    )
    results_df.to_csv(results_file)
    # do_plots(results, results_dir, classify)

    return results_df


def do_plots(results, results_dir, classify):
    results = chance_level(results)
    print("Plotting results...")
    # plot accuracy as barplots
    plot_accuracy(results, results_dir, classify)
    # plot confusion matrices
    plot_confusion(results, results_dir)


def chance_level(df):
    dfs = []
    tasks = df["task_label"].unique()
    conn_measures = df["connectivity"].unique()
    for task in tasks:
        for conn_measure in conn_measures:
            task_df = df[
                (df["connectivity"] == conn_measure)
                & (df["task_label"] == task)
            ]
            classes = []
            n_splits = len(task_df)
            for _, row in task_df.iterrows():
                classes.extend(task_df["true_class"].tolist())
                classes.extend(task_df["predicted_class"].tolist())
            classes = np.concatenate(classes)
            classes = np.asarray(classes)
            classes = np.unique(classes)
            task_df["labels"] = [classes for split in range(n_splits)]
            task_df["n_labels"] = [len(classes) for split in range(n_splits)]
            task_df["chance"] = [1 / len(classes) for split in range(n_splits)]
            dfs.append(task_df)
    df = pd.concat(dfs)
    return df


def plot_accuracy(all_results, results_dir, classify):
    if classify in ["Runs", "Subjects"]:
        hue_order = [
            "RestingState",
            "Raiders",
            "GoodBadUgly",
            "MonkeyKingdom",
        ]
    else:
        hue_order = [
            "RestingState vs Raiders",
            "RestingState vs GoodBadUgly",
            "RestingState vs MonkeyKingdom",
        ]
    # plot accuracy
    task_df = all_results.copy()
    sns.barplot(
        task_df,
        x="connectivity",
        y="accuracy",
        hue="task_label",
        hue_order=hue_order,
    )
    sns.barplot(
        task_df,
        x="connectivity",
        y="chance",
        hue="task_label",
        palette="pastel",
        hue_order=hue_order,
    )
    plot_file = os.path.join(results_dir, f"accuracy.png")
    legend = plt.legend(
        framealpha=0, loc="center left", bbox_to_anchor=(1, 0.5)
    )
    # remove legend repetition for chance level
    for i, (text, handle) in enumerate(
        zip(legend.texts, legend.legend_handles)
    ):
        if i > 3:
            text.set_visible(False)
            handle.set_visible(False)
    legend.set_title("Task")
    plt.xlabel("FC measure")
    plt.ylabel("Accuracy")
    plt.savefig(plot_file, bbox_inches="tight", transparent=True)
    plt.close()


def plot_auc(all_results, results_dir):
    # plot auc
    task_df = all_results.copy()
    sns.barplot(
        task_df,
        x="connectivity",
        y="auc",
        hue="task",
        palette=sns.color_palette()[1:],
    )
    plot_file = os.path.join(results_dir, f"auc.png")
    legend = plt.legend(
        framealpha=0, loc="center left", bbox_to_anchor=(1, 0.5)
    )
    # remove legend repetition for chance level
    for i, (text, handle) in enumerate(
        zip(legend.texts, legend.legend_handles)
    ):
        if i > 3:
            text.set_visible(False)
            handle.set_visible(False)
    plt.xlabel("FC measure")
    plt.ylabel("AUC")
    plt.savefig(plot_file, bbox_inches="tight", transparent=True)
    plt.close()


def plot_confusion(all_results, results_dir):
    # plot confusion matrices
    confusion_dir = os.path.join(results_dir, "confusion_mats")
    os.makedirs(confusion_dir, exist_ok=True)
    tasks = all_results["task_label"].unique()
    connectivity_measures = all_results["connectivity"].unique()
    for task in tasks:
        for conn_measure in connectivity_measures:
            df = all_results[
                (all_results["connectivity"] == conn_measure)
                & (all_results["task_label"] == task)
            ]
            true_class = np.concatenate(df["true_class"].to_numpy())
            predicted_class = np.concatenate(df["predicted_class"].to_numpy())
            cm = confusion_matrix(
                true_class, predicted_class, normalize="true"
            )
            fig, ax = plt.subplots()
            pos = ax.matshow(cm, cmap=plt.cm.Blues)
            ax.set_xticks(np.arange(df["n_labels"].iloc[0]))
            ax.xaxis.tick_bottom()
            ax.set_yticks(np.arange(df["n_labels"].iloc[0]))
            ax.set_xticklabels(df["labels"].iloc[0], rotation=45)
            ax.set_yticklabels(df["labels"].iloc[0])
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            fig.colorbar(pos, ax=ax)
            ax.set_title(f"Classifying {task} using {conn_measure}")
            plot_file = os.path.join(
                confusion_dir,
                f"{task}_{conn_measure}_confusion.png",
            )
            plt.savefig(plot_file, bbox_inches="tight")
            plt.close()


def plot_cv_indices(
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
