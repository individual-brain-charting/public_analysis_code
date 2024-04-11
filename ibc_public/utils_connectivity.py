import os
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ibc_public.utils_data import DERIVATIVES, get_subject_session
from nilearn.connectome import sym_matrix_to_vec
from nilearn.image import high_variance_confounds
from nilearn.maskers import NiftiLabelsMasker
from sklearn.base import clone
from sklearn.covariance import (
    GraphicalLassoCV,
    GraphicalLasso,
    LedoitWolf,
    EmpiricalCovariance,
    empirical_covariance,
    shrunk_covariance,
)
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    balanced_accuracy_score,
)
from sklearn.model_selection import GroupShuffleSplit, LeavePGroupsOut
from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier
from tqdm import tqdm
import itertools

HCP_ROOT = "/storage/store/data/HCP900"


def _get_tr(task, dataset="ibc"):
    if dataset == "ibc":
        if task == "RestingState":
            repetition_time = 0.76
        elif task in ["GoodBadUgly", "Raiders", "MonkeyKingdom", "Mario"]:
            repetition_time = 2
        else:
            raise ValueError(f"Unknown task {task}")
    elif dataset == "hcp":
        repetition_time = 0.72

    return repetition_time


def _get_niftis(task, subject, session, dataset="ibc"):
    if dataset == "ibc":
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
            # elif task == "RestingState":
            #     if run_label == "dir-ap":
            #         run_label = "run-01"
            #     else:
            #         run_label = "run-02"
            run_labels.append(run_label)
            run_files.append(run)
    elif dataset == "hcp":
        run_files = glob(
            os.path.join(
                HCP_ROOT,
                subject,
                "MNINonLinear",
                "Results",
                session,
                f"{session}.nii.gz",
            )
        )
        run_labels = []
        for run in run_files:
            direction = session.split("_")[2]
            if task == "REST":
                rest_ses = session.split("_")[1]
                if direction == "LR" and rest_ses == "REST1":
                    run_label = "run-01"
                elif direction == "RL" and rest_ses == "REST1":
                    run_label = "run-02"
                elif direction == "LR" and rest_ses == "REST2":
                    run_label = "run-03"
                elif direction == "RL" and rest_ses == "REST2":
                    run_label = "run-04"
            else:
                if direction == "LR":
                    run_label = "run-01"
                elif direction == "RL":
                    run_label = "run-02"
            run_labels.append(run_label)

    return run_files, run_labels


def _get_confounds(task, run_num, subject, session, dataset="ibc"):
    if dataset == "ibc":
        return glob(
            os.path.join(
                DERIVATIVES,
                subject,
                session,
                "func",
                f"rp*{task}*{run_num}_bold*",
            )
        )[0]
    elif dataset == "hcp":
        return glob(
            os.path.join(
                HCP_ROOT,
                subject,
                "MNINonLinear",
                "Results",
                session,
                "Movement_Regressors_dt.txt",
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
    dummy_auc,
    dummy_accuracy,
    dummy_predictions,
    balanced_accuracy,
    dummy_balanced_accuracy,
    # weights,
):
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


def _find_hcp_subjects(session_names):
    # load csv file with subject ids and task availability
    df = pd.read_csv(os.path.join(HCP_ROOT, "unrestricted_hcp_s900.csv"))
    df = df[df["3T_Full_MR_Compl"] == True]
    subs = list(df["Subject"].astype(str))
    sub_ses = {}
    for sub in subs:
        sub_ses[sub] = session_names

    return sub_ses


def get_ses_modality(task, dataset="ibc"):
    """Get session numbers and modality for given task

    Parameters
    ----------
    task : str
        name of the task
    dataset : str
        name of the dataset, can be ibc or hcp

    Returns
    -------
    sub_ses : dict
        dictionary with subject as key and session number as value
    modality : str
        modality of the task
    """
    if dataset == "ibc":
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
        elif task == "Mario":
            # session names with mario gameplay data
            session_names = ["mario1"]
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

    elif dataset == "hcp":
        if task == "REST":
            # session names with RestingState state task data
            session_names = ["rfMRI_REST1_LR", "rfMRI_REST2_RL"]
        else:
            # session names with diffusion data
            session_names = [f"tfMRI_{task}_LR", f"tfMRI_{task}_RL"]
        modality = "functional"

        # create dictionary with subject as key and session number as value
        sub_ses = _find_hcp_subjects(session_names)

    return sub_ses, modality


def get_time_series(task, atlas, cache, dataset="ibc"):
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
    repetition_time = _get_tr(task, dataset)
    print(f"Getting time series for {task}...")
    masker = NiftiLabelsMasker(
        labels_img=atlas.maps,
        standardize=True,
        low_pass=0.2,
        high_pass=0.01,
        t_r=repetition_time,
        verbose=0,
        # memory=Memory(location=cache),
        memory_level=0,
        n_jobs=1,
    ).fit()
    subject_sessions, _ = get_ses_modality(task, dataset)
    if dataset == "hcp":
        subject_sessions = dict(itertools.islice(subject_sessions.items(), 50))
    all_time_series = []
    subject_ids = []
    run_labels_ = []
    for subject, sessions in tqdm(
        subject_sessions.items(), desc=task, total=len(subject_sessions)
    ):
        for session in sorted(sessions):
            runs, run_labels = _get_niftis(task, subject, session, dataset)
            for run, run_label in zip(runs, run_labels):
                print(task)
                print(subject, session)
                print(
                    glob(
                        os.path.join(
                            DERIVATIVES,
                            subject,
                            session,
                            "func",
                            f"rp*{task}*{run_label}_bold*",
                        )
                    )
                )
                print(run_label)

                confounds = _get_confounds(
                    task, run_label, subject, session, dataset
                )
                compcor_confounds = high_variance_confounds(run)
                confounds = np.hstack(
                    (np.loadtxt(confounds), compcor_confounds)
                )
                time_series = masker.transform(run, confounds=confounds)
                all_time_series.append(time_series)
                subject_ids.append(subject)
                run_labels_.append(run_label)

    tasks_ = [task for _ in range(len(all_time_series))]

    data = _update_data(
        data, all_time_series, subject_ids, run_labels_, tasks_
    )
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
    try:
        cv = cov_estimator_.fit(X)
    except FloatingPointError as error:
        if isinstance(cov_estimator_, GraphicalLassoCV):
            print(
                "Caught a FloatingPointError, shrinking covariance beforehand..."
            )
            X = empirical_covariance(X, assume_centered=True)
            X = shrunk_covariance(X, shrinkage=1)
            cov_estimator_ = GraphicalLasso(
                alpha=0.52, verbose=0, mode="cd", covariance="precomputed"
            )
            cv = cov_estimator_.fit(X)
        else:
            raise error
    cv_correlation = sym_matrix_to_vec(cv.covariance_, discard_diagonal=True)
    cv_partial = sym_matrix_to_vec(-cv.precision_, discard_diagonal=True)

    return (cv_correlation, cv_partial)


def get_connectomes(cov, data, n_jobs):
    # covariance estimator
    if cov == "Graphical-Lasso":
        cov_estimator = GraphicalLassoCV(
            verbose=11, n_jobs=n_jobs, assume_centered=True
        )
    elif cov == "Ledoit-Wolf":
        cov_estimator = LedoitWolf(assume_centered=True)
    elif cov == "Unregularized":
        cov_estimator = EmpiricalCovariance(assume_centered=True)
    time_series = data["time_series"].tolist()
    connectomes = []
    for ts in tqdm(time_series, desc=cov, leave=True):
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
