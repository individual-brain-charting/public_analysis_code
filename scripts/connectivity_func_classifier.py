import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.svm import LinearSVC
from ibc_public.utils_connectivity import get_ses_modality
from nilearn.connectome import ConnectivityMeasure
from nilearn import datasets
from ibc_public.utils_data import DERIVATIVES
import os
import pandas as pd
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from glob import glob
from nilearn.maskers import NiftiLabelsMasker
from sklearn.covariance import GraphicalLassoCV


def plot_cv_indices(
    cv,
    X,
    y,
    group,
    n_splits,
    out_dir,
    task,
    classify,
    connectivity_measure,
    lw=10,
):
    """Create a sample plot for indices of a cross-validation object."""
    fig, ax = plt.subplots()
    cmap_data = plt.cm.tab20
    cmap_cv = plt.cm.coolwarm
    _, y = np.unique(y, return_inverse=True)
    _, group = np.unique(group, return_inverse=True)
    # Generate the training/testing visualizations for each CV split
    for ii, (tr, tt) in enumerate(cv.split(X=X, y=y, groups=group)):
        # Fill in indices with the training/test groups
        indices = np.array([np.nan] * len(X))
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
    ax.set_title("{}".format(type(cv).__name__), fontsize=15)
    split_dir = os.path.join(out_dir, "test_train_splits")
    os.makedirs(split_dir, exist_ok=True)
    ax.set_title(f"Train/test splits for classifying {classify} in {task}")
    plot_file = f"{task}_{connectivity_measure}_{classify}_cv_indices.png"
    plot_file = os.path.join(split_dir, plot_file)
    fig.savefig(plot_file, bbox_inches="tight")
    plt.close(fig)


def get_time_series(task, atlas, cache):
    if task == "GoodBadUgly":
        # session names with GoodBadUgly movie data
        session_names = ["BBT1", "BBT2", "BBT3"]
        repetition_time = 2
    elif task == "MonkeyKingdom":
        # session names with MonkeyKingdom movie data
        session_names = ["monkey_kingdom"]
        repetition_time = 2
    elif task == "Raiders":
        # session names with Raiders movie data
        session_names = ["raiders1", "raiders2"]
        repetition_time = 2
    elif task == "RestingState":
        # session names with RestingState movie data
        session_names = ["mtt1", "mtt2"]
        repetition_time = 0.76
    else:
        raise ValueError(f"Unknown task {task}")

    masker = NiftiLabelsMasker(
        labels_img=atlas.maps,
        standardize=True,
        low_pass=0.2,
        high_pass=0.01,
        t_r=repetition_time,
        verbose=1,
        memory=cache,
    ).fit()
    subject_sessions, _ = get_ses_modality(task)
    all_time_series = []
    subject_ids = []
    run_nums = []
    for subject, sessions in subject_sessions.items():
        for ses, session in enumerate(sorted(sessions)):
            runs = glob(
                os.path.join(
                    DERIVATIVES,
                    subject,
                    session,
                    "func",
                    f"wrdc*{task}*.nii.gz",
                )
            )
            for run in runs:
                run_num = os.path.basename(run).split("_")[-2]
                # skip repeats of run-01, run-02, run-03 done at the end of
                # the sessions in Raiders and GoodBadUgly
                if task == "Raiders" and int(run_num.split("-")[-1]) > 10:
                    continue
                elif (
                    task == "GoodBadUgly" and int(run_num.split("-")[-1]) > 18
                ):
                    continue
                confounds = glob(
                    os.path.join(
                        DERIVATIVES,
                        subject,
                        session,
                        "func",
                        f"rp*{run_num}_bold*",
                    )
                )[0]
                time_series = masker.transform(run, confounds=confounds)
                all_time_series.append(time_series)
                subject_ids.append(subject)
                if task == "RestingState":
                    if ses == 0:
                        if run_num == "dir-ap":
                            run_num = "run-01"
                        else:
                            run_num = "run-02"
                    elif ses == 1:
                        if run_num == "dir-ap":
                            run_num = "run-03"
                        else:
                            run_num = "run-04"
                run_nums.append(run_num)
    return all_time_series, subject_ids, run_nums


def classify_connectivity(
    all_time_series,
    subject_ids,
    run_nums,
    task,
    connectivity_measure,
    classify,
    out_dir,
    cv_splits=5,
    cv_test_size=0.25,
):
    data = np.asarray(all_time_series, dtype=object)
    if classify == "Runs":
        classes = run_nums
        groups = subject_ids
    elif classify == "Subjects":
        classes = subject_ids
        groups = run_nums
    else:
        raise ValueError(f"Can't classify {classify}")
    classes = np.asarray(classes, dtype=object)
    groups = np.asarray(groups, dtype=object)
    # cross-validation
    cv = GroupShuffleSplit(
        n_splits=cv_splits, random_state=0, test_size=cv_test_size
    )
    # plot the cross-validation splits
    plot_cv_indices(
        cv,
        data,
        classes,
        groups,
        cv_splits,
        out_dir,
        task,
        classify,
        connectivity_measure,
    )
    # initialize lists to store results
    scores = []
    predicted_labels = []
    expected_labels = []
    connectivity_measures = []
    classifying = []
    grouping = []
    train_sets = []
    test_sets = []
    tasks = []
    # fit the classifier
    for train, test in cv.split(data, classes, groups):
        glc_estimator = GraphicalLassoCV(verbose=2, n_jobs=4)
        # get the connectivity measure
        connectivity = ConnectivityMeasure(
            kind=connectivity_measure,
            cov_estimator=glc_estimator,
            vectorize=True,
        )
        # build vectorized connectomes for classes in the train set
        connectomes = connectivity.fit_transform(data[train])
        # fit the classifier
        classifier = LinearSVC().fit(connectomes, classes[train])
        # make predictions for the left-out test subjects
        predictions = classifier.predict(connectivity.transform(data[test]))
        score = accuracy_score(classes[test], predictions)
        # store the accuracy for this cross-validation fold
        scores.append(score)
        # store test labels and predictions
        predicted_labels.append(predictions)
        expected_labels.append(classes[test])
        # store the connectivity measure
        connectivity_measures.append(connectivity_measure)
        # store the classification scenario
        classifying.append(classify)
        # store the group
        grouping.append("Subjects" if classify == "Runs" else "Runs")
        # store the train and test sets
        train_sets.append(train)
        test_sets.append(test)
        # store the task
        tasks.append(task)
    results = pd.DataFrame(
        {
            "accuracy": scores,
            "connectivity": connectivity_measures,
            "classes": classifying,
            "groups": grouping,
            "predicted_labels": predicted_labels,
            "expected_labels": expected_labels,
            "train_sets": train_sets,
            "test_sets": test_sets,
            "task": tasks,
        }
    )
    # save the results
    results_file = os.path.join(
        out_dir, f"{task}_{connectivity_measure}_{classify}_results.csv"
    )
    results.to_csv(results_file)
    return results


def pipeline(
    task,
    atlas,
    cache,
    results_dir,
    connectivity_measures,
    classifying_scenarios,
):
    # get the time series
    all_time_series, subject_ids, run_nums = get_time_series(
        task, atlas, cache
    )
    # classify the time series
    task_results = []
    for connectivity_measure in connectivity_measures:
        for classifying_scenario in classifying_scenarios:
            task_results.append(
                classify_connectivity(
                    all_time_series,
                    subject_ids,
                    run_nums,
                    task,
                    connectivity_measure,
                    classifying_scenario,
                    results_dir,
                    cv_splits=50,
                    cv_test_size=0.25,
                )
            )
    return pd.concat(task_results)


def _clean(series):
    series = series.split()
    l = []
    for i in series:
        l.append(i.strip("'[]"))
    series = np.asarray(l)
    return series


def chance_level(df):
    dfs = []
    tasks = df["task"].unique()
    conn_measures = df["connectivity"].unique()
    for task in tasks:
        for conn_measure in conn_measures:
            for classify in ["Runs", "Subjects"]:
                task_df = df[
                    (df["classes"] == classify)
                    & (df["connectivity"] == conn_measure)
                    & (df["task"] == task)
                ]
                classes = []
                n_splits = len(task_df)
                for _, row in task_df.iterrows():
                    classes.extend(task_df["expected_labels"].tolist())
                    classes.extend(task_df["predicted_labels"].tolist())
                classes = np.asarray(classes)
                classes = np.unique(classes)
                task_df["labels"] = [classes for split in range(n_splits)]
                task_df["n_labels"] = [
                    len(classes) for split in range(n_splits)
                ]
                task_df["chance"] = [
                    1 / len(classes) for split in range(n_splits)
                ]
                dfs.append(task_df)
    df = pd.concat(dfs)
    return df


def plot_accuracy(all_results, results_dir):
    # plot accuracy
    for classify in ["Runs", "Subjects"]:
        classification_df = all_results[all_results["classes"] == classify]
        sns.barplot(
            classification_df, x="connectivity", y="accuracy", hue="task"
        )
        sns.barplot(
            classification_df,
            x="connectivity",
            y="chance",
            hue="task",
            palette="pastel",
        )
        plot_file = os.path.join(results_dir, f"{classify}_accuracy.png")
        legend = plt.legend(framealpha=0)
        # remove legend repetition for chance level
        for i, (text, handle) in enumerate(
            zip(legend.texts, legend.legend_handles)
        ):
            if i > 3:
                text.set_visible(False)
                handle.set_visible(False)
        legend.set_title("Tasks")
        plt.xlabel("Accuracy")
        plt.title(f"Accuracy for classifying {classify}")
        plt.savefig(plot_file, bbox_inches="tight")
        plt.close()


def plot_confusion(all_results, results_dir):
    # plot confusion matrices
    confusion_dir = os.path.join(results_dir, "confusion_mats")
    os.makedirs(confusion_dir, exist_ok=True)
    tasks = all_results["task"].unique()
    connectivity_measures = all_results["connectivity"].unique()
    for task in tasks:
        for conn_measure in connectivity_measures:
            for classify in ["Runs", "Subjects"]:
                df = all_results[
                    (all_results["classes"] == classify)
                    & (all_results["connectivity"] == conn_measure)
                    & (all_results["task"] == task)
                ]
                expected_labels = np.concatenate(
                    df["expected_labels"].to_numpy()
                )
                predicted_labels = np.concatenate(
                    df["predicted_labels"].to_numpy()
                )
                cm = confusion_matrix(
                    expected_labels, predicted_labels, normalize="true"
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
                ax.set_title(
                    f"Classifying {classify} in {task} using {conn_measure}"
                )
                plot_file = os.path.join(
                    confusion_dir,
                    f"{task}_{conn_measure}_{classify}_confusion.png",
                )
                plt.savefig(plot_file, bbox_inches="tight")
                plt.close()


if __name__ == "__main__":
    # cache and output directory
    cache = DATA_ROOT = "/storage/store/work/haggarwa/"
    results_dir = os.path.join(DATA_ROOT, "func_conn_class_results_50_glc")
    os.makedirs(results_dir, exist_ok=True)
    # we will use the resting state and all the movie-watching sessions
    tasks = [
        "RestingState",
        "Raiders",
        "GoodBadUgly",
        "MonkeyKingdom",
    ]
    connectivity_measures = [
        "correlation",
        "partial correlation",
        "tangent",
    ]
    classify = ["Runs", "Subjects"]
    # get the atlas
    atlas = datasets.fetch_atlas_schaefer_2018(
        data_dir=cache, resolution_mm=2, n_rois=400
    )
    all_results = Parallel(n_jobs=4, verbose=1, backend="multiprocessing")(
        delayed(pipeline)(
            task,
            atlas,
            cache,
            results_dir,
            connectivity_measures,
            classify,
        )
        for task in tasks
    )
    all_results = pd.concat(all_results)
    # calculate chance level
    all_results = chance_level(all_results)
    # save the results
    all_results.to_csv(os.path.join(results_dir, "all_results.csv"))
    # plot accuracy as barplots
    plot_accuracy(all_results, results_dir)
    # plot confusion matrices
    plot_confusion(all_results, results_dir)
