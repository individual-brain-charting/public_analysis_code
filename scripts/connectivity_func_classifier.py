import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import LinearSVC
from ibc_public.utils_connectivity import get_ses_modality, get_all_subject_connectivity
import itertools
import os
import pandas as pd
from joblib import Parallel, delayed


def classify(task, connectivity_measures, scenarios, features, DATA_ROOT, cv_splits=5, cv_test_size=0.25, out_dir="func_classification"):
    print(task)
    ss = {}
    sub_ses, mod = get_ses_modality(task)
    for conn_measure in connectivity_measures:
        print(conn_measure)
        ss[conn_measure] = {}
        # get the data for all subjects and runs
        # connectivity is a list of lists of numpy arrays
        # where each numpy array is a connectivity matrix for a run
        # and each list of numpy arrays is for a subject
        connectivity = get_all_subject_connectivity(
            sub_ses, mod, conn_measure, DATA_ROOT
        )
        # transpose the data to have subjects as rows and runs as columns
        t_connectivity = list(
                map(list, itertools.zip_longest(*connectivity, fillvalue=None))
            )
        # remove the last 3 runs for Raiders and GoodBadUgly
        if task == "Raiders" or task == "GoodBadUgly":
            t_connectivity = t_connectivity[:-3]
            connectivity = list(
                map(
                    list,
                    itertools.zip_longest(*t_connectivity, fillvalue=None),
                )
            )
        connectivity = np.asarray(connectivity)
        t_connectivity = np.asarray(t_connectivity)
        # get the number of subjects, runs and rois
        n_subs = connectivity.shape[0]
        n_runs_per_sub = connectivity.shape[1]
        n_rois_per_run = connectivity.shape[2]
        for scenario in scenarios:
            print(scenario)
            for feature in features:
                print(feature)
                ss[conn_measure][f"{scenario}_{feature}"] = []
                # store the data, cross-validation, classes and scores
                if scenario == "subs":
                    if feature == "subsruns":
                        # print(NotImplementedError)
                        # continue
                        # reshape the data to have 
                        # dim 1 (n_samples) = n_subs * n_rois_per_run 
                        # and dim 2 (n_features) = n_runs_per_sub
                        data = connectivity.transpose(2,0,1).reshape(-1, n_runs_per_sub)
                        assert data.shape == (n_subs*n_rois_per_run, n_runs_per_sub)
                        classes = np.asarray(list(sub_ses.keys())).repeat(n_rois_per_run)
                        assert classes.shape == (n_subs*n_rois_per_run, )
                    elif feature == "rois":
                        # reshape the data to have
                        # dim 1 (n_samples) = n_subs * n_runs_per_sub
                        # and dim 2 (n_features) = n_rois_per_run
                        data = connectivity.reshape(n_subs*n_runs_per_sub, n_rois_per_run)
                        assert data.shape == (n_subs*n_runs_per_sub, n_rois_per_run)
                        classes = np.asarray(list(sub_ses.keys())).repeat(n_runs_per_sub)
                        assert classes.shape == (n_subs*n_runs_per_sub, )
                elif scenario == "runs":
                    # print(NotImplementedError)
                    # continue
                    if feature == "subsruns":
                        # reshape the data to have 
                        # dim 1 (n_samples) = n_runs_per_sub * n_rois_per_run 
                        # and dim 2 (n_features) = n_subs
                        data = t_connectivity.transpose(2,0,1).reshape(-1, n_subs)
                        assert data.shape == (n_runs_per_sub*n_rois_per_run, n_subs)
                        classes = np.asarray([f"run-{i:02d}" for i in range(1, n_runs_per_sub + 1)]).repeat(n_rois_per_run)
                        assert classes.shape == (n_runs_per_sub*n_rois_per_run, )
                    elif feature == "rois":
                        # reshape the data to have
                        # dim 1 (n_samples) = n_subs * n_runs_per_sub
                        # and dim 2 (n_features) = n_rois_per_run
                        data = t_connectivity.reshape(n_subs*n_runs_per_sub, n_rois_per_run)
                        assert data.shape == (n_runs_per_sub*n_subs, n_rois_per_run)
                        classes = np.asarray(
                                [f"run-{i:02d}" for i in range(1, n_runs_per_sub + 1)]).repeat(n_subs)
                        assert classes.shape == (n_subs*n_runs_per_sub, )
                    print(data.shape)
                # cross-validation
                cv = StratifiedShuffleSplit(
                        n_splits=cv_splits, random_state=0, test_size=cv_test_size
                    )
                for train, test in cv.split(data, classes):
                    # fit the classifier
                    classifier = LinearSVC().fit(data[train], classes[train])
                    # make predictions for the left-out test subjects
                    predictions = classifier.predict(data[test])
                    # store the accuracy for this cross-validation fold
                    ss[conn_measure][f"{scenario}_{feature}"].append(
                        accuracy_score(classes[test], predictions)
                    )
    # save the results
    df = pd.DataFrame(ss)
    df.reset_index(inplace=True)
    df.rename(columns={"index": "scenario"}, inplace=True)
    df = df.explode(connectivity_measures)
    df.reset_index(inplace=True, drop=True)
    out_dir = os.path.join(DATA_ROOT, out_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    df.to_csv(os.path.join(out_dir, f"class_accuracy_{task}.csv"))
    return df

if __name__ == "__main__":
    # cache and output directory
    cache = DATA_ROOT = "/storage/store/work/haggarwa/"
    connectivity_measures = ["Pearsons_corr", "Pearsons_partcorr", "TangentSpaceEmbedding"]
    # we will use the resting state and all the movie-watching sessions
    tasks = [
        "RestingState",
        "Raiders",
        "GoodBadUgly",
        "MonkeyKingdom",
    ]
    scenarios = ["runs", "subs"]
    features = ["rois", "subsruns"]
    results = Parallel(n_jobs=4, verbose=1, backend="multiprocessing")(
        delayed(classify)(task, connectivity_measures, scenarios, features, DATA_ROOT)
        for task in tasks
    )
    print(results)
