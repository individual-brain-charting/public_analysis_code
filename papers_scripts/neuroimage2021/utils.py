"""
Utilities for script_ibc.py, for generalization to upcoming script_hcp.py

Author: Bertrand Thirion, 2020
"""
from nilearn.decomposition import DictLearning
import numpy as np
from joblib import Parallel, delayed

def make_dictionary(rs_fmri, n_components, cache, mask, n_jobs=1):
    dict_learning = DictLearning(n_components=n_components,
                                 memory=cache, memory_level=2,
                                 verbose=1, random_state=0, n_epochs=1,
                                 mask=mask, n_jobs=n_jobs)
    dict_learning.fit(rs_fmri)
    return dict_learning.components_img_, dict_learning.components_


def adapt_components(Y, subject, rs_fmri_db, masker, n_dim):
    rs_scans = rs_fmri_db[rs_fmri_db.subject == subject].path
    X_ = np.zeros_like(Y)
    for scan in rs_scans.values:
        X = masker.transform(scan)
        U, S, V = np.linalg.svd(X, 0)
        Vk = V[:n_dim]
        X_ += Y.dot(Vk.T).dot(Vk)
    return X_


def make_parcellation(ward, rs_fmri):
    indexes = np.random.randint(len(rs_fmri), size=5)
    ward.fit([rs_fmri[j] for j in indexes])
    return ward.labels_img_


def make_parcellations(ward, rs_fmri, n_parcellations, n_jobs):
    parcellations = Parallel(n_jobs=n_jobs)(delayed(make_parcellation)(
        ward, rs_fmri) for b in range(n_parcellations))
    return parcellations


def predict_Y_oneparcel(parcellations, dummy_masker, train_index,
                          n_parcels, X, models, b):
    labels = np.ravel(dummy_masker.transform(parcellations[b]))
    Y_pred = np.zeros((models[0][0].shape[0], labels.size))
    for q in range(n_parcels):
        parcel = labels == q + 1
        for i in train_index:
            Y_pred[:, parcel] += np.dot(
                X.T[parcel], models[i][b * n_parcels + q].T).T
    return Y_pred


def predict_Y_multiparcel(parcellations, dummy_masker, train_index,
                          n_parcels, Y, X, models, n_jobs):
    n_parcellations = len(parcellations)
    #Y_preds = Parallel(n_jobs=n_jobs)(delayed(predict_Y_oneparcel)(
    #    parcellations, dummy_masker, train_index,
    #    n_parcels, X, models, b) for b in range(n_parcellations))
    Y_preds = []
    for b in range(n_parcellations):
        Y_pred_ = predict_Y_oneparcel(
            parcellations, dummy_masker, train_index,
            n_parcels, X, models, b)
        Y_preds.append(Y_pred_)
    #
    Y_preds = np.array(Y_preds)
    Y_pred = np.sum(Y_preds, 0) / (n_parcellations * len(train_index))
    return Y_pred


def permuted_score(Y, Y_pred, Y_baseline, n_permutations, seed=1):
    rng = np.random.RandomState(seed)
    n_contrasts = Y.shape[1]
    permuted_con_score = []
    permuted_vox_score = []
    for b in range(n_permutations):
        permutation = rng.permutation(n_contrasts)
        Y_ = Y[:, permutation]
        vox_score = 1 - np.sum((Y_ - Y_pred) ** 2, 0) / np.sum((
            Y_ - Y_baseline.mean(0)) ** 2, 0)
        con_score = 1 - np.sum((Y_.T - Y_pred.T) ** 2, 0) / np.sum(
            (Y_.T - Y_baseline.T.mean(0)) ** 2, 0)
        permuted_con_score.append(con_score)
        #permuted_vox_score.append(vox_score)
    return permuted_con_score#, permuted_vox_score


def fit_regressions(individual_components, data, parcellations,
                    dummy_masker, clf, i):
    n_parcellations = len(parcellations)
    X = individual_components[i]
    Y = data[i]
    model = []
    n_parcellations = len(parcellations)
    for b in range(n_parcellations):
        labels = np.ravel(
            dummy_masker.transform(parcellations[b]).astype(np.int))
        n_parcels = len(np.unique(labels))
        for q in range(n_parcels):
            parcel = labels == q + 1
            model_ = clf.fit(X.T[parcel], Y.T[parcel]).coef_
            model.append(model_)
    return model


def predict_Y(parcellations, dummy_masker, n_parcels, X, average_models):
    n_parcellations = len(parcellations)
    Y_preds = []
    for b in range(n_parcellations):
        labels = np.ravel(dummy_masker.transform(parcellations[b]))
        Y_pred = np.zeros((average_models[0].shape[0], labels.size))
        for q in range(n_parcels):
            parcel = labels == q + 1
            Y_pred[:, parcel] = np.dot(
                X.T[parcel], average_models[b * n_parcels + q].T).T
        Y_preds.append(Y_pred)
    Y_preds = np.array(Y_preds)
    Y_pred = np.sum(Y_preds, 0) / (n_parcellations)
    return Y_pred
