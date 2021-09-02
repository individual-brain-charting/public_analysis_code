"""
Utilities for dictionary learning

Authors: Bertrand Thirion, Ana Luisa Pinho

Last update: June 2020

Compatibility: Python 3.5

"""

import os
import numpy as np
from sklearn.manifold import spectral_embedding
import matplotlib.pyplot as plt
from ibc_public.utils_data import CONTRASTS, all_contrasts
from joblib import Memory


def initial_dictionary(n_clusters, X,):
    """Creat initial dictionary"""
    from sklearn.cluster import MiniBatchKMeans
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=0,
                             batch_size=200, n_init=10)
    kmeans = kmeans.fit(X.T)
    dictionary_ = kmeans.cluster_centers_
    dictionary = (dictionary_.T / np.sqrt((dictionary_ ** 2).sum(1))).T
    similarity = np.exp(np.corrcoef(dictionary))
    embedding = spectral_embedding(similarity, n_components=1)
    order = np.argsort(embedding.T).ravel()
    dictionary = dictionary[order]
    return dictionary


def make_dictionary(X, n_components=20, alpha=5., write_dir='/tmp/',
                    contrasts=[], method='multitask', l1_ratio=.5,
                    n_subjects=13):
    """Create dictionary + encoding"""
    from sklearn.decomposition import dict_learning_online, sparse_encode
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import MultiTaskLasso, MultiTaskElasticNet

    mem = Memory(write_dir, verbose=0)
    dictionary = mem.cache(initial_dictionary)(n_components, X)
    np.savez(os.path.join(write_dir, 'dictionary.npz'),
             loadings=dictionary, contrasts=contrasts)
    if method == 'online':
        components, dictionary = dict_learning_online(
                X.T, n_components, alpha=alpha,
                dict_init=dictionary,
                batch_size=200,
                method='cd',
                return_code=True,
                shuffle=True,
                n_jobs=1,
                positive_code=True)
        np.savez(os.path.join(write_dir, 'dictionary.npz'),
                 loadings=dictionary, contrasts=contrasts)
    elif method == 'sparse':
        components = sparse_encode(
            X.T, dictionary, alpha=alpha, max_iter=10, n_jobs=1,
            check_input=True, verbose=0, positive=True)
    elif method == 'multitask':
        # too many hard-typed parameters !!!
        n_voxels = X.shape[1] // n_subjects
        components = np.zeros((X.shape[1], n_components))
        clf = MultiTaskLasso(alpha=alpha)
        clf = MultiTaskElasticNet(alpha=alpha, l1_ratio=l1_ratio)
        for i in range(n_voxels):
            x = X[:, i: i + n_subjects * n_voxels: n_voxels]
            components[i: i + n_subjects * n_voxels: n_voxels] =\
                clf.fit(dictionary.T, x).coef_
    return dictionary, components


def cluster(Xr, n_components=20, write_dir='/tmp/', contrasts=[]):
    """Kmeans clustering"""
    from sklearn.cluster import MiniBatchKMeans
    kmeans = MiniBatchKMeans(n_clusters=n_components, random_state=0,
                             batch_size=200, n_init=10)
    kmeans = kmeans.fit(Xr.T)
    dictionary = kmeans.cluster_centers_
    labels = kmeans.labels_
    n_samples = labels.size
    components = np.zeros((n_samples, n_components))
    components[np.arange(n_samples), labels] = 1
    similarity = np.exp(np.corrcoef(dictionary))
    embedding = spectral_embedding(similarity, n_components=1)
    order = np.argsort(embedding.T).ravel()
    dictionary = dictionary[order]
    components = components[:, order]
    np.savez(os.path.join(write_dir, 'dictionary.npz'),
             loadings=dictionary, contrasts=contrasts)
    return dictionary, components


def dictionary2labels(dictionary, task_list, path, facecolor=[.5, .5, .5],
                      contrasts=[], best_labels=[]):
    """Create a figure with labels reflecting the input dictionary"""
    from matplotlib.cm import gist_ncar
    LABELS = _make_labels(all_contrasts, task_list)
    w = dictionary
    plt.figure(facecolor=facecolor, figsize=(2.3, 6))
    cmap = plt.get_cmap('gist_ncar')
    colors = cmap([133, 15, 168, 143, 173, 235, 195, 70, 205, 0, 107, 178,
                   123, 215, 189, 27, 163, 35, 48, 55])
    # colors = gist_ncar(np.linspace(0, 1, len(dictionary)))

    for k, comp in enumerate(dictionary):
        if len(best_labels) >= len(dictionary):
            best_label = best_labels[k]
        else:
            weights = comp
            labels = [LABELS[contrast][x > 0] for (x, contrast) in
                      zip(comp, contrasts)]
            order = np.argsort(-weights)
            spec = weights == w.max(0)
            if spec.any():
                best_label = np.array(labels)[order][spec[order]][:2]
            else:
                best_label = labels[order[0]]
            print(best_label)
            best_label = np.unique(best_label)
            best_label = str(best_label).replace('[', '').replace(']', '')
            best_label = best_label.replace("' '", ", ")
            best_label = best_label.replace("'", "")
            best_labels.append(best_label)
        plt.text(0, .05 * k, best_label, weight='bold', color=colors[k],
                 fontsize=14)
    plt.axis('off')
    plt.subplots_adjust(left=.01, bottom=.01, top=.99, right=.99)
    plt.savefig(path, facecolor=facecolor, dpi=300)
    plt.show(block=False)
    return best_labels


def _make_labels(contrasts, task_list):
    labels = {}
    for i in range(len(CONTRASTS)):
        if CONTRASTS.task[i] in task_list:
            labels[CONTRASTS.contrast[i]] = [contrasts['negative label'][i],
                                             contrasts['positive label'][i]]
    return labels
