""" Various utilities for surface-based plotting of brain maps
"""
import numpy as np
import nibabel as nib
import os
from nilearn import plotting


def surface_one_sample(df, contrast, side):
    from scipy.stats import ttest_1samp, norm
    mask = (df.contrast.values == contrast) * (df.side.values == side)
    X = np.array([nib.load(texture).darrays[0].data
                  for texture in list(df.path[mask].values)])
    # print (X.shape, np.sum(np.isnan(X)))
    t_values, p_values = ttest_1samp(X, 0)
    p_values = .5 * (1 - (1 - p_values) * np.sign(t_values))
    z_values = norm.isf(p_values)
    return z_values


def surface_conjunction(df, contrast, side, percentile=25):
    from conjunction import _conjunction_inference_from_z_values
    mask = (df.contrast.values == contrast) * (df.side.values == side)
    Z = np.array([nib.load(texture).darrays[0].data
                  for texture in list(df.path[mask].values)]).T
    pos_conj = _conjunction_inference_from_z_values(Z, percentile * .01)
    neg_conj = _conjunction_inference_from_z_values(-Z, percentile * .01)
    conj = pos_conj
    conj[conj < 0] = 0
    conj[neg_conj > 0] = - neg_conj[neg_conj > 0]
    return conj


def make_thumbnail_surface(func, hemi, threshold=3.0, vmax=10.,
                           output_dir='/tmp'):
    if os.path.exists('/neurospin/ibc'):
        dir_ = '/neurospin/ibc/derivatives/sub-01/ses-00/anat/fsaverage/surf'
    else:
        dir_ = '/storage/store/data/ibc/derivatives/sub-01/ses-00/anat/' + \
               'fsaverage/surf'
    if hemi == 'right':
        mesh = os.path.join(dir_, 'rh.inflated')
        bg_map = os.path.join(dir_, 'rh.sulc')
    else:
        mesh = os.path.join(dir_, 'lh.inflated')
        bg_map = os.path.join(dir_, 'lh.sulc')

    medial = '/tmp/surf_medial_%s.png' % hemi
    lateral = '/tmp/surf_lateral_%s.png' % hemi
    # threshold = fdr_threshold(func, .05)
    plotting.plot_surf_stat_map(mesh, func, hemi=hemi, vmax=vmax,
                                threshold=threshold, bg_map=bg_map,
                                view='lateral', output_file=lateral)
    plotting.plot_surf_stat_map(mesh, func, hemi=hemi, vmax=vmax,
                                threshold=threshold, bg_map=bg_map,
                                view='medial', output_file=medial)
    return medial, lateral


def make_atlas_surface(label, hemi, name='', output_dir='/tmp'):
    if os.path.exists('/neurospin/ibc'):
        dir_ = '/neurospin/ibc/derivatives/sub-01/ses-00/anat/fsaverage/surf'
    else:
        dir_ = '/storage/store/data/ibc/derivatives/sub-01/ses-00/anat/' + \
               'fsaverage/surf'
    if hemi == 'right':
        mesh = os.path.join(dir_, 'rh.inflated')
        bg_map = os.path.join(dir_, 'rh.sulc')
    else:
        mesh = os.path.join(dir_, 'lh.inflated')
        bg_map = os.path.join(dir_, 'lh.sulc')

    medial = os.path.join(output_dir, '%s_medial_%s.png' % (name, hemi))
    lateral = os.path.join(output_dir, '%s_lateral_%s.png' % (name, hemi))
    plotting.plot_surf_roi(mesh, label, hemi=hemi, bg_map=bg_map,
                           view='lateral', output_file=lateral, alpha=.9)
    plotting.plot_surf_roi(mesh, label, hemi=hemi, bg_map=bg_map,
                           view='medial', output_file=medial, alpha=.9)


def faces_2_connectivity(faces):
    from scipy.sparse import coo_matrix
    n_features = len(np.unique(faces))
    edges = np.vstack((faces.T[:2].T, faces.T[1:].T, faces.T[0:3:2].T))
    weight = np.ones(edges.shape[0])
    connectivity = coo_matrix((weight, (edges.T[0], edges.T[1])),
                              (n_features, n_features))  # .tocsr()
    # Making it symmetrical
    connectivity = (connectivity + connectivity.T) / 2
    return connectivity


def connected_components_cleaning(connectivity, _map, cluster_size=10):
    from scipy.sparse import csgraph, coo_matrix
    n_features = connectivity.shape[0]
    weight = connectivity.data.copy()
    edges = connectivity.nonzero()
    i_idx, j_idx = edges
    weight[_map[i_idx] == 0] = 0
    weight[_map[j_idx] == 0] = 0
    mask = weight != 0
    reduced_connectivity = coo_matrix(
        (weight[mask], (i_idx[mask], j_idx[mask])), (n_features, n_features))
    # Clustering step: getting the connected components of the nn matrix
    n_components, labels = csgraph.connected_components(reduced_connectivity)
    label, count = np.unique(labels, return_counts=True)
    good_labels = label[count >= cluster_size]
    map_ = np.zeros_like(_map)
    for gl in good_labels:
        map_[labels == gl] = _map[labels == gl]
    return map_


def clean_surface_map(maps, hemi, cluster_size):
    """Clean surface maps by removing small connected components"""
    from nilearn.surface import load_surf_mesh
    if os.path.exists('/neurospin/ibc'):
        dir_ = '/neurospin/ibc/derivatives/sub-01/ses-00/anat/fsaverage/surf'
    else:
        dir_ = '/storage/store/data/ibc/derivatives/sub-01/ses-00/anat/' + \
               'fsaverage/surf'
    if hemi == 'right':
        mesh = os.path.join(dir_, 'rh.inflated')
    else:
        mesh = os.path.join(dir_, 'lh.inflated')

    _, faces = load_surf_mesh(mesh)
    connectivity = faces_2_connectivity(faces)
    for i in range(maps.shape[1]):
        maps[:, i] = connected_components_cleaning(
            connectivity, maps[:, i], cluster_size=cluster_size)
    return maps
