"""
This algorithm 
* removes singleton fibers
* outputs a colormap for remaining fibers
"""
import os
import matplotlib.pyplot as plt
import numpy as np
from dipy.io.streamline import load_tck, save_tck, load_trk
from dipy.segment.clustering import QuickBundles

workdir = '/neurospin/ibc/derivatives/sub-04/ses-08/dwi'
#f = os.path.join(workdir,
#                 'reduced-tracks-100k_sub-04_ses-08.tck')
f = os.path.join(workdir,
                 'tracks_sub-04_ses-08_t1.tck')

ref = os.path.join(workdir,
                   'sub-04_ses-08_desc-denoise-eddy-correct_dwi.nii.gz')
tract = load_tck(f, ref)

"""
qb = QuickBundles(threshold=20.)

# want to get symmetric clusters
streamlines = tract.streamlines.copy()
for streamline in streamlines:
    streamline[:, 0] = np.abs(streamline[:, 0])

clusters = qb.cluster(streamlines)
sizes = clusters.clusters_sizes()
print(sizes)
size_threshold = 100
labels = np.zeros(len(tract.streamlines), dtype=int)

q = 1
for i, cluster in enumerate(clusters):
    if sizes[i] > size_threshold:
        labels[cluster.indices] = q
        q += 1

# proportion = .1
# labels *= (np.random.rand(len(labels)) < proportion)
tract.streamlines = tract.streamlines[labels > 0]
labels = labels[labels > 0]
labels -= 1
unique_labels = np.unique(labels)
np.random.seed(1)
np.random.shuffle(unique_labels)
labels = unique_labels[labels]
n_valid_labels = len(unique_labels)
np.savetxt(os.path.join(workdir, 'palette.txt'), labels)

print(save_tck(
    tract,
    os.path.join(workdir, 'cleaned-tracks-100k_sub-04_ses-08.tck'),
    bbox_valid_check=True))
"""


from dipy.align.streamlinear import whole_brain_slr
from dipy.segment.bundles import RecoBundles
from dipy.data import fetch_bundle_atlas_hcp842, get_bundle_atlas_hcp842
from dipy.io.utils import create_tractogram_header

atlas_file, atlas_folder = fetch_bundle_atlas_hcp842()
atlas_file, all_bundles_files = get_bundle_atlas_hcp842()
sft_atlas = load_trk(atlas_file, "same", bbox_valid_check=False)
atlas = sft_atlas.streamlines
atlas_header = create_tractogram_header(atlas_file,
                                        *sft_atlas.space_attributes)

moved, transform, qb_centroids1, qb_centroids2 = whole_brain_slr(
    atlas, load_tck(f, ref).streamlines,
    x0='affine', verbose=True, progressive=True,
    rng=np.random.RandomState(1984))

rb = RecoBundles(moved, verbose=True, rng=np.random.RandomState(2001))

import glob
bundle_files = sorted(glob.glob(all_bundles_files))

clusters = []
for bf in bundle_files:
    model = load_trk(bf, "same", bbox_valid_check=False).streamlines
    recognized, label = rb.recognize(model_bundle=model,
                                     model_clust_thr=0.05,
                                     reduction_thr=10,
                                     pruning_thr=5,
                                     reduction_distance='mdf',
                                     pruning_distance='mdf',
                                     slr=True)
    clusters.append(label)

n_fibers= len(moved)
labels = np.zeros(n_fibers, dtype=int)
for i, cluster in enumerate(clusters):
    labels[cluster] =  i + 1

tract.streamlines = tract.streamlines[labels > 0]
labels_ = labels[labels > 0]
labels_ -= 1

unique_bundles = np.arange(len(bundle_files))
np.random.seed(1)
np.random.shuffle(unique_bundles)
labels_ = unique_bundles[labels_]
np.savetxt(os.path.join(workdir, 'palette.txt'), labels_)

print(save_tck(
    tract,
    os.path.join(workdir, 'bundle-tracks-all_sub-04_ses-08.tck'),
    bbox_valid_check=True))
