"""
This script generates figures that assess low-level data quality
for the second release of the IBC dataset:
* amount of motion
* tSNR of the data
Authors: Bertrand Thirion, Ana Luisa Pinho 2020
Compatibility: Python 3.5
"""
import os
import glob
import json
import warnings

from joblib import Parallel, delayed

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nilearn.input_data import NiftiMasker
from nilearn import plotting

from ibc_public.utils_data import (data_parser, DERIVATIVES)

# ############################### INPUTS ######################################

# Extract data from dataframe only referring to a pre-specified subset of
# participants
sub_no = [4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15]

# ### All tasks ###
# Tasks terminology must conform with the one used in all_contrasts.tsv and
# main_contrasts.tsv
task_list = ['MathLanguage', 'SpatialNavigation', 'EmoReco', 'EmoMem',
             'StopNogo', 'Catell', 'FingerTapping', 'VSTMC',
             'FaceBody', 'Scene'] 
sufix = 'rel3'

cache = '/storage/store3/work/aponcema/IBC_paperFigures/data_quality3/'\
        'cache_data_quality3'
mem = '/storage/store3/work/aponcema/IBC_paperFigures/data_quality3/'\
      'cache_data_quality3'

# #############################################################################

# Define subjects' paths
sub_path = [os.path.join(DERIVATIVES, 'sub-%02d' % s) for s in sub_no]
SUBJECTS = [os.path.basename(full_path) for full_path in sub_path]

# BIDS conversion of task names
# Load dictionary file
with open(os.path.join('bids_preprocessed.json'), 'r') as f:
    task_dic = json.load(f)

TASKS = [task_dic[tkey] for tkey in task_list]
TASKS = [item for sublist in TASKS for item in sublist]


def compute_tsnr_(img, masker):
    x = masker.transform(img)
    return(x.mean(0) / np.maximum(1.e-8, x.std(0)))

def compute_tsnr(imgs, masker, n_jobs=6):
    """Compute the SNR of the provided image"""
    import numpy as np
    tsnr = Parallel(n_jobs=n_jobs)(delayed(compute_tsnr_)(img, masker)
                              for img in imgs)
    return(np.array(tsnr))


def average_brain_mask(derivatives=DERIVATIVES):
    """Compute an average brain masks across all the brain masks available"""
    from nilearn.masking import intersect_masks
    masks = glob.glob(os.path.join(derivatives, 'sub-*/ses-00/mask.nii.gz'))
    return(intersect_masks(masks, .25))


def motion_histogram(db):
    """compute motion histograms from realignment files"""
    rps = list(db[db.contrast == 'motion'].path)
    n_bins = 100
    bins = np.linspace(-2, 2, n_bins + 1)
    H = np.zeros((6, n_bins))
    xlist = np.empty((6, 0))
    for rp in rps:
        X = np.loadtxt(rp).T
        X[3:] *= (180. / np.pi)
        H += np.array([np.histogram(x, bins)[0] for x in X])
        xlist = np.hstack((xlist, X))

    # Process values do get convidence intervals
    xlist.sort(1)
    left, right = int(.0005 * xlist.shape[1]), int(.9995 * xlist.shape[1])
    print('.999 confindence interval')
    print(xlist[:, left])
    print(xlist[:, right])
    left, right = int(.005 * xlist.shape[1]), int(.995 * xlist.shape[1])
    print('.99 confindence interval')
    print(xlist[:, left])
    print(xlist[:, right])

    # Plot the histograms
    H = (H.T / H.sum(1))
    mbins = .5 * (bins[1:] + bins[:-1])
    plt.figure(figsize=(6, 4))
    plt.plot(mbins, H, linewidth=1)
    plt.fill(mbins, H, alpha=.3)
    plt.legend(['translation x', 'translation y', 'translation z',
                'rotation x', 'rotation y', 'rotation z'], fontsize=10)
    plt.xlabel('mm/degrees')
    plt.ylabel('normalized histogram')
    plt.title('Histogram of motion parameters')
    # plot the support
    for i, color  in enumerate(['b', 'g', 'r', 'c', 'm', 'y']):
        plt.plot([xlist[i, left], xlist[i, right]],
                 [-0.001 - .003 * i, -.001 - .003 * i], linewidth=3,
                 color=color)
    # dummy line for
    plt.plot([xlist[i, left], xlist[i, right]], [-0.018, -.018], color='w')
    plt.axis('tight')
    plt.subplots_adjust(bottom=.12, left=.14)
    plt.savefig(os.path.join(cache, 'rp_%s.png' % sufix), dpi=600)
    # plt.show()

if __name__ == '__main__':
    db = data_parser(derivatives=DERIVATIVES, subject_list = SUBJECTS,
                     task_list=TASKS)
    mask = average_brain_mask()
    mask.to_filename('/tmp/mask.nii.gz')
    masker = NiftiMasker(mask_img=mask, memory=mem).fit()
    imgs = list(db[db.contrast == 'preprocessed'].path)
    tsnr = compute_tsnr(imgs, masker, n_jobs=1)
    tsnr_map = masker.inverse_transform(tsnr.mean(0))
    tsnr_map.to_filename(os.path.join(cache, 'average_tsnr_%s.nii.gz' % sufix))
    # Load pre-computed .nii.gz files
    # tsnr_map = os.path.join(cache, 'average_tsnr_%s.nii.gz' % sufix)
    tsnr_plot = plotting.plot_epi(tsnr_map, vmax=70, colorbar=True)
    tsnr_plot.savefig(os.path.join(cache, 'average_tsnr_%s.png' % sufix), dpi=600)
    motion_histogram(db)
    