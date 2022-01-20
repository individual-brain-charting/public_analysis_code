"""
Small script to compare the merits of RFX and conjunctio and explain
why we used the latter. Try to avoid "I'm getting More with this
approach" type of argument.

Author: Bertrand Thirion, Ana Luisa Pinho - 2020
export PYTHONPATH=$PYTHONPATH:../high_level_analysis_scripts/

Compatibility: Python 3.5

"""

import os
import pandas as pd
import numpy as np
from joblib import Memory, Parallel, delayed
from nilearn.input_data import NiftiMasker
from nilearn import plotting
import matplotlib.pyplot as plt
import ibc_public
from ibc_public.utils_data import (
    data_parser, SMOOTH_DERIVATIVES, DERIVATIVES, SUBJECTS, CONTRASTS)
from nistats.thresholding import map_threshold
from nistats.second_level_model import SecondLevelModel
from sklearn.metrics import jaccard_similarity_score


def one_sample_test(imgs, glm, threshold=3.1, height_control='none'):
    """ Do a one sample t-test of the contrast images in dataframe

    Parameters
    ----------
    imgs: list of strings,
          input images
    glm: nistats.SecondLevelModel instance
         the model used for analysis (with preset mask)

    verbose: bool, optional,
             verbosity mode

    Returns
    -------
    z: array of shape (mask.sum()),
       the z-transformed contrast values within the mask
    """
    n_samples = len(imgs)
    dmtx = pd.DataFrame(np.ones(n_samples), columns=['intercept'])
    glm.fit(imgs, design_matrix=dmtx)
    rfx_img = glm.compute_contrast([1], output_type='z_score')
    _, threshold_ = map_threshold(
        rfx_img, threshold=threshold, height_control=height_control)
    rfx = glm.masker_.transform(rfx_img)
    rfx = rfx * (np.abs(rfx) > threshold)
    return rfx_img, threshold_, rfx


def conjunction_img(imgs, masker, contrast, percentile=50, threshold=3.1,
                    height_control='none'):
    """ generate conjunction statsitics of the contrat images in dataframe

    Parameters
    ----------
    df: pandas dataframe,
        holding information on the database indexed by task, contrast, subject

    contrasts: list of strings,
               The contrasts to be tested
    masker: niftimasker instance,
            to define the spatial context of the analysis
    precentile:  float,
        Percentile used for the conjunction analysis
    """
    from conjunction import _conjunction_inference_from_z_values
    contrast_mask = (df.contrast.values == contrast)
    Z = masker.transform(imgs).T
    pos_conj = _conjunction_inference_from_z_values(Z, percentile * .01)
    neg_conj = _conjunction_inference_from_z_values(-Z, percentile * .01)
    conj = pos_conj
    conj[conj < 0] = 0
    conj[neg_conj > 0] = - neg_conj[neg_conj > 0]
    conj_img = masker.inverse_transform(conj)
    _, threshold_ = map_threshold(
        conj_img, threshold=threshold, height_control=height_control)
    conj = conj * (np.abs(conj) > threshold)
    return conj_img, threshold_, conj

# caching
main_parent_dir = '/neurospin/tmp/'
alt_parent_dir = '/storage/tompouce/'
main_dir = 'bthirion'
alt_dir = 'agrilopi'

if os.path.exists(main_parent_dir):
    cache = main_parent_dir + alt_dir
else:
    cache = alt_parent_dir + main_dir

# output directory
mem = Memory(cachedir=cache, verbose=0)
write_dir = cache
if not os.path.exists(write_dir):
    os.mkdir(write_dir)

# Access to the data
subject_list = SUBJECTS
task_list = ['ArchiEmotional', 'ArchiSocial', 'ArchiSpatial',
             'ArchiStandard', 'HcpEmotion', 'HcpGambling',
             'HcpLanguage', 'HcpMotor', 'HcpRelational',
             'HcpSocial', 'HcpWm',
             'RSVPLanguage']

# Mask of the grey matter across subjects
_package_directory = os.path.dirname(
    os.path.abspath(ibc_public.utils_data.__file__))
mask_gm = os.path.join(
    _package_directory, '../ibc_data', 'gm_mask_1_5mm.nii.gz')

masker = NiftiMasker(mask_img=mask_gm).fit()
glm = SecondLevelModel(mask=mask_gm)
qval = .05
height_control = 'fdr'
n_bootstrap = 100


def analyse_contrast(df, masker, glm, contrast, n_bootstrap, qval,
                     height_control):
    print(contrast)
    contrast_mask = (df.contrast.values == contrast) &\
                    (df.acquisition == 'ffx')
    imgs = df.path[contrast_mask].values.tolist()
    img1, threshold, ref1 = one_sample_test(imgs, glm, threshold=qval,
                                            height_control=height_control)
    plotting.plot_stat_map(img1, threshold=threshold, title='RFX',
                           vmax=10)
    img2, threshold, ref2 = conjunction_img(
        imgs, masker, contrast, percentile=25,  threshold=qval,
        height_control=height_control)
    plotting.plot_stat_map(img2, threshold=threshold, title='conj 25%',
                           vmax=10)
    img3, threshold, ref3 = conjunction_img(
        imgs, masker, contrast, percentile=50, threshold=qval,
        height_control=height_control)
    plotting.plot_stat_map(img3, threshold=threshold, title='conj 50%',
                           vmax=10)

    score1, score2, score3 = [], [], []
    for b in range(n_bootstrap):
        bootstrap = np.random.randint(0, len(imgs), len(imgs))
        imgs_ = [imgs[b] for b in bootstrap]
        _, _, sample1 = one_sample_test(imgs_, glm, threshold=qval,
                                        height_control=height_control)
        score1.append(.5 * (
            jaccard_similarity_score(ref1 > 0, sample1 > 0) +
            jaccard_similarity_score(ref1 < 0, sample1 < 0)))

        _, _, sample2 = conjunction_img(
            imgs_, masker, contrast, percentile=25,  threshold=qval,
            height_control=height_control)
        score2.append(.5 * (jaccard_similarity_score(ref2 > 0, sample2 > 0) +
                            jaccard_similarity_score(ref2 < 0, sample2 < 0)))

        _, _, sample3 = conjunction_img(
            imgs_, masker, contrast, percentile=50,  threshold=qval,
            height_control=height_control)
        score3.append(.5 * (jaccard_similarity_score(ref3 > 0, sample3 > 0) +
                            jaccard_similarity_score(ref3 < 0, sample3 < 0)))

    return score1, score2, score3

# ******************************************************************************
# Use this snippet of code to compute the bootsprap set and, then,
# jaccard index

# Smooth derivatives
df = data_parser(derivatives=SMOOTH_DERIVATIVES, subject_list=SUBJECTS,
                 conditions=CONTRASTS, task_list=task_list)
df = df[df.modality == 'bold']

scores_ = Parallel(n_jobs=6)(delayed(analyse_contrast)(
    df, masker, glm, contrast, n_bootstrap, qval, height_control)
                            for contrast in df.contrast.unique())
scores = np.rollaxis(np.array(scores_), 1)
smooth_rfx, smooth_c25, smooth_c50 = scores

# Non-smooth derivatives
df = data_parser(derivatives=DERIVATIVES, subject_list=SUBJECTS,
                 conditions=CONTRASTS, task_list=task_list)
df = df[df.modality == 'bold']

scores_ = Parallel(n_jobs=6)(delayed(analyse_contrast)(
    df, masker, glm, contrast, n_bootstrap, qval, height_control)
                            for contrast in df.contrast.unique())
scores = np.rollaxis(np.array(scores_), 1)
rfx, c25, c50 = scores
plt.close('all')


np.savez(os.path.join(cache, 'conj_vs_rfx_smooth.npz'),
         smooth_rfx=smooth_rfx, smooth_c25=smooth_c25, smooth_c50=smooth_c50,
         rfx=rfx, c25=c25, c50=c50)
# *****************************************************************************

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# # Uncomment this snippet of code only for plotting
npzfile_smooth = np.load(os.path.join(cache, 'conj_vs_rfx_smooth.npz'))
smooth_rfx = npzfile_smooth['smooth_rfx']
smooth_c25 = npzfile_smooth['smooth_c25']
rfx = npzfile_smooth['rfx']
c25 = npzfile_smooth['c25']
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# # Plot Fig. 1
# plt.figure()
# plt.plot(np.concatenate(smooth_rfx), 'b', label='random effects, smooth')
# plt.plot(np.concatenate(smooth_c25), 'r', label='conjunction 25%, smooth')
# plt.plot(np.concatenate(rfx), 'g', label='random effects')
# plt.plot(np.concatenate(c25), 'k', label='conjunction 25%')
# plt.legend()

# Plot Fig. 2 - boxplots
fig, axes = plt.subplots(figsize=(7, 4))
bplot = axes.boxplot(np.vstack((np.concatenate(smooth_rfx),
                                np.concatenate(rfx),
                                np.concatenate(smooth_c25),
                                np.concatenate(c25))).T,
                     patch_artist=True, sym='+')

# Median line
for median in bplot['medians']:
    median.set(linewidth=1.5)

# Fill with colors
colors = ['royalblue', 'royalblue', 'teal', 'teal']
for i, (patch, color) in enumerate(zip(bplot['boxes'], colors)):
    patch.set_facecolor(color)
    if i % 2 == 0:
        patch.set_alpha(.5)

plt.xticks(range(1, 5), ['RFX\n smoothed',
                         'RFX\n unsmoothed',
                         'Conjunction 25%\n smoothed',
                         'Conjunction 25%\n unsmoothed'],
           ha='center')
plt.title('Distribution of map consistency for all contrasts ' +
          '\n between original set and bootstrap resampling',
          fontweight='bold')
plt.ylabel('Jaccard index')
plt.subplots_adjust(bottom=.125)
plt.savefig(os.path.join(cache, 'conj_vs_rfx.eps'), format='eps')
plt.show(block=False)
