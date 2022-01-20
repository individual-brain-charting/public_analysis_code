"""
Script for RS-fMRI to task-fMRI analysis.

Works on IBC 3mm data 
"""

import os
import glob
from nilearn.input_data import NiftiMasker
import pandas as pd
from joblib import Memory, Parallel, delayed
from ibc_public.utils_data import (
    data_parser, SUBJECTS, LABELS, CONTRASTS, CONDITIONS, data_parser)
import ibc_public
import numpy as np
from utils import (
    make_dictionary, adapt_components, make_parcellations,
    predict_Y_multiparcel, permuted_score, fit_regressions)

DERIVATIVES = '/neurospin/ibc/3mm'
# cache
cache = '/neurospin/tmp/bthirion/rsfmri2tfmri'
write_dir = cache
if not os.path.exists(write_dir):
    os.mkdir(write_dir)
memory = Memory(cache, verbose=0)
n_jobs = 5

###############################################################################
# Get and fetch data: RS-fMRI and T-fMRI

# resting-state fMRI
wc = os.path.join(
    DERIVATIVES, 'sub-*', 'ses-*', 'func', 'wrdc*RestingState*.nii.gz')
rs_fmri = sorted(glob.glob(wc))

subjects = []
for img in rs_fmri:
    subject = img.split('/')[-4]
    subjects.append(subject)

rs_fmri_db = pd.DataFrame({'subject': subjects,
                           'path': rs_fmri})

# task fmri
mem = Memory(cachedir=cache, verbose=0)
subjects = rs_fmri_db.subject.unique()
n_subjects = len(subjects)

# Access to the data
task_list = ['ArchiEmotional', 'ArchiSocial', 'ArchiSpatial',
             'ArchiStandard', 'HcpEmotion', 'HcpGambling',
             'HcpLanguage', 'HcpMotor', 'HcpRelational',
             'HcpSocial', 'HcpWm',
             'RSVPLanguage']
db = data_parser(derivatives=DERIVATIVES, subject_list=subjects,
                 conditions=CONTRASTS, task_list=task_list)
df = db[db.task.isin(task_list)]
df = df.sort_values(by=['subject', 'task', 'contrast'])
contrasts = df.contrast.unique()

# mask of grey matter
_package_directory = os.path.dirname(
    os.path.abspath(ibc_public.utils_data.__file__))
mask_gm = os.path.join(
    _package_directory, '../ibc_data', 'gm_mask_3mm.nii.gz')

###############################################################################
# Dictionary learning of RS-fMRI

from nilearn.plotting import plot_prob_atlas, plot_stat_map
import matplotlib.pyplot as plt

n_components = 100


make_dictionary = mem.cache(make_dictionary)

dictlearning_components_img, Y = make_dictionary(
    rs_fmri, n_components, cache, mask_gm)
"""
from nilearn.decomposition import CanICA

canica = CanICA(n_components=n_components,
                memory=cache, memory_level=2,
                verbose=10,
                mask=mask_gm,
                random_state=0)
canica.fit(rs_fmri)
Y = canica.components_
dictlearning_components_img = canica.components_img_
"""
dictlearning_components_img.to_filename(
    os.path.join(write_dir, 'components.nii.gz'))

# visualize the results
plot_prob_atlas(dictlearning_components_img,
                title='All DictLearning components')






###############################################################################
# Dual regression to get individual components
masker = NiftiMasker(mask_img=mask_gm, smoothing_fwhm=4, memory=cache).fit()
dummy_masker = NiftiMasker(mask_img=mask_gm, memory=cache).fit()
n_dim = 200

individual_components = Parallel(n_jobs=n_jobs)(delayed(adapt_components)(
    Y, subject, rs_fmri_db, masker, n_dim) for subject in subjects)

# visualize the results
for i, subject in enumerate(subjects):
    individual_components_img = masker.inverse_transform(
        individual_components[i])
    #plot_prob_atlas(individual_components_img,
    #                title='DictLearning components, subject %s' % subject)

###############################################################################
# Generate brain parcellations
from nilearn.regions import Parcellations
n_parcellations = 20
n_parcels = 256

ward = Parcellations(method='ward', n_parcels=n_parcels,
                     standardize=False, smoothing_fwhm=4.,
                     memory='nilearn_cache', memory_level=1,
                     verbose=1, mask=mask_gm)

make_parcellations = memory.cache(make_parcellations)

parcellations = make_parcellations(ward, rs_fmri, n_parcellations, n_jobs)

###############################################################################
# Cross-validated predictions
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
clf = RidgeCV()

data =  np.array([masker.transform([df[df.subject == subject][
    df.contrast == contrast].path.values[-1] for contrast in contrasts])
         for subject in subjects])

    
# training the model
models = Parallel(n_jobs=n_jobs)(delayed(fit_regressions)(
    individual_components, data, parcellations,
    dummy_masker, clf, i) for i, subject in enumerate(subjects))


n_splits = 5
n_subject = len(subjects)
cv = KFold(n_splits=n_splits)

scores = []
vox_scores = []
con_scores = []
permuted_con_scores = []
permuted_vox_scores = []
n_permutations = 1000

for train_index, test_index in cv.split(range(n_subjects)):
    # construct the predicted maps
    for j in test_index:
        X = individual_components[j]
        Y = data[j]
        Y_baseline = np.mean(Y[train_index], 0)
        Y_pred = predict_Y_multiparcel(
            parcellations, dummy_masker, train_index,
            n_parcels, Y, X, models, n_jobs)
        score = 1 - (Y - Y_pred) ** 2 / Y ** 2
        vox_score_ = r2_score(Y, Y_pred, multioutput='raw_values')
        #vox_score_ = 1 - np.sum((Y - Y_pred) ** 2, 0) / np.sum((
        #    Y - Y.mean(0)) ** 2, 0)
        vox_score = 1 - np.sum((Y - Y_pred) ** 2, 0) / np.sum((
            Y - Y_baseline.mean(0)) ** 2, 0)
        con_score_ = r2_score(Y.T, Y_pred.T, multioutput='raw_values')
        #con_score = 1 - np.sum((Y.T - Y_pred.T) ** 2, 0) / np.sum(
        #    (Y.T - Y.T.mean(0)) ** 2, 0)
        con_score = 1 - np.sum((Y.T - Y_pred.T) ** 2, 0) / np.sum(
            (Y.T - Y_baseline.T.mean(0)) ** 2, 0)

        scores.append(score)
        vox_scores.append(vox_score)
        con_scores.append(con_score)
        if n_permutations > 0:
            #permuted_con_score, permuted_vox_score = permuted_score(
            #    Y, Y_pred, Y_baseline, n_permutations=100, seed=1)
            permuted_con_score = permuted_score(
                Y, Y_pred, Y_baseline, n_permutations=100, seed=1)
            permuted_con_scores.append(permuted_con_score)
            # permuted_vox_scores.append(permuted_vox_score)

        
mean_scores = np.array(scores).mean(0)
masker.inverse_transform(mean_scores).to_filename(
    os.path.join(write_dir, 'scores.nii.gz'))

mean_mean_scores = mean_scores.mean(0)
masker.inverse_transform(mean_mean_scores).to_filename(
    os.path.join(write_dir, 'mean_score.nii.gz'))

con_scores = np.array(con_scores)

mean_vox_scores = np.array(vox_scores).mean(0)

if n_permutations > 0:
    permuted_con_scores = np.array(permuted_con_scores).mean(0)
    con_percentile = np.percentile(permuted_con_scores.max(0), 95)
    #permuted_vox_scores = np.array(permuted_vox_scores).mean(0)
    #vox_percentile = np.percentile(permuted_vox_scores.max(0), 95)
    print(con_percentile, np.median(permuted_con_scores.max(0)),
          permuted_con_scores.max(0))
    #print(vox_percentile, np.median(permuted_vox_scores.max(0)))

###############################################################################
#  Ouputs, figures

from  nilearn.plotting import view_img
from nilearn.surface import vol_to_surf
from nilearn.plotting import plot_surf_stat_map
from nilearn.datasets import fetch_surf_fsaverage

#view_img(masker.inverse_transform(mean_mean_scores), vmax=.5, title='grand mean').open_in_browser()
mean_vox_scores_img = masker.inverse_transform(mean_vox_scores)
view_img(mean_vox_scores_img,
         vmax=.5,
         title='mean voxel scores').open_in_browser()

plt.figure(figsize=(8, 5))
nice_contrasts = [contrast.replace('_', ' ') for contrast in contrasts]
half_contrasts = len(contrasts) // 2
ax = plt.axes([0.28, 0.05, 0.23, .94]) # plt.subplot(121)
ax.boxplot(con_scores[:, :half_contrasts], vert=False)
ax.set_yticklabels(nice_contrasts[:half_contrasts])
ax.plot([0.017, 0.017], [0, len(contrasts)], 'g', linewidth=2)

#
#ax = plt.subplot(122)
ax = plt.axes([.76, 0.05, .23, .94])
ax.boxplot(con_scores[:, half_contrasts:], vert=False)
ax.set_yticklabels(nice_contrasts[half_contrasts:])
ax.plot([0.017, 0.017], [0, len(contrasts)], 'g', linewidth=2)
#
#plt.subplots_adjust(left=.4, right=.99, bottom=.03, top=.99)
plt.savefig(os.path.join(write_dir, 'score_per_contrast.png'))
plt.show(block=False)


print(n_dim, n_parcellations, np.mean(con_scores))

fsaverage = fetch_surf_fsaverage()
texture = vol_to_surf(mean_vox_scores_img, fsaverage.pial_right)
output_file = os.path.join(write_dir, 'score_rh.png')
plot_surf_stat_map(fsaverage.infl_right, texture, hemi='right',
                   colorbar=False, output_file=output_file,
                   threshold=0., bg_map=fsaverage.sulc_right, vmax=.25)
texture = vol_to_surf(mean_vox_scores_img, fsaverage.pial_left)
output_file = os.path.join(write_dir, 'score_lh.png')
plot_surf_stat_map(fsaverage.infl_left, texture, hemi='left',
                   colorbar=False, output_file=output_file,
                   threshold=0., bg_map=fsaverage.sulc_left, vmax=.25)

###############################################################################
# Compare with margulies
from nilearn.image import resample_to_img

gradient_img = 'gradient map from MArgulies et al. 2016.nii.gz'
# resampled_gradient = resample_to_img(gradient_img, mean_vox_scores_img)
gradient = masker.transform(gradient_img)

print(np.corrcoef(gradient, mean_vox_scores))



plt.show(block=False)
