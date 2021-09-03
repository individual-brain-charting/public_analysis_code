"""
Script for RS-fMRI to task-fMRI analysis.

Works on IBC 3mm data 
"""

import os
import glob
from nilearn.input_data import NiftiMasker
import pandas as pd
from joblib import Memory, Parallel, delayed
import ibc_public
import numpy as np
from utils import (
    make_dictionary, adapt_components, make_parcellations,
    predict_Y_multiparcel, permuted_score, fit_regressions)


# cache
cache = '/storage/store/work/bthirion/rsfmri2tfmri'
# cache = '/storage/store2/bthirion/rsfmri2tfmri'
write_dir = cache
if not os.path.exists(write_dir):
    os.mkdir(write_dir)
memory = Memory(cache, verbose=0)
n_jobs = 10

###############################################################################
# Get and fetch data: RS-fMRI and T-fMRI

# resting-state fMRI
DATADIR = '/storage/store/data/HCP900/'
wc = os.path.join(DATADIR, '*', 'MNINonLinear', 'Results',
                  'rfMRI_REST*', 'rfMRI_REST*LR.nii.gz')
rs_fmri = sorted(glob.glob(wc))
wc = os.path.join(DATADIR, '*', 'MNINonLinear', 'Results',
                  'rfMRI_REST*', 'rfMRI_REST*RL.nii.gz')
rs_fmri += sorted(glob.glob(wc))

subjects = []
for img in rs_fmri:
    subject = img.split('/')[-5]
    subjects.append(subject)

rs_fmri_db = pd.DataFrame({'subject': subjects,
                           'path': rs_fmri})

# task fmri
mem = Memory(cachedir=cache, verbose=0)
subjects = rs_fmri_db.subject.unique()
n_subjects = len(subjects)

# Access to the data
task_list = ['EMOTION', 'GAMBLING', 'LANGUAGE', 'MOTOR', 'RELATIONAL',
             'SOCIAL', 'WM']
wc = os.path.join(DATADIR, 'glm', '*', '*', 'level2', 'z_maps', '*.nii.gz')
tfmri = sorted(glob.glob(wc))
wc = os.path.join(DATADIR, 'glm', '*', '*', 'level2', 'z_maps', '*.nii.gz')
tfmri += sorted(glob.glob(wc))

subjects = []
task = []
contrasts = []
for img in tfmri:
    parts = img.split('/')
    subjects.append(parts[-5])
    task.append(parts[-4])
    contrasts.append(parts[-1][2:-7])
    
db = pd.DataFrame({'subject': subjects,
                   'path': tfmri,
                   'task': task,
                   'contrast': contrasts,})

df = db[db.task.isin(task_list)]
df = df.sort_values(by=['subject', 'task', 'contrast'])
contrasts = df.contrast.unique()

# mask of grey matter
mask_gm = 'gm_mask_2mm.nii.gz'

subjects = [subject for subject in rs_fmri_db.subject.unique()
            if subject in df.subject.unique()]
subjects = [subject for subject in subjects if subject != '122620']

n_subjects = len(subjects)
n_subjects = 200
subjects = subjects[:n_subjects]
n_train = 60

###############################################################################
# Dictionary learning of RS-fMRI

from nilearn.plotting import plot_prob_atlas, plot_stat_map
import matplotlib.pyplot as plt

n_components = 100
n_samples = 60

rs_fmri = np.hstack([rs_fmri_db[rs_fmri_db.subject == subject].path.values
                     for subject in subjects[:n_train]])
rng = np.random.RandomState(1)
random_mask = rng.random_integers(0, len(rs_fmri), n_samples)

make_dictionary = mem.cache(make_dictionary)
dictlearning_components_img, Y = make_dictionary(
    rs_fmri[random_mask], n_components, cache, mask_gm, n_jobs=n_jobs)

dictlearning_components_img.to_filename(
    os.path.join(write_dir, 'components.nii.gz'))

# visualize the results
plot_prob_atlas(dictlearning_components_img,
                title='All DictLearning components',
                output_file=os.path.join(write_dir, 'atlas.png'))

###############################################################################
# Dual regression to get individual components
masker = NiftiMasker(mask_img=mask_gm, smoothing_fwhm=4, memory=cache).fit()
dummy_masker = NiftiMasker(mask_img=mask_gm, memory=cache).fit()
n_dim = 200


def adapt_components_(Y, subjects, rs_fmri_db, masker, n_dim, n_jobs):
    return Parallel(n_jobs=n_jobs)(delayed(adapt_components)(
        Y, subject, rs_fmri_db, masker, n_dim) for subject in subjects)

adapt_components_ =  mem.cache(adapt_components_)
individual_components = adapt_components_(Y, subjects, rs_fmri_db, masker, n_dim, n_jobs)

# visualize the results
"""
for i, subject in enumerate(subjects):
    individual_components_img = masker.inverse_transform(
        individual_components[i])
"""

###############################################################################
# Generate brain parcellations
from nilearn.regions import Parcellations
n_parcellations = 20
n_parcels = 256

ward = Parcellations(method='ward', n_parcels=n_parcels,
                     standardize=False, smoothing_fwhm=4.,
                     memory=cache, memory_level=1,
                     verbose=1, mask=mask_gm)

make_parcellations = memory.cache(make_parcellations)

parcellations = make_parcellations(ward, rs_fmri, n_parcellations, n_jobs)

###############################################################################
# Cross-validated predictions
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from utils import predict_Y
clf = RidgeCV()


data =  np.array([masker.transform([df[df.subject == subject][
    df.contrast == contrast].path.values[-1] for contrast in contrasts])
         for subject in subjects])

# training the model
models = Parallel(n_jobs=n_jobs)(delayed(fit_regressions)(
    individual_components, data, parcellations,
    dummy_masker, clf, i) for i, subject in enumerate(subjects[:n_train]))

# create average model
average_models = [np.zeros_like(models[0][i]) for i in range(len(models[0]))]
for model in models[:n_train]:
    for i in range(len(model)):
        average_models[i] += model[i]

for i in range(len(average_models)):
    average_models[i] /= n_train



scores = []
vox_scores = []
con_scores = []
permuted_con_scores = []
permuted_vox_scores = []
n_permutations = 0

train_index = np.arange(n_train)  #subjects[: n_train]
test_index = np.arange(n_train, n_subjects)  #subjects[n_train:]

# construct the predicted maps
for j in test_index:
    X = individual_components[j]
    Y = data[j]
    Y_baseline = np.mean(Y[train_index], 0)
    Y_pred = predict_Y(
        parcellations, dummy_masker, 
        n_parcels, X, average_models)
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

plt.figure(figsize=(8, 6))
nice_contrasts = [contrast.replace('_', ' ') for contrast in contrasts]
half_contrasts = len(contrasts) // 2
ax = plt.axes([0.2, 0.05, 0.29, .94]) # plt.subplot(121)
ax.boxplot(con_scores[:, :half_contrasts], vert=False)
ax.set_yticklabels(nice_contrasts[:half_contrasts])
ax.plot([0.017, 0.017], [0, len(contrasts)], 'g', linewidth=2)

#
#ax = plt.subplot(122)
ax = plt.axes([.7, 0.05, .29, .94])
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
                   threshold=0., bg_map=fsaverage.sulc_right, vmax=.5)
texture = vol_to_surf(mean_vox_scores_img, fsaverage.pial_left)
output_file = os.path.join(write_dir, 'score_lh.png')
plot_surf_stat_map(fsaverage.infl_left, texture, hemi='left',
                   colorbar=False, output_file=output_file,
                   threshold=0., bg_map=fsaverage.sulc_left, vmax=.5)

# make the colormap
output_file = os.path.join(write_dir, 'cmap.svg')
plot_surf_stat_map(fsaverage.infl_left, texture, hemi='left',
                   colorbar=True, output_file=output_file,
                   threshold=0., bg_map=fsaverage.sulc_left, vmax=.5)

plt.figure(figsize=(8, 3), facecolor='w', edgecolor='w')
imgs = ['lh', 'rh']
for i, img in enumerate(imgs):
    ax = plt.axes([.45 * np.mod(i, 2), 0, .45, 1])
    path = os.path.join(write_dir, 'score_%s.png' % img)
    img = plt.imread(path)
    # img[img.sum(2) == 4, :3] = 0
    ax.imshow(img[60:-60, 90:-90])
    plt.axis('off')

# colormap
#ax = plt.axes([.9, 0, .1, 1])
#path = os.path.join(write_dir, 'cmap.png' % img)
#img = plt.imread(path)
#ax.imshow(img[60:-60, -90:])
#plt.axis('off')
plt.savefig(os.path.join(write_dir, 'montage_hcp.png'), facecolor='w')
#plt.savefig(os.path.join(write_dir, 'montage_hcp.svg'), facecolor='w')
#plt.savefig(os.path.join(write_dir, 'montage_hcp.pdf'), facecolor='w')
plt.show(block=False)


###############################################################################
# Compare with margulies
from nilearn.image import resample_to_img

gradient_img = 'gradient map from MArgulies et al. 2016.nii.gz'
# resampled_gradient = resample_to_img(gradient_img, mean_vox_scores_img)
gradient = masker.transform(gradient_img)

print(np.corrcoef(gradient, mean_vox_scores))



plt.show(block=False)
