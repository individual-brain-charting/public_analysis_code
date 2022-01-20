"""
Script in which one predicts brain maps from others

authors: Bertrand Thirion, Ana Luisa Pinho

Last update: May 2020

Compatibility: Python 3.5
"""
import os
import glob
import json

from joblib import Memory, Parallel, delayed

import numpy as np
import nibabel as nib

from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_score
from nilearn.image import math_img
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt

from nilearn.regions import Parcellations
from nilearn.input_data import NiftiMasker
from nilearn.image import math_img, mean_img, concat_imgs
from nilearn import plotting
from nilearn import datasets
from nilearn import surface


from ibc_public.utils_data import (
    data_parser, SMOOTH_DERIVATIVES, DERIVATIVES, SUBJECTS, LABELS, CONTRASTS,
    make_surf_db)
import ibc_public


cache = '/storage/store/work/agrilopi/predictive_model'
write_dir = '/storage/store/work/agrilopi/predictive_model/results_3out_100'
if not os.path.exists(write_dir):
    os.mkdir(write_dir)

mem = Memory(cachedir=cache, verbose=0)
subject_list = SUBJECTS
n_subjects = len(subject_list)

# Access to the data
task_list = ['ArchiEmotional', 'ArchiSocial', 'ArchiSpatial',
             'ArchiStandard', 'HcpEmotion', 'HcpGambling',
             'HcpLanguage', 'HcpMotor', 'HcpRelational',
             'HcpSocial', 'HcpWm',
             'RSVPLanguage']

# BIDS conversion of task names
# Load dictionary file
with open(os.path.join('bids_postprocessed.json'), 'r') as f:
    task_dic = json.load(f)

task_tags = [task_dic[tkey] for tkey in task_list]
task_tags = [item for sublist in task_tags for item in sublist]

# Mask of the grey matter across subjects
db = data_parser(derivatives=SMOOTH_DERIVATIVES, subject_list=subject_list,
                 conditions=CONTRASTS, task_list=task_tags)
df = db[db.task.isin(task_list)]
df = df.sort_values(by=['subject', 'task', 'contrast'])

_package_directory = os.path.dirname(
    os.path.abspath(ibc_public.utils_data.__file__))
mask_gm = os.path.join(
    _package_directory, '../ibc_data', 'gm_mask_1_5mm.nii.gz')


# synopsis
# load the fmri data
# parcellate the brain
# for each parcel, create a cross_validation scheme

paths = []
subjects = []
tasks = []
contrasts = []
for subject in subject_list:
    for contrast in df.contrast.unique():
        mask = (df.contrast == contrast).values *\
               (df.subject == subject).values *\
               (df.acquisition == 'ffx').values
        paths.append(df[mask].path.values[-1])
        subjects.append(subject)
        contrasts.append(contrast)
        tasks.append(df[mask].task.values[-1])

tasks = np.array(tasks)[:len(np.unique(contrasts))]
subjects = np.array(subjects)
paths = np.array(paths)
n_contrasts = len(df.contrast.unique())
n_parcels = 100

ward = Parcellations(method='ward', n_parcels=n_parcels, mask=mask_gm,
                     standardize=False, memory_level=1, memory=cache,
                     verbose=1)
# Call fit on functional dataset: single subject (less samples).
ward.fit(paths)


def _scorer(clf, X, Y):
    """ Custom scorer"""
    if Y.ndim > 1:
        return 1 - np.sum((Y - clf.predict(X)) ** 2, 1) / np.sum((Y) ** 2, 1)
    else:
        return 1 - (Y - clf.predict(X)) ** 2 / Y ** 2


def _cross_val_score(clf, X, Y, cv, groups, _scorer):
    """ Custom cross_val score"""
    scores = []
    n_voxels = np.sum(groups == 0)
    for train, test in cv.split(X, Y, groups):
        X_train, X_test = X[train], X[test]
        Y_train, Y_test = Y[train], Y[test]
        clf.fit(X_train, Y_train)
        #  print(clf.alpha_)
        n_tests = len(np.unique(groups[test]))
        score = _scorer(clf, X_test, Y_test).reshape(n_tests, n_voxels)
        scores.append(score.mean(0))
    return np.mean(scores, 0)


def _permute_Y(Y, groups):
    """"""
    permutation = np.unique(groups)
    np.random.shuffle(permutation)
    return np.vstack((Y[groups == i] for i in permutation))


def get_cv_score(i, ward, paths, clf, gkf, permutation=False):
    score = {}
    mask_i = math_img('img == %d' % i, img=ward.labels_img_)
    masker = NiftiMasker(
        mask_img=mask_i, memory=cache, smoothing_fwhm=None).fit()
    Z = np.array([masker.transform(paths[subjects == subject]).T
                 for subject in subject_list])
    Z = np.reshape(Z, (len(subject_list) * Z.shape[1], n_contrasts)).T
    groups = np.repeat(np.arange(n_subjects), Z.shape[1] / n_subjects)
    for task in task_list:
        X = Z[tasks != task].T
        Y = Z[tasks == task].T
        if permutation:
            Y = _permute_Y(Y, groups)
        score[task] = _cross_val_score(clf, X, Y, gkf, groups, _scorer)
    return score


# affix = 'loocv'
# nsplits = n_subjects

# affix = '2out'
# nsplits = n_subjects // 2

affix = '3out'
nsplits = n_subjects // 3

# affix = 'split-half'
# nsplits = n_subjects // 6

gkf = GroupKFold(n_splits=nsplits)
clf = RidgeCV()
# clf = DummyClassifier(strategy="most_frequent")
# clf = GradientBoostingRegressor()
# clf = RandomForestRegressor()
scores = Parallel(n_jobs=2)(
    delayed(get_cv_score)(i, ward, paths, clf, gkf)
    for i in range(1, 1 + n_parcels))

score_imgs = []
for task in task_list:
    score_map = math_img('0. * img ', img=ward.labels_img_).get_data()
    for i in range(1, 1 + n_parcels):
        score_map[ward.labels_img_.get_data() == i] = scores[i - 1][task]

    score_img = nib.Nifti1Image(score_map, ward.labels_img_.affine)
    filename = os.path.join(write_dir, 'score_' + affix + '_%s.nii.gz' % task)
    score_img.to_filename(filename)
    score_imgs.append(score_img)


# Run with scrambled subjects to see the difference
scores_ = Parallel(n_jobs=2)(
    delayed(get_cv_score)(i, ward, paths, clf, gkf, permutation=True)
    for i in range(1, 1 + n_parcels))

for task in task_list:
    diff_score_map = math_img('0. * img ', img=ward.labels_img_).get_data()
    for i in range(1, 1 + n_parcels):
        diff_score_map[ward.labels_img_.get_data() == i] = scores_[i - 1][task]

    diff_score_img = nib.Nifti1Image(diff_score_map,
                                     ward.labels_img_.affine)
    filename = os.path.join(write_dir, 'diff_score_' + affix +
                            '_%s.nii.gz' % task)
    diff_score_img.to_filename(filename)


# Now make images out of all that

imgs = [os.path.join(write_dir, 'score_' + affix + '_%s.nii.gz' % task)
        for task in task_list]
mean_img_ = mean_img([math_img('np.maximum(i1, 0)', i1=img) for img in imgs])

# compute the median
masker = NiftiMasker(mask_img=mask_gm).fit()
X = masker.transform(imgs)
median_img_ = masker.inverse_transform(np.median(X, 0))

max_img = imgs[0]
for img in imgs:
    max_img = math_img('np.maximum(i1, i2)', i1=max_img, i2=img)

max_img.to_filename(os.path.join(write_dir, 'max_score_' + affix + '.nii.gz'))

plot_img_ = max_img

# plotting.view_img(plot_img_, vmin=-1, vmax=1).open_in_browser()
fsaverage = datasets.fetch_surf_fsaverage()
subplots = []
for hemi in ['left', 'right']:
    bg_map = fsaverage.sulc_right
    mesh = fsaverage.pial_right
    inflated = fsaverage.infl_right
    if hemi == 'left':
        bg_map = fsaverage.sulc_left
        mesh = fsaverage.pial_left
        inflated = fsaverage.infl_left
    texture = surface.vol_to_surf(plot_img_, mesh)
    output_file = os.path.join(write_dir, 'score_' + affix +
                               '_%s_lateral.png' % hemi)
    plotting.plot_surf_stat_map(
        inflated, texture, hemi=hemi, colorbar=True, view='lateral',
        bg_map=bg_map, vmax=.5, title='')
    plt.savefig(output_file, dpi=300)
    subplots.append(output_file)
    output_file = os.path.join(write_dir, 'score_' + affix +
                               '_%s_medial.png' % hemi)
    plotting.plot_surf_stat_map(
        inflated, texture, hemi=hemi, colorbar=True, view='medial',
        bg_map=bg_map, vmax=.5, title='')
    plt.savefig(output_file, dpi=300)
    subplots.append(output_file)

# colormap
dummy_file = '/tmp/tmp.png'
plotting.plot_stat_map(plot_img_, vmax=.5, colorbar=True)
plt.savefig(dummy_file, dpi=100)

# montage
plt.figure(facecolor='w', edgecolor='w')
for i in range(4):
    ax = plt.axes([.45 * (i // 2), .5 * np.mod(i, 2), .45, .5])
    img = plt.imread(subplots[i])
    ax.imshow(img[190: -190, 290: -290])
    plt.axis('off')

# add color bar
ax = plt.axes([.86, -.135, .125, 1.35])
img = plt.imread(dummy_file)
ax.imshow(img[:, -60:])
ax.axis('off')
plt.subplots_adjust(left=0, right=1, hspace=0, wspace=0, bottom=0, top=1)
output_file = os.path.join(write_dir, 'all_score_' + affix + '.svg')
plt.savefig(output_file)

# histogram of voxels with positive R2
hits = []
diff_hits = []
for task in task_list:
    img = os.path.join(write_dir, 'score_' + affix + '_%s.nii.gz' % task)
    hits.append(np.sum(nib.load(img).get_data() > 0))
    img = os.path.join(write_dir, 'diff_score_' + affix + '_%s.nii.gz' % task)
    diff_hits.append(np.sum(nib.load(img).get_data() > 0))

n_voxels = np.sum(ward.mask_img_.get_data() > 0)
n_voxels = n_voxels.astype(float)
hits = np.array(hits) / n_voxels
diff_hits = np.array(diff_hits) / n_voxels
np.savez(os.path.join(write_dir, 'predictive_model_' + affix + '.npz'),
         hits=hits, diff_hits=diff_hits)

plt.figure(figsize=(8, 6))
ylabels = [task.replace('_', ' ') for task in task_list]
ylabels = [ylabel.replace('wm', 'Working Memory') for ylabel in ylabels]
ylabels = [ylabel.title() for ylabel in ylabels]
ylabels = [ylabel.replace('rchi', 'rchi'.upper()) for ylabel in ylabels]
ylabels = [ylabel.replace('cp', 'cp'.upper()) for ylabel in ylabels]
ylabels = [ylabel.replace('svp', 'svp'.upper()) for ylabel in ylabels]
y_pos = np.arange(len(task_list))
plt.barh(y_pos + .4, hits, height=.4, color='g', label='consistent')
plt.barh(y_pos, diff_hits, height=.4, color='g', label='scrambled',
         alpha=.5, tick_label=ylabels)
plt.legend(prop={'size': 14})
plt.axis('tight')
plt.xlabel('Proportion of voxels with a positive $R^{2}$−score', fontsize=14,
           fontweight='bold')
plt.subplots_adjust(left=.3, right=.945, bottom=.1, top=.98)
props = dict(boxstyle='round', facecolor='wheat', alpha=1.)
plt.text(0.225, 7.05, 'Chance level = 0.00', fontsize=12, bbox=props)
plt.xlim([0., .35])
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.savefig(os.path.join(write_dir, 'diff_score_' + affix + '.svg'))
plt.show(block=False)

print(np.sum(max_img.get_data() > 0) / n_voxels)
pos = np.sum(concat_imgs(imgs).get_data() > 0) / (12 * n_voxels)
print(pos)
print(hits)
print(diff_hits)
