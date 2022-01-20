"""
Cross-validation experiments to predict whether a given voxel belongs to a
pre-specified ROI based on its functional activation
This script uses subject-specific ROIs based on dual regression.

Authors: Ana Luisa Pinho, Bertrand Thirion

Last update: June 2020

Compatibility: Python 3.5

"""

import os
import glob
import json
import itertools

from joblib import Memory

import numpy as np
from scipy.special import comb

from ibc_public.utils_data import (data_parser, SMOOTH_DERIVATIVES,
                                   SUBJECTS, CONTRASTS)

from nilearn.input_data import NiftiMasker

from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
from sklearn.metrics import confusion_matrix

import matplotlib
matplotlib.use('Agg')  # to work in Drago
from matplotlib import pyplot as plt
from matplotlib import cm


def flatten(li):
    return sum(([x] if not isinstance(x, list) else flatten(x)
                for x in li), [])


def extract_signals(contrasts, df, roi_images):
    """ Extract signals from all regions, all subjects, all contrasts"""
    n_rois = len(np.unique(roi_images[0].get_data())) - 1
    X = []
    for s, subject in enumerate(SUBJECTS):
        img_paths = []
        for contrast in contrasts:
            img_path = df[df.contrast == contrast][df.subject == subject]\
                       .path.values[-1]
            img_paths.append(img_path)

        Xs = []
        for r in range(1, n_rois + 1):
            mask = math_img('img == %d' %r, img=roi_images[s])
            masker = NiftiMasker(mask_img=mask, smoothing_fwhm=10).fit()
            x = np.vstack(masker.transform(img_paths))
            Xs.append(x)
        X.append(Xs)
    return X


def cv_scores(X, y, G, clf):
    """
    Function to compute the cross-validation scores.
    """
    # Compute CV-scores
    cv = LeaveOneGroupOut()
    cv_scores = cross_val_score(clf, X, y, cv=cv, verbose=1, groups=G,
                                n_jobs=-1)
    return cv_scores


def compute_confusion_matrix(X, y, G, clf):
    """
    Function to compute the confusion matrix.
    """
    # Matrix
    bg = np.mod(G, 2)
    X_train, X_test = X[bg == 0], X[bg == 1]
    y_train, y_test = y[bg == 0], y[bg == 1]
    clf.fit(X_train, y_train)
    matrix = confusion_matrix(y_test, clf.predict(X_test))
    return matrix


def estimates_labels(parser_df, tasks_unique):
    """
    Get corresponding contrasts and tasks for estimates
    of the prediction models.
    """
    contrasts_list = []
    tasks_list = []
    for tt in tasks_unique:
        ct_list = parser_df[parser_df.task == tt]\
                           [parser_df.acquisition == 'ffx'].contrast.unique()
        contrasts_list.extend(ct_list)
        tasks_list.extend(np.repeat(tt, ct_list.size))
    contrasts_array = np.array(contrasts_list)
    tasks_array = np.array(tasks_list)

    return contrasts_array, tasks_array


def low_up_matrix(labels_array, upper_vals, lower_vals):
    """
    Matrix to store values between labels of type 1 in the upper triangle
    and scores of type 2 in the lower triangle. Diagonal is masked.
    """
    matrix = []
    x = len(labels_array) - 1
    count = 0
    for p in np.arange(len(labels_array)):
        row = []
        # Dummy scores -- lower part of the matrix
        if p > 0:
            step = 0
            for d in np.arange(p):
                if d == 0:
                    iteration = p - 1
                    row.extend([lower_vals[iteration]])
                else:
                    row.extend([lower_vals[iteration + step]])
                    iteration = iteration + step
                step = x - 1 - d
        # Diagonal
        row.extend([-1])
        # Svc scores -- upper part of the matrix
        lim_min = p*x - count
        lim_max = (p + 1)*x - (p + count)
        count = count + p
        if p < x:
            row.extend(upper_vals[lim_min:lim_max])
        # Append row to final matrix
        matrix.append(row)
    return matrix


# ######################### OUTPUT PATHS ######################################

main_parent_dir = '/neurospin/tmp/bthirion/'
alt_parent_dir = '/storage/store/work/agrilopi/'

# study_folder = 'ibc_roi_analysis_pallier'
study_folder = 'ibc_roi_analysis_hcplang900_II_z16'

if os.path.exists(main_parent_dir):
    cache = main_parent_dir + study_folder
else:
    cache = alt_parent_dir +  study_folder

mem = Memory(cachedir=cache, verbose=0)

# ############################## INPUTS #######################################

# ####### Select a subset of rois #######

# ### For Pallier's ROIs ###
# rois_folder = 'rois_pallier'
# rois_subset_idx = [0, 1, 4, 5, 6, 8]
# fig_labels = {'IFGorb': 'IFG pars orbitalis',
#               'IFGtri': 'IFG pars triangularis',
#               'TP': 'Temporal Pole',
#               'TPJ': 'Temporoparietal Junction',
#               'aSTS': 'anterior STS',
#               'pSTS': 'posterior STS'}

# ### For HCPlang900 ROIs ###
rois_folder = 'rois_hcplang900_z16'
fig_labels = {'left_FG': 'Fusiform Gyrus',
              'left_FP': 'Frontal Pole',
              'left_IFG': 'IFG pars orbitalis\n +' + \
                          '             \n IFG pars triangularis',
              'left_TPJ': 'Temporoparietal Junction',
              'left_aSTS_TP': 'anterior STS\n +' + \
                              '         \n Temporal Pole',
              'left_pSTS': 'posterior STS',
              'vmPFC': 'Ventromedial PFC'}

# ## Only with IFG, TPJ, aSTS, vmPFC
rois_subset_idx = [2, 3, 4, 6]

# Load the masks of the ROIs...
rois_path = os.path.abspath(rois_folder)
rois_list = sorted(glob.glob(os.path.join(rois_path, '*.nii.gz')))
rois_subset = [rois_list[x] for x in rois_subset_idx]
print(rois_subset)

# Access to the ffx z-maps of all participants
task_list = ['ArchiEmotional', 'ArchiSocial', 'ArchiSpatial',
             'ArchiStandard', 'HcpEmotion', 'HcpGambling',
             'HcpLanguage', 'HcpMotor', 'HcpRelational',
             'HcpSocial', 'HcpWm',
             'RSVPLanguage']

# BIDS conversion of task names
# Load dictionary file
with open(os.path.join('bids_postprocessed.json'), 'r') as f:
    task_dic = json.load(f)

TASKS = [task_dic[tkey] for tkey in task_list]
TASKS = flatten(TASKS)

parser = data_parser(derivatives=SMOOTH_DERIVATIVES,
                     subject_list=SUBJECTS,
                     conditions=CONTRASTS,
                     task_list=TASKS)

chosen_contrasts = ['sentence-word', 'sentence-jabberwocky', 'word-pseudo',
                    'word-consonant_string', 'pseudo-consonant_string',
                    'reading-listening', 'reading-checkerboard',
                    'computation-sentences']

not_chosen_contrasts = [c for c in
                        parser[parser.acquisition == 'ffx'].contrast.unique()
                        if c not in chosen_contrasts]

# ##############################################################
# Subject -specific ROIs
from nilearn.image import smooth_img, math_img
from nilearn.plotting import plot_roi
import ibc_public

_package_directory = os.path.dirname(
    os.path.abspath(ibc_public.utils_data.__file__))
mask_gm = os.path.join(
    _package_directory, '../ibc_data', 'gm_mask_1_5mm.nii.gz')
masker = NiftiMasker(mask_img=mask_gm).fit()
not_chosen_contrasts = [c for c in
                        parser[parser.acquisition == 'ffx'].contrast.unique()
                        if c not in chosen_contrasts]
all_contrasts = chosen_contrasts + not_chosen_contrasts


def make_rois_img(rois_aggregate):
    rois_img = rois_aggregate[0]
    for i in range(1, len(rois_aggregate)):
        rois_img = math_img('i1 + %d * i2' % (i + 1),
                            i1=rois_img, i2=rois_aggregate[i])
    return rois_img


def adapt_roi(rois_group, subject, parser, masker, contrasts_set):
    func = parser[parser.subject == subject][parser.acquisition == 'ffx']\
           [parser.contrast.isin(contrasts_set)].path.values
    X = masker.transform(func)
    indiv_rois = []
    for roi in rois_group:
        print(roi)
        smooth_roi = smooth_img(math_img('img * 1.0', img=roi), fwhm=20.)
        y = masker.transform(roi).astype(np.int) * 1.0
        n_voxels = int(np.sum(y > 0))
        y_ = y.dot(np.linalg.pinv(X)).dot(X)
        # masker.inverse_transform(y_)\
        #     .to_filename('/tmp/%s' % os.path.basename(roi))
        z = np.ravel(y_ * masker.transform(smooth_roi))
        threshold = z[np.argsort(-z)[n_voxels // 2]]
        z = z > threshold
        indiv_roi = masker.inverse_transform(z)
        indiv_rois.append(indiv_roi)
    rois_img = make_rois_img(indiv_rois)
    return rois_img

rois_img_ = make_rois_img(rois_subset)
rois_imgs = []

for subject in parser.subject.unique():
    rois_img = mem.cache(adapt_roi)(rois_subset, subject, parser, masker,
                                    not_chosen_contrasts)
    rois_imgs.append(rois_img)
    plot_roi(rois_img)

# X = extract_signals(chosen_contrasts, parser, rois_imgs)
X = extract_signals(all_contrasts, parser, rois_imgs)

##############################################################
# Classification EXPERIMENTS


# ### Build the Classifiers ###
# Main classifiers
svc = LinearSVC()
# Classifier to compute the chance level
dc = DummyClassifier(strategy="most_frequent")

# ### CV experiment for pairs of ROI's ###
roi_names = [os.path.basename(roi_name).split('.')[0]
             for roi_name in rois_subset]
all_svc_scores  = []
all_dummy_scores = []
pairs = list(itertools.combinations(range(len(rois_subset_idx)), 2))
for pair in pairs:

    X_pair = np.hstack([X[subject][i] for subject in range(len(X))
                        for i in pair]).T
    y_pair = np.hstack([np.repeat(i, X[subject][i].shape[1])
                        for subject in range(len(X)) for i in pair])
    subject_pair = np.hstack([np.repeat(subject, X[subject][i].shape[1])
                            for subject in range(len(X)) for i in pair])

    # Compute CV scores and print their means
    svc_scores_pair = cv_scores(X_pair, y_pair, subject_pair, svc)
    all_svc_scores.extend([svc_scores_pair.mean()])
    dummy_scores_pair = cv_scores(X_pair, y_pair, subject_pair, dc)
    all_dummy_scores.extend([dummy_scores_pair.mean()])
    # Compute and print confusion matrices
    svc_conf_matrix_pair = compute_confusion_matrix(X_pair, y_pair,
                                                    subject_pair, svc)
    print([roi_names[i] for i in pair])
    print('Prediction score w/ support vector classification',
          svc_scores_pair.mean())
    print('Chance level', dummy_scores_pair.mean())
    print('Confusion matrix w/ support vector classification:')
    print(svc_conf_matrix_pair)


# ## Score matrix for selected contrasts
# scores_matrix_selected = low_up_matrix(rois_subset, all_svc_scores,
#                                        all_dummy_scores)
# print(scores_matrix_selected)

# ##Score matrix for all contrasts
scores_matrix_all = low_up_matrix(rois_subset, all_svc_scores,
                                  all_dummy_scores)
print(scores_matrix_all)

# =============================================================================

# # For selected contrasts
# scores_matrix = scores_matrix_selected
# colour_map = cm.viridis
# # output_file = 'predictions_selected_adapted' # Pallier
# output_file = 'predictions_selected_adapted_hcp' # HCP

# # For all contrasts
scores_matrix = scores_matrix_all
colour_map = cm.copper
# output_file = 'predictions_selected_adapted' # Pallier
output_file = 'predictions_all_adapted_hcp' # HCP

# =============================================================================
# ### Plot ###

# Generate annotation matrix for plotting
n_repetitions = comb(len(rois_subset), 2, exact=True)
annot_matrix = low_up_matrix(rois_subset, np.repeat('PS', n_repetitions),
                             np.repeat('CL', n_repetitions))

# Mask diagonal
scores_array = np.array(scores_matrix)
scores_array = np.ma.masked_where(scores_array == -1, scores_array)
annot_array = np.array(annot_matrix)
annot_array = np.ma.masked_where(annot_array == -1, annot_array)

# Set colormap
cmap = colour_map
cmap.set_bad(color='white')

# Set figure
fig = plt.figure(figsize=(6.5, 5.))
# Axes of suplot
ax1 = plt.subplot(111)
# Axes of matrix
ax2 = plt.axes([.34,.36,.61,.61])

# Remove frame of subplot and matrix
for axis in ['top', 'bottom', 'left', 'right']:
    ax1.spines[axis].set_linewidth(0)
    ax2.spines[axis].set_linewidth(0)

# Generate PIL image
plt.imshow(scores_array, interpolation='nearest', cmap=cmap, vmin=.5, vmax=1)
# Loop over data dimensions and create text annotations.
for i in range(len(rois_subset)):
    for j in range(len(rois_subset)):
        text = plt.text(j, i, annot_array[i, j],
                       ha="center", va="center", color="w")

# Remove ticks and labels
ax1.xaxis.set_tick_params(width=0, color='w')
ax1.yaxis.set_tick_params(width=0, color='w')
ax2.xaxis.set_tick_params(width=0, color='w')
ax2.yaxis.set_tick_params(width=0, color='w')
ax1.set_xticklabels('')
ax1.set_yticklabels('')

# Set grid
# Gridlines based on minor ticks
ax2.set_xticks(np.arange(.5, len(rois_subset)-1, 1), minor=True);
ax2.set_yticks(np.arange(.5, len(rois_subset)-1, 1), minor=True);
ax2.grid(which='minor', color='w', linestyle='-', linewidth=3)
ax2.tick_params(which="minor", bottom=False, left=False)

# Tick labels
roi_labels = [fig_labels[roi_name] for roi_name in roi_names]
plt.xticks(np.arange(len(rois_subset)), roi_labels, rotation=45, ha='right',
           fontsize=12)
plt.yticks(np.arange(len(rois_subset)), roi_labels, fontsize=12)

# Legend
textstr = '\n'.join((
    'PS: Prediction Score',
    'CL: Chance Level'))
# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='linen', alpha=0.5)
# Place a text box in upper left in axes coords
ax1.text(.7, 0., textstr, transform=ax1.transAxes, fontsize=12,
        verticalalignment='top', bbox=props)

# Colorbar
plt.colorbar()

# Save figure
plt.savefig(os.path.join(cache, output_file + '.png'), dpi=600)

