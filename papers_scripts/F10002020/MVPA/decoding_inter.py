import glob
import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from joblib import Parallel, delayed

from nilearn.input_data import NiftiMasker
from nistats.first_level_model import FirstLevelModel
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import LeavePGroupsOut, LeaveOneGroupOut, \
                                    StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

import ibc_public.utils_data
from ibc_public.utils_data import get_subject_session
from utils_tonotopy import make_dmtx, make_dmtxs, parse_z

# IBC package path
ibc = '/storage/store/data/ibc'
_package_directory = os.path.dirname(os.path.abspath
                                     (ibc_public.utils_data.__file__))

# Input dir
data_dir = os.path.join(ibc, 'smooth_derivatives')

# Output for scores
metrics_dir = os.path.join(os.getcwd(), 'metrics')
if not os.path.exists(metrics_dir):
    os.mkdir(metrics_dir)

# 1.5mm mask
mask_gm = os.path.join(_package_directory, '../ibc_data/gm_mask_1_5mm.nii.gz')

# Resolution of the images to use
use_3mm = False

if use_3mm:
    mask_gm = os.path.join(_package_directory, '../ibc_data/gm_mask_3mm.nii.gz')
    data_dir = os.path.join(ibc, '3mm')

# GLM mode
glm_mode = 'z_maps'

# Print and save confusion matrix
conf_matrix = True

# Output dir for images
write_dir = '/storage/store/work/juanjesus/decoding/tonotopy/{}'.format(glm_mode)
if not os.path.exists(write_dir):
    os.makedirs(write_dir)

# Repetition time
t_r = 2.0

# Task list
task_list = ['audio1', 'audio2']

# Contitions
conditions = ['voice', 'nature', 'animal',
              'music', 'speech', 'tools', 'tool']


# %% Functions
def mask_data(masker, file):
    masked_file = masker.transform(file)
    return masked_file


def _fit_glms(fmri, run, trial, dmtx, conds, sub_dir, model, save=False, mumford=False):
    if mumford:
        conditions = conds
    else:
        conditions = [condition + "_00" for condition in conds]

    column = dmtx.columns.intersection(conditions)
    assert len(column) == 1
    label = column[0].strip('_00')
    full_trial = "{}_{}".format(label, trial)
    filename = os.path.join(sub_dir, "{}_{}.nii.gz".format(run, full_trial))

    if os.path.exists(filename):
        print("Image {} already computed! Skipping...".format(filename))
        image = []

    else:
        model.fit(fmri, design_matrices=[dmtx])

        image = model.compute_contrast(dmtx.columns == column[0])

        if save:
            image.to_filename(filename)
            print("Saved {}".format(filename))

    return image, filename


def get_images_inter(session_list, conditions, data_dir, write_dir,
                     mask, t_r, use_3mm=False, save=True, glm_mode='z_maps'):
    """
    Run first level model for one subject and return images and filenames.

    Parameters
    ----------

    session_list: list of tuples
                  First element is the subject number, and the second element
                  is a list of strings with all session numbers for that subject

    conditions: list of str
                Names of all the conditions of the experiment

    data_dir: str or path object
              Path to the input images

    write_dir: str or path object
               Output path for the new images

    mask: mask object
          Mask that will be used on the data

    t_r: float
         Repetition time of the experimental task

    use_3mm: bool, default=False
             If true, use 3mm images, otherwise use 1.5mm

    save: bool, default: False
          If True, the images will be saved to disk

    glm_mode: str, ['z_maps', 'glms'], default: 'z_maps'
              How to fit the design matrices. 'z_maps' will get z_maps for
              averaged trials for every condition, while 'glms' will create
              one design matrix for every trial, and fit hte model separately.

    Returns
    -------

    images: list of np.array objects
            Images corresponding to single trials of the experiment each

    names: list of str
           Filenames of the images containing the label of each image
    """

    if glm_mode == 'z_maps':
        # Get data
        images = []
        names = parse_z(data_dir, conditions)

    elif glm_mode == 'glms':

        try:
            model = FirstLevelModel(mask_img=mask,
                                    smoothing_fwhm=5,
                                    t_r=t_r, high_pass=.01)
        except:
            print("There was an error generating a model with high-pass filter "
                  ", generating alternative model...")

            model = FirstLevelModel(mask_img=mask,
                                    smoothing_fwhm=5,
                                    t_r=t_r, period_cut=128, )

        images = []
        names = []

        # Handle for a task with multiple acquisition sessions
        for sessions_info in session_list:
            subject = sessions_info[0]
            sessions = sessions_info[1]

            runs = []
            events = []
            confounds = []

            for session in sessions:
                path = os.path.join(data_dir, "{}/{}/func".format(subject, session))

                if use_3mm:
                    runs_path = os.path.join(data_dir,
                                             "../3mm/{}/{}/func".format(subject, session))
                else:
                    runs_path = path

                this_runs = sorted(glob.glob(os.path.join(runs_path, 'w*bold.nii.gz')))
                this_events = sorted(glob.glob(os.path.join(path, '*.tsv')))
                this_confounds = sorted(glob.glob(os.path.join(path, 'rp_*.txt')))

                runs.extend(this_runs)
                events.extend(this_events)
                confounds.extend(this_confounds)

            # Create output dir
            sub_dir = os.path.join(write_dir, subject)
            if not os.path.exists(sub_dir):
                os.mkdir(sub_dir)

            if use_3mm:
                res = '3mm'
            else:
                res = '1.5mm'

            for fmri, events, confounds in zip(runs, events, confounds):
                # Get run number from filename
                run = os.path.basename(fmri).strip('.nii.gz').split('_')[-2][-2:]

                dmtx_list, _ = make_dmtxs(events, fmri, confounds, mumford=False)

                bad_trials = ['silence_00', 'catch_00', 'fixation_00']
                dmtx_list = [dmtx for dmtx in dmtx_list if not any(bad_trial in dmtx.columns
                                                                   for bad_trial in bad_trials)]

                this_images = []
                this_filenames = []
                print("Starting fit for subject {}, run {}".format(subject, run))
                for j, dmtx in enumerate(dmtx_list):
                    image, filename = _fit_glms(fmri, run, j, dmtx, conditions,
                                                sub_dir, model, mumford=False, save=save)

                    if not image:
                        this_images.extend(image)
                    else:
                        this_images.append(image)

                    this_filenames.append(filename)

                images.extend(this_images)
                names.extend(this_filenames)

    else:
        raise ValueError("Invalid glm_mode! It must be 'z_maps' "
                         "or 'single_glm, it was: {}".format(glm_mode))

    return images, names


def compute_conf_matrix(pipeline, images, labels, cv):
    """Compute and save confusion matrix"""

    # Convert everything to arrays
    labels = np.array(labels)

    # Get labels for the display
    unique_labels, order = np.unique(labels, return_index=True)
    unique_labels = unique_labels[np.argsort(order)]

    # Initialize conf matrix list
    n_splits = cv.get_n_splits(images, labels)
    n_classes = len(unique_labels)

    conf_matrix_list = np.zeros((n_splits, n_classes, n_classes), dtype=np.float32)

    # Separate data
    for n_split, (train_index, test_index) in enumerate(list(cv.split(images, labels))):

        images_train, images_test = images[train_index], images[test_index]
        labels_train, labels_test = labels[train_index], labels[test_index]

        print('Fitting clf for confusion matrix...')

        pipeline.fit(images_train, labels_train)

        disp = plot_confusion_matrix(pipeline,
                                     images_test,
                                     labels_test,
                                     display_labels=unique_labels,
                                     cmap=plt.cm.Blues,
                                     normalize='true')

        # Save the conf matrix
        matrix_ax = disp.ax_
        matrix_ax.figure.savefig(os.path.join(metrics_dir,
                                              'conf_matrix_{}_{}.png'.format(glm_mode, n_split)),
                                 dpi=200)

        # Append the array to the list of conf matrices
        conf_matrix_list[n_split] = disp.confusion_matrix

    # Get mean confusion matrix from all the folds, and plot it
    mean_conf_matrix = conf_matrix_list.mean(axis=0)
    disp.confusion_matrix = mean_conf_matrix

    avg_disp = disp.plot(cmap=plt.cm.Blues)
    avg_ax = avg_disp.ax_
    avg_ax.figure.savefig(os.path.join(metrics_dir,
                                       'avg_conf_matrix_{}.png'.format(glm_mode)),
                          dpi=200)


def decode_inter(imgs, names, mask, pipeline, conf_matrix=False):
    """
    Run decoding and return a cross-validation score

    Parameters
    ----------

    imgs: list of np.array objects or empty list
          Images corresponding to single trials of the experiment each.
          It can be empty, meaning we will get info from the filenames

    names: list of str
           Filenames of each image

    mask: mask object
          Mask that will be used for the data

    pipeline: sklearn pipeline
              Steps to run

    conf_matrix: bool, default False
                 Whether to compute and save the confusion matrix or not

    Returns
    -------

    cv_score: list of float
              Cross-validation score
    """

    # Get labels, runs and subs from filenames
    if glm_mode == 'glms':
        labels = [name.split('/')[-1].split('_')[1] for name in names]
        subs = [name.split('/')[-1].split('_')[0] for name in names]
    elif glm_mode == 'z_maps':
        labels = [name.split('/')[-1].split('-')[0] for name in names]
        subs = [name.split('/')[-5].split('-')[-1] for name in names]
        runs = [int(name.split("/")[-3].split("_")[-2]) for name in names]

    print('Masking data...')
    masker = NiftiMasker(mask_img=mask).fit()

    if imgs:
        fmri_masked = Parallel(n_jobs=15, verbose=True)(delayed(mask_data)(masker, img)
                                                        for img in imgs)
    else:
        fmri_masked = Parallel(n_jobs=15, verbose=True)(delayed(mask_data)(masker, filename)
                                                        for filename in names)

    fmri_masked = np.concatenate(fmri_masked, axis=0)
    print("Running cross-validation...")
    # cv = LeavePGroupsOut(n_groups=2)
    cv = StratifiedKFold(n_splits=66, shuffle=True, random_state=42)

    cv_score = cross_validate(pipeline,
                              fmri_masked,
                              labels,
                              cv=cv,
                              groups=runs,
                              return_train_score=True)

    if conf_matrix:
        compute_conf_matrix(pipeline,
                            fmri_masked,
                            labels,
                            cv=cv)

    return cv_score


def make_decode_inter(session_list, conditions, data_dir, write_dir,
                      mask, t_r, pipeline, use_3mm=False, glm_mode='z_maps',
                      conf_matrix=False):
    """
    Run the entire decoding pipeline and return the scores

    Parameters
    ----------

    session_list: list of tuples
                  First element is the subject number, and the second element
                  is a list of strings with all session numbers for that subject

    conditions: list of str
                Labels for classification

    data_dir: str or path object
            Input directory

    write_dir: str or path object
            Output directory

    mask: mask object
        Mask to standardize the images before classification

    t_r: float
        Repetition time corresponding to the tasks

    pipeline: sklearn pipeline object
            Steps for the classification

    use_3mm: bool, default=False
             Whether or not to fetch 3mm images

    conf_matrix: bool, default False
                 Whether to compute and save the confusion matrix or not

    Returns
    -------

    cv_score: dict
              Cross-validation scores
    subject: int
             Subject number
    """

    imgs, names = get_images_inter(session_list=session_list,
                                   conditions=conditions,
                                   data_dir=data_dir,
                                   write_dir=write_dir,
                                   mask=mask,
                                   t_r=t_r,
                                   use_3mm=use_3mm,
                                   glm_mode=glm_mode)

    cv_score = decode_inter(imgs, names, mask, pipeline, conf_matrix=conf_matrix)

    return cv_score


# Model
svc = LinearSVC(max_iter=10000)
feature_selection = SelectPercentile(f_classif, percentile=25)
pipeline = Pipeline([('anova', feature_selection), ('svc', svc)])

# Subjects
session_1 = sorted(get_subject_session([task_list[0]]))
session_2 = sorted(get_subject_session([task_list[1]]))

session_list = [(sub1, (ses1, ses2)) for
                (sub1, ses1), (sub2, ses2) in zip(session_1, session_2)]

# Classification
scores = make_decode_inter(session_list,
                           conditions,
                           data_dir=data_dir,
                           write_dir=write_dir,
                           mask=mask_gm,
                           t_r=t_r,
                           pipeline=pipeline,
                           use_3mm=False,
                           glm_mode=glm_mode,
                           conf_matrix=conf_matrix)

# Print and save the scores
print("Scores for inter-sub classification")
print("Train score: {}".format(scores['train_score']))
print("Test score: {}".format(scores['test_score']))
filename = 'cv_score_inter.tsv'
score_df = pd.DataFrame(scores)
score_df.to_csv(os.path.join(metrics_dir, filename), sep='\t', index=False)
