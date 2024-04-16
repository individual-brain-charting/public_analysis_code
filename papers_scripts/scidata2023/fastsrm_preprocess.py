"""
Mask preprocessed data for FastSRM

Authors: Ana Luisa Pinho, Hugo Richard

Created: October 2020
Last update: July 2022

Compatibility: Python 3.7
"""

import os
import glob
import re
from shutil import copy2

from time import time
from joblib import Parallel, delayed

import numpy as np

import nibabel
from nilearn.image import new_img_like
from nilearn.input_data import NiftiMasker
from nilearn.surface import load_surf_data

import ibc_public.utils_data
from ibc_public.utils_data import data_parser, DERIVATIVES

from ibc_fastsrm_utils import reshape_preprocdata


# ######################### FUNCTIONS ##################################


def separate_and_mask(func, mask_fn, input_parent_dir, output_dir):
    # Load data
    N = nibabel.load(func).get_data()
    I = new_img_like(func, N)
    # Mask data
    masker = NiftiMasker(mask_img=mask_fn, detrend=True, standardize=True,
                         smoothing_fwhm=5).fit()
    # Transpose the data to fit with SRM conventions
    X = masker.transform(I).T
    # Save data
    if input_parent_dir == '/neurospin/tmp/agrilopi':
        np.save(os.path.join(output_dir, func[46:-7] + "_masked"), X)
    elif input_parent_dir == '/storage/store/work/agrilopi/':
        np.save(os.path.join(output_dir, func[56:-7] + "_masked"), X)
    else:
        raise NameError('Input parent directory not known!')


def copy_files(img_paths, destination_folder):
    for subject_array in img_paths:
        for task_array in subject_array:
            for run_array in task_array:
                copy2(run_array, destination_folder)


def surf_concatenate(subject, tasks_list, inputs_path, outputs_path):
    for tsk in tasks_list:
        lh_files = 'rdcsub-%02d_ses-*_task-%s_dir-*_run-*' % (subject, tsk) + \
                   '_bold_fsaverage_lh.gii'
        lh_path = os.path.join(inputs_path, lh_files)
        lh = glob.glob(lh_path)
        for r in np.arange(1, len(lh) + 1):
            surf_name = 'rdcsub-%02d_ses-*_task-%s_dir-*' % (subject, tsk) + \
                        '_run-%02d_bold_fsaverage_*.gii' % (r)
            surf_path = os.path.join(inputs_path, surf_name)
            surf = glob.glob(surf_path)
            surf_lh = load_surf_data(surf[0])
            surf_rh = load_surf_data(surf[1])
            surf_stacked = np.vstack((surf_lh, surf_rh))
            ses = int(re.match('.*ses-(..)_task-*',
                               surf[0]).groups()[0])
            acq = re.match('.*dir-(..)_run-*', surf[0]).groups()[0]
            surfstack_name = 'rdcsub-%02d_ses-%02d_' % (subject, ses) + \
                             'task-%s_dir-%s' % (tsk, acq) + \
                             '_run-%02d_bold_fsaverage' % (r)
            np.save(os.path.join(outputs_path, surfstack_name), surf_stacked)


def surf_chancelevel(subject, task, ipath, opath):
    srlist = glob.glob(os.path.join(ipath, 'rdcsub-%02d' % subject +
                                    '_ses-*_task-%s' % task + '*.npy'))
    sruns = [np.load(sr).tolist() for sr in srlist]
    sruns = np.concatenate(sruns, axis=1)
    sruns = np.split(sruns, 13, axis=1)
    sruns = np.array(sruns)
    for srun, spath in zip(sruns, srlist):
        sname = re.match(ipath + '/(.*)', spath).groups()[0]
        np.save(os.path.join(opath, sname), srun)

# ######################################################################

# Extract data from data-frame only referring to a pre-specified subset of
# participants and tasks
participants = [1, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15]
naturalistic_tasks = ['ClipsTrn', 'ClipsVal', 'Raiders']

# Define pathways
main_parent_dir = '/neurospin/tmp/agrilopi'
alt_parent_dir = '/storage/store/work/agrilopi/'
main_dir = 'fastsrm/'

if os.path.exists(main_parent_dir):
    mem = main_parent_dir
    cache = main_parent_dir + main_dir
    preproc_paths = '/neurospin/ibc/derivatives'
else:
    mem = alt_parent_dir
    cache = alt_parent_dir + main_dir
    preproc_paths = '/storage/store2/data/ibc/derivatives'

# Define subjects' paths
sub_path = [os.path.join(preproc_paths, 'sub-%02d' % s)
            for s in participants]
SBJS = [os.path.basename(full_path) for full_path in sub_path]

# ###### INPUTS #####1#
# Mask of the grey matter of the IBC participants
_package_directory = os.path.dirname(
    os.path.abspath(ibc_public.utils_data.__file__))
mask_gm = os.path.join(_package_directory,
                       '../ibc_data', 'gm_mask_1_5mm.nii.gz')

# Input fMRI preprocessed data
db = data_parser(subject_list=SBJS, task_list=naturalistic_tasks)
preproc_imgs = db[db.contrast == 'preprocessed'].path.tolist()

# ###### OUTPUT FOLDER ######
volume_preprocdata = cache + 'volume_preprocdata'
surface_preprocdata = cache + 'surface_preprocdata'
surface_preprocdata_concatenated = cache + 'surface_preprocdata_concatenated'
surface_chancelevel = cache + 'surface_preprocraiders_chancelevel'

if not os.path.exists(volume_preprocdata):
    os.makedirs(volume_preprocdata)

if not os.path.exists(surface_preprocdata):
    os.makedirs(surface_preprocdata)

if not os.path.exists(surface_preprocdata_concatenated):
    os.makedirs(surface_preprocdata_concatenated)

# ############################### RUN ##################################

if __name__ == "__main__":

    # # ####### COMPUTE THE IBC MASKED-PREPROCESSED VOLUME DATA ##########

    # t0 = time()
    # Parallel(n_jobs=3, verbose=10)(
    #     delayed(separate_and_mask)(func, mask_gm, mem, volume_preprocdata)
    #     for func in preproc_imgs)

    # print("Done in %.2f seconds" % (time() - t0))

    # # ################ CONCATENATE IBC SURFACE DATA ####################

    # # Copy surface data of naturalistic tasks from paths set in
    # # the IBC data parser to a separate folder
    # surfimg_paths = reshape_preprocdata(participants, naturalistic_tasks,
    #                                     DERIVATIVES, input_type='surf')
    # copy_files(surfimg_paths, surface_preprocdata)

    # # Concatenate data from lh and rh and store in a different folder
    # Parallel(n_jobs=3, verbose=10)(
    #     delayed(surf_concatenate)(participant, naturalistic_tasks,
    #                               surface_preprocdata,
    #                               surface_preprocdata_concatenated)
    #     for participant in participants)

    # ###### CREATE RUNS OF SAME LENGTH TO COMPUTE CHANCE LEVEL ########

    Parallel(n_jobs=3, verbose=10)(
        delayed(surf_chancelevel)(participant, naturalistic_tasks[2],
                                  surface_preprocdata_concatenated,
                                  surface_chancelevel)
        for participant in participants)
