"""
Encoding of the IBC data from the Clips and Raiders tasks
using the fast-shared response model (FastSRM)

Authors: Ana Luisa Pinho, Hugo Richard, Bertrand Thirion

Created: October 2020
Last update: October 2022

Compatibility: Python 3.9.1
"""

import os

from time import time
from tqdm import tqdm

import numpy as np
import pandas as pd

from scipy import stats

from nilearn.input_data import NiftiMasker
from nilearn.datasets import fetch_surf_fsaverage
from nilearn.plotting import plot_stat_map, plot_surf_stat_map
from nilearn.mass_univariate import permuted_ols

import ibc_public.utils_data

from ibc_fastsrm_utils import flatten, reshape_preprocdata, stacker

from fastsrm.identifiable_srm import IdentifiableFastSRM
from sklearn.model_selection import KFold


# ########################## FUNCTIONS #################################


def split_session(img_paths, split_idx=6):
    splitted = []
    for split in img_paths:
        first_half = split[0][:split_idx]
        second_half = split[0][split_idx:]
        subject = [first_half, second_half]
        splitted.append(subject)
    return splitted


def fsrm_encoding(preprocdata, compcache, n_iterations=10000, chance=False):

    print("Running reconstruction experiment")
    t0 = time()

    # ################# DEFINE THE FASTSRM MODEL #######################

    fastsrm = IdentifiableFastSRM(
        n_jobs=4,
        n_iter=1,
        tol=1e-7,
        n_components=20,
        n_iter_reduced=n_iterations,
        identifiability="decorr",
        verbose=False,
        temp_dir=compcache,
        aggregate="mean"  # transform will return the mean of subject specific
                          # shared response
    )

    # ################# DEFINE SOME EXTRA-INPUTS #######################

    # List in which we record for each CV iteration
    # the test R2 scores per voxels across subjects
    if TASKS == ['ClipsTrn', 'ClipsVal', 'Raiders']:
        task_nsplits = 3
        sarray = TASKS
    else:
        task_nsplits = 2
        sarray = preprocdata[0]

    n_voxels = np.load(preprocdata[0][0][0]).shape[0]
    r2_subjects = np.empty((len(participants), task_nsplits, n_voxels))
    corr_subjects = np.empty((len(participants), task_nsplits, n_voxels))
    pval_subjects = np.empty((len(participants), task_nsplits, n_voxels))

    # ########################### CV ###################################

    # Divide all subjects into train subjects and test subjects
    # CV scheme: leave-4-subjects-out
    for subjects_train, subjects_test in tqdm(
            KFold(n_splits=3, shuffle=False).split(
                np.arange(len(participants)))):
        # Divide tasks into train tasks and test tasks
        # CV scheme: split-half
        for tasks_train, tasks_test in tqdm(
                KFold(n_splits=task_nsplits, shuffle=True).split(np.arange(len(
                    sarray)))):

            tr_sub_tr_task_paths = stacker(
                subjects_train, tasks_train, preprocdata)
            tr_sub_ts_task_paths = stacker(
                subjects_train, tasks_test, preprocdata)
            ts_sub_tr_task_paths = stacker(
                subjects_test, tasks_train, preprocdata)
            ts_sub_ts_task_paths = stacker(
                subjects_test, tasks_test, preprocdata)

            n_subjects_train = len(subjects_train)
            n_subjects_test = len(subjects_test)

            if chance:
                for sub_n in np.arange(n_subjects_train):
                    np.random.shuffle(tr_sub_tr_task_paths[sub_n, :])

            # Fit the model on the train subjects in the train task runs
            fastsrm.fit(tr_sub_tr_task_paths)
            # Compute a shared response from the train subjects
            # in the train task runs
            shared_train = fastsrm.transform(tr_sub_tr_task_paths)
            # Compute a shared response from the train subjects
            # in the test task run
            shared_test = fastsrm.transform(tr_sub_ts_task_paths)
            # Add test subjects to the model
            fastsrm.add_subjects(ts_sub_tr_task_paths, shared_train)
            # Reconstruct the data of the test subjects during test task
            # run
            reconstructed_data_test_set = fastsrm.inverse_transform(
                shared_test, subjects_indexes=np.arange(
                    n_subjects_train, n_subjects_train + n_subjects_test))

            # Release memory
            del tr_sub_tr_task_paths
            del tr_sub_ts_task_paths
            del ts_sub_tr_task_paths
            del shared_train
            del shared_test

            # ## Diff/Corr between the reconstructed data and the
            # ## original data
            # List in which we record for each subject
            # the test R2 scores per voxels
            for sub_idx, subject_test in enumerate(subjects_test):
                r2_runs = []
                corr_runs = []
                pval_runs = []
                for run_idx in np.arange(len(ts_sub_ts_task_paths[sub_idx])):
                    # Load the original test data
                    run_img = np.load(ts_sub_ts_task_paths[sub_idx][run_idx])
                    n_voxels = run_img.shape[0]
                    # Compute the diff for each run at every voxel
                    diff = np.array(
                        reconstructed_data_test_set[sub_idx][run_idx] -
                        run_img)
                    # Compute the R-squared **per voxel** and append results
                    # for each run and subject
                    r2 = 1 - diff.var(axis=1)
                    r2_runs.append(r2)
                    # Release memory
                    del diff
                    del r2
                    # Compute Pearson correlation and corresponding two-tailed
                    # p-value and append results for each run and subject
                    corr = np.array([
                        stats.pearsonr(
                            reconstructed_data_test_set[
                                sub_idx][run_idx][voxel], run_img[voxel])[0]
                        for voxel in np.arange(n_voxels)])
                    pval = np.array([
                        stats.pearsonr(
                            reconstructed_data_test_set[
                                sub_idx][run_idx][voxel], run_img[voxel])[1]
                        for voxel in np.arange(n_voxels)])
                    corr_runs.append(corr)
                    pval_runs.append(pval)
                    # Release memory
                    del run_img
                    del corr
                    del pval
                # Compute the R-squared median as well as
                # Pearson-correlation median plus its aggregated p-value for
                # all runs per subject **between homologous voxels**
                r2_subject = np.median(r2_runs, axis=0)
                corr_subject = np.median(corr_runs, axis=0)
                pval_runs_swapped = np.swapaxes(pval_runs, 0, 1)
                pval_subject = np.array([stats.combine_pvalues(
                    pvals, method='fisher')[1] for pvals in pval_runs_swapped])
                # Release memory
                del r2_runs
                del corr_runs
                del pval_runs
                del pval_runs_swapped
                # Place results in numpy array for all subjects and cvs
                # shape: (subjects, cv, voxels)
                r2_subjects[subject_test][tasks_test[0]] = r2_subject
                corr_subjects[subject_test][tasks_test[0]] = corr_subject
                pval_subjects[subject_test][tasks_test[0]] = pval_subject
                # Release memory
                del r2_subject
                del corr_subject
                del pval_subject
            # Release memory
            del ts_sub_ts_task_paths
            del reconstructed_data_test_set
    print("Done in %.2f" % (time() - t0))

    return r2_subjects, corr_subjects, pval_subjects


def plotting(gmatter_mask, results, eval_metric, final_fig, axis):
    masker = NiftiMasker(mask_img=gmatter_mask).fit()
    for name in ["FastSRM"]:
        plot_stat_map(masker.inverse_transform(results),
                      display_mode=axis,
                      title='%s ' % name + eval_metric,
                      output_file=final_fig)


def threshold(correlations, p_vals, alpha, height_control='fdr'):
    """Return the Benjamini-Hochberg FDR or Bonferroni threshold for
    the input correlations + corresponding p-values.
    """
    if alpha < 0 or alpha > 1:
        raise ValueError(
            'alpha should be between 0 and 1. {} was provided'.format(alpha))

    p_vals_ = np.sort(p_vals)
    idx = np.argsort(p_vals)
    correlations_ = correlations[idx]
    n_samples = len(p_vals_)

    if height_control == 'fdr':
        pos = p_vals_ < alpha * np.linspace(1 / n_samples, 1, n_samples)
    elif height_control == 'bonferroni':
        pos = p_vals_ < alpha / n_samples
    else:
        raise ValueError('Height-control method not valid.')

    return (correlations_[pos][-1] - 1.e-12) if pos.any() else np.infty


# ######################## GENERAL INPUTS ##############################

# List of participants' numbers
participants = [1, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15]
# Define subjects' list
SUBJECTS = ['sub-%02d' % s for s in participants]

# TASKS = ['ClipsTrn', 'ClipsVal', 'Raiders']
TASKS = ['ClipsTrn', 'ClipsVal']
# TASKS = ['Raiders']

main_parent_dir = '/neurospin/tmp/agrilopi'
alt_parent_dir = '/storage/store/work/agrilopi'
fastsrm_dir = 'fastsrm'
main_dir = 'encoding_analysis'

if TASKS == ['ClipsTrn', 'ClipsVal', 'Raiders']:
    # suffix = 'all'
    suffix = 'all_chance'
    r2_title = 'R-squared'
    corr_title = 'Pearson correlations'
elif TASKS == ['ClipsTrn', 'ClipsVal']:
    suffix = 'clips'
    # suffix = 'clips_chance'
    r2_title = 'R-squared for Clips'
    corr_title = 'Pearson correlations for Clips'
else:
    assert TASKS == ['Raiders']
    suffix = 'raiders'
    # suffix = 'raiders_chance'
    r2_title = 'R-squared for Raiders'
    corr_title = 'Pearson correlations for Raiders'

if os.path.exists(main_parent_dir):
    mem = main_parent_dir
    cache = os.path.join(main_parent_dir, fastsrm_dir, main_dir)
else:
    mem = alt_parent_dir
    cache = os.path.join(alt_parent_dir, fastsrm_dir, main_dir)

# Mask of the grey matter of the IBC participants
_package_directory = os.path.dirname(
    os.path.abspath(ibc_public.utils_data.__file__))
mask_gm = os.path.join(_package_directory,
                       '../ibc_data', 'gm_mask_1_5mm.nii.gz')

# FastSRM cache
fs_cache = '/storage/store/work/agrilopi/fastsrm/encoding_analysis/cache'

#  Loads the high-resolution fsaverage mesh (163842 nodes)
fsaverage = fetch_surf_fsaverage(mesh='fsaverage')

# Main directory containing the masked, preprocessed data
volume_preprocdata = os.path.join(
    alt_parent_dir, fastsrm_dir, 'volume_preprocdata')
surface_preprocdata_concatenated = os.path.join(
    alt_parent_dir, fastsrm_dir, 'surface_preprocdata_concatenated')
surface_preprocraiders_chancelevel = os.path.join(
    alt_parent_dir, fastsrm_dir, 'surface_preprocraiders_chancelevel')


# ################################## RUN ###############################

if __name__ == "__main__":

    # ########################### DATA PARSER ##########################

    # volimg_paths = reshape_preprocdata(participants, TASKS,
    #                                    volume_preprocdata)

    # ***********

    surfimg_paths = reshape_preprocdata(participants, TASKS,
                                        surface_preprocdata_concatenated,
                                        input_type='surf')

    # To compute the chance level for RAIDERS task on surface data, ...
    # ...run this command line instead
    # surfimg_paths = reshape_preprocdata(participants, ['Raiders'],
    #                                     surface_preprocraiders_chancelevel,
    #                                     input_type='surf')

    if TASKS == ['Raiders']:
        # volimg_paths = split_session(volimg_paths, split_idx=6)
        surfimg_paths = split_session(surfimg_paths, split_idx=6)


    # ################### FASTSRM ENCODING EXPERIMENT ##################

    # # _, corr_sbjs, pval_sbjs = fsrm_encoding(volimg_paths, fs_cache)

    # Main results
    _, corr_sbjs, _ = fsrm_encoding(surfimg_paths, fs_cache)
    # Chance level
    # _, corr_sbjs, pval_sbjs = fsrm_encoding(surfimg_paths, fs_cache,
    #                                         chance= True)

    # Remove NaN values from Clips data
    if TASKS[0] in ['ClipsTrn', 'ClipsVal']:
        corr_sbjs = np.where(np.isnan(corr_sbjs), 0., corr_sbjs)
        # pval_sbjs = np.where(np.isnan(pval_sbjs), 0.5, pval_sbjs)

    # Median per subject
    individual_corr = np.median(corr_sbjs, axis=1)

    # Compute group-level results
    ols_outputs = permuted_ols(
        np.ones(len(individual_corr)),
        individual_corr,
        model_intercept=False,
        tfce=False,
        n_perm=10000,
        verbose=1,
        n_jobs=3,
        output_type='dict',
    )

    #  ######################## STORE RESULTS ##########################

    # *************************** Surface ******************************

    np.save(
        os.path.join(cache, suffix, 'surface_corr' + '_' + suffix), corr_sbjs)

    np.save(
        os.path.join(cache, suffix, 'surface_individual_corr' + '_' + suffix),
        individual_corr)

    np.savez(os.path.join(cache, suffix, 'surface_group_corr' + '_' + suffix),
             t=ols_outputs['t'], logp_max_t=ols_outputs['logp_max_t'],
             h0_max_t=ols_outputs['h0_max_t'])


    # ########################### PLOTTING #############################

    # ******************************************************************
    # **************************** Surface *****************************
    # ******************************************************************

    # ++++++++++++++++++++++++ Correlation +++++++++++++++++++++++++++++

    # # Load file
    # ols_outputs = \
    #     np.load(os.path.join(
    #         cache, suffix, 'surface_group_corr' + '_' + suffix + '.npz'),
    #             allow_pickle=True)

    # Path of output files with thresholded results
    corrfig_lhl = os.path.join(
        cache, suffix,
        'surface_groupcorr' + '_lh_lateral_' + suffix + '.png')
    corrfig_lhm = os.path.join(
        cache, suffix,
        'surface_groupcorr' + '_lh_medial_' + suffix + '.png')
    corrfig_rhl = os.path.join(
        cache, suffix,
        'surface_groupcorr' + '_rh_lateral_' + suffix + '.png')
    corrfig_rhm = os.path.join(
        cache, suffix,
        'surface_groupcorr' + '_rh_medial_' + suffix + '.png')

    # Split results into the two hemispheres
    pmaxt = np.reshape(ols_outputs['logp_max_t'],
                       (ols_outputs['logp_max_t'].shape[1],))
    clipped_gcorr_lh = np.split(pmaxt, 2, axis=0)[0]
    clipped_gcorr_rh = np.split(pmaxt, 2, axis=0)[1]

    # Plot thresholded group-level results
    threshold = - np.log10(.1)  # 10% corrected
    # vmax = np.amax(ols_outputs['logp_max_t'])
    vmax = 3.221892176893219

    plot_surf_stat_map(fsaverage.infl_left, clipped_gcorr_lh, hemi='left',
                       view='lateral', colorbar=True, vmax=vmax,
                       bg_map=fsaverage.sulc_left, threshold=threshold,
                       output_file=corrfig_lhl)
    plot_surf_stat_map(fsaverage.infl_left, clipped_gcorr_lh, hemi='left',
                       view='medial', colorbar=True, vmax=vmax,
                       bg_map=fsaverage.sulc_left, threshold=threshold,
                       output_file=corrfig_lhm)
    plot_surf_stat_map(fsaverage.infl_right, clipped_gcorr_rh, hemi='right',
                       view='lateral', colorbar=True, vmax=vmax,
                       bg_map=fsaverage.sulc_right, threshold=threshold,
                       output_file=corrfig_rhl)
    plot_surf_stat_map(fsaverage.infl_right, clipped_gcorr_rh, hemi='right',
                       view='medial', colorbar=True, vmax=vmax,
                       bg_map=fsaverage.sulc_right, threshold=threshold,
                       output_file=corrfig_rhm)
