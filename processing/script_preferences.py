"""
This script adds a fixed-level analysis to the preference protocol
Author: Bertrand Thirion, 2018
"""
import os
from utils_pipeline import fixed_effects_img
from pipeline import get_subject_session
from nilearn.plotting import plot_stat_map
from nilearn.image import math_img


subject_session = get_subject_session('preferences')
workdir = '/neurospin/ibc/derivatives'
categories = ['face', 'food', 'house', 'painting']
categories_ = ['faces', 'food', 'houses', 'paintings']

contrasts = ['constant', 'linear', 'quadratic']
mask_img = '/neurospin/ibc/smooth_derivatives/group/resampled_gm_mask.nii.gz'


def compute_contrast(con_imgs, var_imgs, mask_img):
    import nibabel as nib
    import numpy as np
    if isinstance(mask_img, basestring):
        mask_img = nib.load(mask_img)

    mask = mask_img.get_data().astype(np.bool)
    con, var = [], []
    for (con_img, var_img) in zip(con_imgs, var_imgs):
        if isinstance(con_img, basestring):
            con_img = nib.load(con_img)
        if isinstance(var_img, basestring):
            var_img = nib.load(var_img)
        con.append(con_img.get_data()[mask])
        var.append(var_img.get_data()[mask])
        
    fixed_con = np.array(con).sum(0)
    fixed_var = np.array(var).sum(0)
    stat = fixed_con / np.sqrt(fixed_var)
    outputs = []
    for array in [fixed_con, fixed_var, stat]:
        vol = mask.astype(np.float)
        vol[mask] = array.ravel()
        outputs.append(nib.Nifti1Image(vol, mask_img.get_affine()))
    return outputs


def elementary_contrasts(con_imgs, var_imgs, mask_img):
    """ """
    outputs = []
    n_contrasts = 4
    for i in range(n_contrasts):
        effects = [math_img('-(1. / 3) * i1', i1=con_img)
                   for con_img in con_imgs]
        effects[i] = con_imgs[i]
        variance = [math_img('(1. / 9) * i1', i1=var_img)
                   for var_img in var_imgs]
        variance[i] = var_imgs[i]
        
        output = compute_contrast(
            effects, variance, mask_img)
        outputs.append(output)
    return(outputs)


for (subject, session) in subject_session:
    print(subject, session)
    anat = os.path.join(workdir, subject, 'ses-00', 'anat',
                        'w%s_ses-00_T1w.nii.gz' % subject)
    write_dir = os.path.join(workdir, subject, session,
                             'res_stats_preference_ffx')        
    effect_dir = os.path.join(write_dir, 'effect_size_maps')
    variance_dir = os.path.join(write_dir, 'effect_variance_maps')
    stat_dir = os.path.join(write_dir, 'stat_maps')
    dirs = [write_dir, effect_dir, variance_dir, stat_dir]
    for dir_ in dirs:
        if not os.path.exists(dir_):
            os.mkdir(dir_)

    for contrast in contrasts:
        effects =  [os.path.join(
            workdir, subject, session,
            'res_stats_preference_%s_ffx' % category_, 'effect_size_maps',
            '%s_%s.nii.gz' % (category, contrast))
                    for category, category_ in zip(categories, categories_)]
        variance = [os.path.join(
            workdir, subject, session,
            'res_stats_preference_%s_ffx' % category_, 'effect_variance_maps',
            '%s_%s.nii.gz' % (category, contrast))
                    for category, category_ in zip(categories, categories_)]
        fixed_effect, fixed_variance, fixed_stat = fixed_effects_img(
            effects, variance, mask_img)

        fixed_effect.to_filename(
            os.path.join(effect_dir, 'preference_%s.nii.gz' % contrast))
        fixed_variance.to_filename(
            os.path.join(variance_dir, 'preference_%s.nii.gz' % contrast))
        fixed_stat.to_filename(
            os.path.join(stat_dir, 'preference_%s.nii.gz' % contrast))

        output_file = os.path.join(stat_dir, 'preference_%s.png' % contrast)
        plot_stat_map(fixed_stat, bg_img=anat, dim=0,
                      output_file=output_file, threshold=4.0)

    # Compare categories
    contrast = 'constant'
    effects =  [os.path.join(
        workdir, subject, session,
        'res_stats_preference_%s_ffx' % category_, 'effect_size_maps',
        '%s_%s.nii.gz' % (category, contrast))
                for category, category_ in zip(categories, categories_)]
    variance = [os.path.join(
        workdir, subject, session,
        'res_stats_preference_%s_ffx' % category_, 'effect_variance_maps',
        '%s_%s.nii.gz' % (category, contrast))
                for category, category_ in zip(categories, categories_)]
    outputs = elementary_contrasts(
        effects, variance, mask_img)
    for output, category in zip(outputs, categories):
        fixed_effect, fixed_variance, fixed_stat = output
        fixed_effect.to_filename(
            os.path.join(effect_dir, '%s-others.nii.gz' % category))
        fixed_variance.to_filename(
            os.path.join(variance_dir, '%s-others.nii.gz' % category))
        fixed_stat.to_filename(
            os.path.join(stat_dir, '%s-others.nii.gz' % category))
        
        output_file = os.path.join(stat_dir, '%s-others.png' % category)
        plot_stat_map(fixed_stat, bg_img=anat, dim=0,
                      output_file=output_file, threshold=4.0)
