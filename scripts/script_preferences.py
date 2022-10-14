"""
This script adds a fixed-level analysis to the preference protocol
Author: Bertrand Thirion, 2018
"""
import os
import nibabel as nib
import numpy as np
from ibc_public.utils_pipeline import fixed_effects_img, fixed_effects_surf
from pipeline import get_subject_session
from nilearn.plotting import plot_stat_map
from nilearn.image import math_img


# where to work and write
SMOOTH_DERIVATIVES = '/neurospin/ibc/smooth_derivatives'
DERIVATIVES = '/neurospin/ibc/derivatives'
THREE_MM = '/neurospin/ibc/3mm'

# misc info on preference protocol
subject_session = get_subject_session('preference')
categories = ['Faces', 'Food', 'Houses', 'Paintings']
categories_ = categories
contrasts = ['constant', 'linear', 'quadratic']

# mask image
_package_directory = os.path.dirname(os.path.abspath(__file__))
mask_img = os.path.join(
    _package_directory, '../ibc_data', 'gm_mask_1_5mm.nii.gz')


def compute_contrast(con_imgs, var_imgs, mask_img):
    if isinstance(mask_img, str):
        mask_img = nib.load(mask_img)

    mask = mask_img.get_fdata().astype(np.bool)
    con, var = [], []
    for (con_img, var_img) in zip(con_imgs, var_imgs):
        if isinstance(con_img, str):
            con_img = nib.load(con_img)
        if isinstance(var_img, str):
            var_img = nib.load(var_img)
        con.append(con_img.get_fdata()[mask])
        var.append(var_img.get_fdata()[mask])

    fixed_con = np.array(con).sum(0)
    fixed_var = np.array(var).sum(0)
    stat = fixed_con / np.sqrt(fixed_var)
    outputs = []
    for array in [fixed_con, fixed_var, stat]:
        vol = mask.astype(np.float)
        vol[mask] = array.ravel()
        outputs.append(nib.Nifti1Image(vol, mask_img.affine))
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


def elementary_contrasts_surf(con_imgs, var_imgs):
    """ """
    from nibabel.gifti import GiftiDataArray, GiftiImage
    outputs = []
    n_contrasts = 4
    for i in range(n_contrasts):
        con = nib.load(con_imgs[i]).darrays[0].data
        var = nib.load(var_imgs[i]).darrays[0].data
        effects = [con - nib.load(con_imgs[j]).darrays[0].data
                   for j in range(n_contrasts) if j != i]
        variance = [var + nib.load(var_imgs[j]).darrays[0].data
                    for j in range(n_contrasts) if j != i]

        fixed_con = np.array(effects).sum(0)
        fixed_var = np.array(variance).sum(0)
        stat = fixed_con / np.sqrt(fixed_var)
        output = []
        intents = ['NIFTI_INTENT_ESTIMATE', 'NIFTI_INTENT_ESTIMATE', 't test']
        arrays = [fixed_con, fixed_var, stat]
        for array, intent in zip(arrays, intents):
            gii = GiftiImage(
                darrays=[GiftiDataArray().from_array(array, intent)])
            output.append(gii)
        outputs.append(output)
    return(outputs)


# in-volume computation
workdir = THREE_MM
if workdir == THREE_MM:
    mask_img = os.path.join(
        _package_directory, '../ibc_data', 'gm_mask_3mm.nii.gz')
for (subject, session) in subject_session:
    print(subject, session)
    anat = os.path.join(DERIVATIVES, subject, 'ses-00', 'anat',
                        'w%s_ses-00_T1w.nii.gz' % subject)
    write_dir = os.path.join(workdir, subject, session,
                             'res_stats_Preference_ffx')
    effect_dir = os.path.join(write_dir, 'effect_size_maps')
    variance_dir = os.path.join(write_dir, 'effect_variance_maps')
    stat_dir = os.path.join(write_dir, 'stat_maps')
    dirs = [write_dir, effect_dir, variance_dir, stat_dir]
    for dir_ in dirs:
        if not os.path.exists(dir_):
            os.mkdir(dir_)
    for contrast in contrasts:
        effects = [os.path.join(
            workdir, subject, session,
            'res_stats_Preference%s_ffx' % category_, 'effect_size_maps',
            '%s_%s.nii.gz' % (category, contrast))
                    for category, category_ in zip(categories, categories_)]
        variance = [os.path.join(
            workdir, subject, session,
            'res_stats_Preference%s_ffx' % category_, 'effect_variance_maps',
            '%s_%s.nii.gz' % (category, contrast))
                    for category, category_ in zip(categories, categories_)]
        fixed_effect, fixed_variance, fixed_stat = fixed_effects_img(
            effects, variance, mask_img)

        fixed_effect.to_filename(
            os.path.join(effect_dir, 'Preference%s.nii.gz' % contrast))
        fixed_variance.to_filename(
            os.path.join(variance_dir, 'Preference%s.nii.gz' % contrast))
        fixed_stat.to_filename(
            os.path.join(stat_dir, 'Preference%s.nii.gz' % contrast))

        output_file = os.path.join(stat_dir, 'Preference%s.png' % contrast)
        plot_stat_map(fixed_stat, bg_img=anat, dim=0,
                      output_file=output_file, threshold=4.0)

    # Compare categories
    contrast = 'constant'
    effects = [os.path.join(
        workdir, subject, session,
        'res_stats_Preference%s_ffx' % category_, 'effect_size_maps',
        '%s_%s.nii.gz' % (category, contrast))
                for category, category_ in zip(categories, categories_)]
    variance = [os.path.join(
        workdir, subject, session,
        'res_stats_Preference%s_ffx' % category_, 'effect_variance_maps',
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


# on-surface computation
workdir = DERIVATIVES
for (subject, session) in subject_session:
    print(subject, session)
    for mesh in ['fsaverage5', 'fsaverage7', 'individual']:
        for hemi in ['lh', 'rh']:
            write_dir = os.path.join(workdir, subject, session,
                                     'res_%s_Preference_ffx' % mesh)
            effect_dir = os.path.join(write_dir, 'effect_size_maps')
            variance_dir = os.path.join(write_dir, 'effect_variance_maps')
            stat_dir = os.path.join(write_dir, 'stat_maps')
            dirs = [write_dir, effect_dir, variance_dir, stat_dir]
            for dir_ in dirs:
                if not os.path.exists(dir_):
                    os.mkdir(dir_)
            for contrast in contrasts:
                effects = [os.path.join(
                    workdir, subject, session,
                    'res_%s_Preference%s_ffx' % (mesh, category_), 'effect_size_maps',
                    '%s_%s_%s.gii' % (category, contrast, hemi))
                            for category, category_ in zip(categories, categories_)]
                variance = [os.path.join(
                    workdir, subject, session,
                    'res_%s_Preference%s_ffx' % (mesh, category_), 'effect_variance_maps',
                    '%s_%s_%s.gii' % (category, contrast, hemi))
                            for category, category_ in zip(categories, categories_)]
                fixed_effect, fixed_variance, fixed_stat = fixed_effects_surf(
                    effects, variance)
                fixed_effect.to_filename(
                    os.path.join(effect_dir, 'Preference%s_%s.gii' % (contrast, hemi)))
                fixed_variance.to_filename(
                    os.path.join(variance_dir, 'Preference%s_%s.gii' % (contrast, hemi)))
                fixed_stat.to_filename(
                    os.path.join(stat_dir, 'Preference%s_%s.gii' % (contrast, hemi)))

            # Compare categories
            contrast = 'constant'
            effects = [os.path.join(
                workdir, subject, session,
                'res_%s_Preference%s_ffx' % (mesh, category_), 'effect_size_maps',
                '%s_%s_%s.gii' % (category, contrast, hemi))
                        for category, category_ in zip(categories, categories_)]
            variance = [os.path.join(
                workdir, subject, session,
                'res_%s_Preference%s_ffx' % (mesh, category_), 'effect_variance_maps',
                '%s_%s_%s.gii' % (category, contrast, hemi))
                        for category, category_ in zip(categories, categories_)]
            outputs = elementary_contrasts_surf(effects, variance)

            for output, category in zip(outputs, categories):
                fixed_effect, fixed_variance, fixed_stat = output
                fixed_effect.to_filename(
                    os.path.join(effect_dir, '%s-others_%s.gii' % (category, hemi)))
                fixed_variance.to_filename(
                    os.path.join(variance_dir, '%s-others_%s.gii' % (category, hemi)))
                fixed_stat.to_filename(
                    os.path.join(stat_dir, '%s-others_%s.gii' % (category, hemi)))
