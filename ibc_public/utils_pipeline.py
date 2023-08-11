"""
This module contains dataset-independent utilities
to run fMRI data analyses.

Author: Bertrand Thirion, 2015
"""
import os
import time
import numpy as np
import nibabel as nib
import warnings
from pandas import read_csv
from nilearn.masking import compute_multi_epi_mask
from nilearn.image import high_variance_confounds

from nilearn.glm.first_level.design_matrix import (
    make_first_level_design_matrix,
    check_design_matrix)

# from pypreprocess.reporting.base_reporter import ProgressReport
# from pypreprocess.reporting.glm_reporter import generate_subject_stats_report

from ibc_public.utils_contrasts import make_contrasts
from ibc_public.utils_paradigm import make_paradigm
from nilearn.reporting import make_glm_report


def _make_topup_param_file(field_maps, acq_params_file):
    """Create the param file based on the json info attached to fieldmaps"""
    import json
    jsons = [os.path.join(os.path.dirname(fm), '../../..',
                          os.path.basename(fm).split('_')[-2] + '_epi.json')
             for fm in field_maps]
    fw = open(acq_params_file, 'w')
    for json_ in jsons:
        info = json.load(open(json_, 'r'))
        if info['PhaseEncodingDirection'] == 'j':
            vals = '0.0 1.0 0.0 %f\n' % (info['TotalReadoutTime'] * 1000)
        elif info['PhaseEncodingDirection'] == 'j-':
            vals = '0.0 -1.0 0.0 %f\n' % (info['TotalReadoutTime'] * 1000)
        fw.write(vals)
    fw.close()


def _bids_filename_to_dic(filename):
    """Make a dictionary of properties from a bids filename"""
    parts = os.path.basename(filename).split('_')
    dic = {}
    for part in parts:
        if '-' in part:
            key, value = part.split('-')
            dic[key] = value
    return dic


def _make_merged_filename(fmap_dir, basenames):
    """Create filename for merged field_maps"""
    dic0 = _bids_filename_to_dic(basenames[0])
    dic1 = _bids_filename_to_dic(basenames[1])
    if 'sub' not in dic0.keys():
        dic0['sub'] = dic0['pilot']
    if 'sub' not in dic1.keys():
        dic0['sub'] = dic1['pilot']

    if 'acq' in dic0.keys():
        merged_basename = (
            'sub-' + dic0['sub'] + '_ses-' + dic0['ses'] + '_acq-' +
            dic0['acq'] + '_dir-' + dic0['dir'] + dic1['dir'] + '_epi.nii.gz')
    else:
        merged_basename = (
            'sub-' + dic0['sub'] + '_ses-' + dic0['ses'] + '_dir-' +
            dic0['dir'] + dic1['dir'] + '_epi.nii.gz')
    # merged_basename = basenames[0][:19] + basenames[1][18:]
    return(os.path.join(fmap_dir, merged_basename))


def fsl_topup(field_maps, fmri_files, mem, write_dir, modality='func'):
    """ This function calls topup to estimate distortions from field maps
    then apply the ensuing correction to fmri_files"""
    # merge the 0th volume of both fieldmaps
    fmap_dir = os.path.join(write_dir, 'fmap')
    basenames = [os.path.basename(fm) for fm in field_maps]
    merged_zeroth_fieldmap_file = _make_merged_filename(fmap_dir, basenames)
    zeroth_fieldmap_files = field_maps  # FIXME
    fslmerge_cmd = "fsl5.0-fslmerge -t %s %s %s" % (
        merged_zeroth_fieldmap_file, zeroth_fieldmap_files[0],
        zeroth_fieldmap_files[1])
    print("\r\nExecuting '%s' ..." % fslmerge_cmd)
    print(os.system(fslmerge_cmd))
    # add one slide if the number is odd
    odd = (np.mod(nib.load(merged_zeroth_fieldmap_file).shape[2], 2) == 1)
    if odd:
        cmd = "fsl5.0-fslroi %s /tmp/pe 0 -1 0 -1 0 1 0 -1" %\
              merged_zeroth_fieldmap_file
        print(cmd)
        os.system(cmd)
        cmd = "fsl5.0-fslmerge -z %s /tmp/pe %s" % (
            merged_zeroth_fieldmap_file, merged_zeroth_fieldmap_file)
        print(cmd)
        os.system(cmd)

    # TOPUP
    acq_params_file = os.path.join(fmap_dir, 'b0_acquisition_params_AP.txt')
    _make_topup_param_file(field_maps, acq_params_file)
    # import shutil
    # shutil.copy('b0_acquisition_params_AP.txt', acq_params_file)
    topup_results_basename = os.path.join(fmap_dir, 'topup_result')
    if os.path.exists(topup_results_basename):
        os.system('rm -f %s' % topup_results_basename)
    topup_cmd = (
        "/i2bm/local/fsl/bin/topup --imain=%s --datain=%s --config=b02b0.cnf "
        "--out=%s" % (merged_zeroth_fieldmap_file, acq_params_file,
                      topup_results_basename))
    print("\r\nExecuting '%s' ..." % topup_cmd)
    print(os.system(topup_cmd))
    # apply topup to images
    func_dir = os.path.join(write_dir, modality)
    for i, f in enumerate(fmri_files):
        dcf = os.path.join(func_dir, "dc" + os.path.basename(f))
        if '-ap' in os.path.basename(f):
            inindex = 1
        elif '-pa' in os.path.basename(f):
            inindex = 2
        else:
            inindex = 2

        applytopup_cmd = (
            "fsl5.0-applytopup --imain=%s --verbose --inindex=%s "
            "--topup=%s --out=%s --datain=%s --method=jac" % (
                f, inindex, topup_results_basename, dcf, acq_params_file))
        print("\r\nExecuting '%s' ..." % applytopup_cmd)
        print(os.system(applytopup_cmd))


def run_glm(dmtx, contrasts, fmri_data, mask_img, subject_dic,
            subject_session_output_dir, tr, slice_time_ref,
            smoothing_fwhm=False):
    """ Run the GLM on a given session and compute contrasts

    Parameters
    ----------
    dmtx : array-like
        the design matrix for the model
    contrasts : dict
        holding the numerical specification of contrasts
    fmri_data : Nifti1Image
        the fMRI data fir by the model
    mask_img : Nifti1Image
        the mask used for the fMRI data
    """
    from nilearn.glm.first_level import FirstLevelModel
    fmri_4d = nib.load(fmri_data)

    # GLM analysis
    print('Fitting a GLM (this takes time)...')
    fmri_glm = FirstLevelModel(mask_img=mask_img, t_r=tr,
                               slice_time_ref=slice_time_ref,
                               smoothing_fwhm=smoothing_fwhm).fit(
        fmri_4d, design_matrices=dmtx)

    # compute contrasts
    z_maps = {}
    for contrast_id, contrast_val in contrasts.items():
        print("\tcontrast id: %s" % contrast_id)

        # store stat maps to disk
        for map_type in ['z_score', 'stat', 'effect_size', 'effect_variance']:
            stat_map = fmri_glm.compute_contrast(
                contrast_val, output_type=map_type)
            map_dir = os.path.join(
                subject_session_output_dir, '%s_maps' % map_type)
            if not os.path.exists(map_dir):
                os.makedirs(map_dir)
            map_path = os.path.join(map_dir, '%s.nii.gz' % contrast_id)
            print("\t\tWriting %s ..." % map_path)
            stat_map.to_filename(map_path)

            # collect zmaps for contrasts we're interested in
            if map_type == 'z_score':
                z_maps[contrast_id] = map_path
    return z_maps, fmri_glm


def run_surface_glm(dmtx, contrasts, fmri_path, subject_session_output_dir):
    """ """
    from nibabel.gifti import read, write, GiftiDataArray, GiftiImage
    from nilearn.glm.first_level import run_glm as run_glm_nl
    from nilearn.glm import compute_contrast
    Y = np.array([darrays.data for darrays in read(fmri_path).darrays])
    labels, res = run_glm_nl(Y, dmtx.values)
    # Estimate the contrasts
    print('Computing contrasts...')
    side = fmri_path[-6:-4]
    for index, contrast_id in enumerate(contrasts):
        print('  Contrast % i out of %i: %s' %
              (index + 1, len(contrasts), contrast_id))
        # compute contrasts
        con_ = contrasts[contrast_id]
        contrast_ = compute_contrast(labels, res, con_)
        stats = [contrast_.z_score(), contrast_.stat_, contrast_.effect,
                 contrast_.variance]
        for map_type, out_map in zip(['z_score', 'stat', 'effect_size', 'effect_variance'], stats):
            map_dir = os.path.join(
                subject_session_output_dir, '%s_maps' % map_type)
            if not os.path.exists(map_dir):
                os.makedirs(map_dir)
            map_path = os.path.join(map_dir, '%s_%s.gii' % (contrast_id, side))
            print("\t\tWriting %s ..." % map_path)
            tex = GiftiImage(
                darrays=[GiftiDataArray().from_array(
                    out_map, intent='t test')])
            write(tex, map_path)


def masking(func, output_dir):
    """compute the mask for all sessions"""
    # compute the mask for all sessions
    # save computed mask
    mask_img = compute_multi_epi_mask(
        func, upper_cutoff=.7, lower_cutoff=.4)
    vox_vol = np.abs(np.linalg.det(mask_img.affine[:3, :3]))
    full_vol = mask_img.get_data().sum() * vox_vol
    ref_vol = 1350000
    if full_vol < ref_vol:
        raise ValueError("wrong mask: volume is %f, should be larger than %f" %
                         (full_vol, ref_vol))
    mask_path = os.path.join(output_dir, "mask.nii.gz")
    print("Saving mask image %s" % mask_path)
    mask_img.to_filename(mask_path)
    # todo: cache this then add masking in pypreprocess
    return mask_img


def _audio_frametimes_and_onset():
    mask = np.array([1, 0, 1, 1, 0, 1, 1, 0, 1, 1])
    n_cycles = 28
    cycle_duration = 20
    t_r = 2
    cycle = np.arange(0, cycle_duration, t_r)[mask > 0]
    frametimes = np.tile(cycle, n_cycles) +\
                 np.repeat(np.arange(n_cycles) * cycle_duration, mask.sum())
    frametimes = frametimes[:-2].astype(float)  # for some reason...
    cycle = np.arange(0, cycle_duration, t_r)[mask == 0]
    onsets = np.tile(cycle, n_cycles) + np.repeat(np.arange(n_cycles) * cycle_duration, len(cycle))
    onsets = onsets[:-2].astype(float)
    return frametimes, onsets


def _orthogonalize_confounds_wrt_task(confounds, frametimes, paradigm, hrf_model='spm'):
    """ These function orthogonalizes the provided confounds wrt the task effects.
    Should be used with compcorr regressor typically.
    """
    # First create a design matrix
    X = make_first_level_design_matrix(
        frametimes, paradigm, hrf_model, drift_model=None).values
    
    # then orthogonalise the confounds
    confounds -= X.dot(np.linalg.pinv(X).dot(confounds))

    assert np.linalg.norm(X.T.dot(confounds)) < (
        1.e-6 * np.linalg.norm(X) * np.linalg.norm(confounds))
    return confounds

    
def first_level(subject_dic, additional_regressors=None, compcorr=False,
                smooth=None, mesh=False, mask_img=None, orthogonalize_confounds=False):
    """ Run the first-level analysis (GLM fitting + statistical maps)
    in a given subject

    Parameters
    ----------
    subject_dic: dict,
                 exhaustive description of an individual acquisition
    additional_regressors: dict or None,
                 additional regressors provided as an already sampled
                 design_matrix
                 dictionary keys are session_ids
    compcorr: Bool, optional,
              whether confound estimation and removal should be done or not
    smooth: float or None, optional,
            how much the data should spatially smoothed during masking
    """
    start_time = time.ctime()
    # experimental paradigm meta-params
    motion_names = ['tx', 'ty', 'tz', 'rx', 'ry', 'rz']
    hrf_model = subject_dic['hrf_model']
    high_pass = subject_dic['high_pass']
    drift_model = subject_dic['drift_model']
    tr = subject_dic['TR']
    slice_time_ref = .5 ###
    
    if not mesh and (mask_img is None):
        mask_img = masking(subject_dic['func'], subject_dic['output_dir'])

    if additional_regressors is None:
        additional_regressors = dict(
            [(session_id, None) for session_id in subject_dic['session_id']])

    for session_id, fmri_path, onset, motion_path in zip(
            subject_dic['session_id'], subject_dic['func'],
            subject_dic['onset'], subject_dic['realignment_parameters']):

        task_id = _session_id_to_task_id([session_id])[0]

        if mesh is not False:
            from nibabel.gifti import read
            n_scans = np.array(
                [darrays.data for darrays in read(fmri_path).darrays]).shape[0]
        else:
            n_scans = nib.load(fmri_path).shape[3]

        # motion parameters
        motion = np.loadtxt(motion_path)
        # define the time stamps for different images
        frametimes = np.linspace(slice_time_ref, (n_scans - 1 + slice_time_ref) * tr, n_scans)
        if task_id == 'Audio':
            frametimes, audio_onsets = _audio_frametimes_and_onset()
            # frametimes += 1
            # slice_time_ref = 0
            
        if mesh is not False:
            compcorr = False  # XXX Fixme

        if onset is None:
            warnings.warn('Onset file not provided. Trying to guess it')
            task = os.path.basename(fmri_path).split('task')[-1][4:]
            onset = os.path.join(
                os.path.split(os.path.dirname(fmri_path))[0], 'model001',
                'onsets', 'task' + task + '_run001', 'task%s.csv' % task)

        if not os.path.exists(onset):
            warnings.warn('non-existant onset file. proceeding without it')
            paradigm = None
        else:
            paradigm = make_paradigm(onset, task_id)
            if task_id == 'Audio':
                paradigm.onset = audio_onsets

        if compcorr:
            confounds = high_variance_confounds(fmri_path, mask_img=mask_img)
            if orthogonalize_confounds:
                confounds = _orthogonalize_confounds_wrt_task(
                    confounds, frametimes, paradigm, hrf_model='spm')
            confounds = np.hstack((confounds, motion))
            confound_names = ['conf_%d' % i for i in range(5)] + motion_names
        else:
            confounds = motion
            confound_names = motion_names
        
        # handle manually supplied regressors
        add_reg_names = []
        if additional_regressors[session_id] is None:
            add_regs = confounds
        else:
            df = read_csv(additional_regressors[session_id])
            add_regs = []
            for regressor in df:
                add_reg_names.append(regressor)
                add_regs.append(df[regressor])
            add_regs = np.array(add_regs).T
            add_regs = np.hstack((add_regs, confounds))

        add_reg_names += confound_names
        
        # create the design matrix
        design_matrix = make_first_level_design_matrix(
            frametimes, paradigm, hrf_model=hrf_model, drift_model=drift_model,
            high_pass=high_pass, add_regs=add_regs,
            add_reg_names=add_reg_names)
        _, dmtx, names = check_design_matrix(design_matrix)

        # create the relevant contrasts
        contrasts = make_contrasts(task_id, names)

        subject_session_output_dir = _session_output_dir_from_session_id(
            session_id, mesh, subject_dic['output_dir'])
        
        if not os.path.exists(subject_session_output_dir):
            os.makedirs(subject_session_output_dir)
        np.savez(os.path.join(subject_session_output_dir, 'design_matrix.npz'),
                 design_matrix=design_matrix)

        if mesh is not False:
            run_surface_glm(
                design_matrix, contrasts, fmri_path, subject_session_output_dir)
        else:
            z_maps, fmri_glm = run_glm(
                design_matrix, contrasts, fmri_path, mask_img, subject_dic,
                subject_session_output_dir, tr=tr, slice_time_ref=slice_time_ref,
                smoothing_fwhm=smooth)

            # do stats report
            anat_img = nib.load(subject_dic['anat'])
            stats_report_filename = os.path.join(
                 subject_session_output_dir, 'report_stats.html')
            make_glm_report(fmri_glm,
                            contrasts=contrasts,
                            bg_img=anat_img,
                            title=session_id,
            ).save_as_html(stats_report_filename)


def _guess_run_and_dir_from_session_id(session_id):
            run_, dir_ = None, None
            parts = session_id.split('_')
            for part in parts:
                if part[:4] == 'run-':
                    run_ = part[4:]
                if part[:4] == 'dir-':
                    dir_ = part[4:]
            return run_, dir_


def _session_id_to_task_id(session_ids):
    """ Converts a session_id to a task _id
    by removing non-zero digits and _ap or _pa suffixes"""
    task_ids = []
    for i, session_id in enumerate(session_ids):
        task_id = session_id.split('_')[0]
        if task_id[:5] == 'task-':
            task_id = task_id[5:]
        task_ids.append(task_id)

    # customization for wedge and ring stimuli
    for i, task_id in enumerate(task_ids):
        if 'wedge' in task_id:
            task_ids[i] = 'wedge'
        if 'ring' in task_id:
            task_ids[i] = 'ring'
    return task_ids


def _session_output_dir_from_session_id(session_id, mesh, output_dir):
    """ Generate output_dir name from session_id.
    TODO: session -> run
    """
    run_, dir_ = _guess_run_and_dir_from_session_id(session_id)
    task_id = _session_id_to_task_id([session_id])[0]
    if mesh in ['fsaverage5', 'fsaverage7', 'individual']:
        # this is low-resolution data
        subject_session_output_dir = os.path.join(
            output_dir,
            'res_task-%s_space-%s' % (task_id, mesh))
    else:
        subject_session_output_dir = os.path.join(
            output_dir, 'res_task-%s_space-MNI305' % task_id)
    if run_:
        subject_session_output_dir += '_run-%s' % run_
    if dir_:
        subject_session_output_dir += '_dir-%s' % dir_
    return subject_session_output_dir


def _load_summary_stats(output_dir, sessions, contrast, data_available=True,
                        side=False, mesh=None):
    """ data fetcher for summary statistics"""
    effect_size_maps = []
    effect_variance_maps = []
    if side is False:
        for session_id in sessions:
            sess_dir = _session_output_dir_from_session_id(session_id, mesh, output_dir)
            # sess_dir = os.path.join(output_dir, 'res_%s' % session_id)
            if not os.path.exists(sess_dir):
                warnings.warn('Missing session dir, skipping')
                data_available = False
                continue
            effect_size_maps.append(
                os.path.join(sess_dir, 'effect_size_maps',
                             '%s.nii.gz' % contrast))
            effect_variance_maps.append(
                os.path.join(sess_dir, 'effect_variance_maps',
                             '%s.nii.gz' % contrast))
    else:
        for session_id in sessions:
            sess_dir = _session_output_dir_from_session_id(session_id, mesh, output_dir)
            #sess_dir = os.path.join(
            #    output_dir, 'res_%s_%s' % (mesh, session_id))
            if not os.path.exists(sess_dir):
                warnings.warn('Missing session dir, skipping')
                data_available = False
                continue
            effect_size_maps.append(
                os.path.join(sess_dir, 'effect_size_maps', '%s_%s.gii' %
                             (contrast, side)))
            effect_variance_maps.append(
                os.path.join(sess_dir, 'effect_variance_maps', '%s_%s.gii' %
                             (contrast, side)))
    return effect_size_maps, effect_variance_maps, data_available


def fixed_effects_analysis(subject_dic, mask_img=None,
                           mesh=False):
    """ Combine the AP and PA images """
    from nibabel import load, save
    from nilearn.plotting import plot_stat_map

    session_ids = subject_dic['session_id']
    task_ids = _session_id_to_task_id(session_ids)
    paradigms = np.unique(task_ids)
    if mask_img is None:
        mask_img = os.path.join(subject_dic['output_dir'], "mask.nii.gz")

    # Guessing paradigm from file name
    for paradigm in paradigms:
        # select the sessions relevant for the paradigm
        session_paradigm = [session_id for (session_id, task_id) in
                            zip(session_ids, task_ids)
                            if task_id == paradigm]
        # define the relevant contrasts
        contrasts = make_contrasts(paradigm).keys()
        # create write_dir
        if mesh is not False:
            write_dir = os.path.join(subject_dic['output_dir'],
                                     'res_task-%s_space-%s_dir-ffx' % (paradigm, mesh))
            dirs = [os.path.join(write_dir, stat) for stat in [
                    'effect_size_maps', 'effect_variance_maps', 'stat_maps']]
        else:
            write_dir = os.path.join(subject_dic['output_dir'],
                                     'res_task-%s_space-MNI305_dir-ffx' % paradigm)
            dirs = [os.path.join(write_dir, stat) for stat in [
                'effect_size_maps', 'effect_variance_maps', 'stat_maps']]
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
        print(write_dir)

        # iterate across contrasts
        for contrast in contrasts:
            print('fixed effects for contrast %s. ' % contrast)
            if mesh is not False:
                from nibabel.gifti import write
                for side in ['lh', 'rh']:
                    effect_size_maps, effect_variance_maps, data_available =\
                        _load_summary_stats(
                            subject_dic['output_dir'],
                            np.unique(session_paradigm),
                            contrast,
                            data_available=True, side=side, mesh=mesh)
                    if not data_available:
                        raise ValueError('Missing texture stats files for '
                                         'fixed effects computations')
                    ffx_effects, ffx_variance, ffx_stat = fixed_effects_surf(
                        effect_size_maps, effect_variance_maps)
                    write(ffx_effects, os.path.join(
                        write_dir, 'effect_size_maps/%s_%s.gii' % (contrast, side)))
                    write(ffx_variance, os.path.join(
                        write_dir, 'effect_variance_maps/%s_%s.gii' %
                        (contrast, side)))
                    write(ffx_stat, os.path.join(
                        write_dir, 'stat_maps/%s_%s.gii' % (contrast, side)))
            else:
                effect_size_maps, effect_variance_maps, data_available =\
                    _load_summary_stats(
                        subject_dic['output_dir'], session_paradigm, contrast,
                        data_available=True)
                shape = load(effect_size_maps[0]).shape
                if len(shape) > 3:
                    if shape[3] > 1:  # F contrast, skipping
                        continue
                ffx_effects, ffx_variance, ffx_stat = fixed_effects_img(
                    effect_size_maps, effect_variance_maps, mask_img)
                
                save(ffx_effects, os.path.join(
                    write_dir, 'effect_size_maps/%s.nii.gz' % contrast))
                save(ffx_variance, os.path.join(
                    write_dir, 'effect_variance_maps/%s.nii.gz' % contrast))
                save(ffx_stat, os.path.join(
                    write_dir, 'stat_maps/%s.nii.gz' % contrast))
                plot_stat_map(
                    ffx_stat, bg_img=subject_dic['anat'], display_mode='z',
                    dim=0, cut_coords=7, title=contrast, threshold=3.0,
                    output_file=os.path.join(write_dir,
                                             'stat_maps/%s.png' % contrast))


def fixed_effects_surf(con_imgs, var_imgs):
    """Idem fixed_effects_img but for surfaces"""
    from nibabel import load
    from nibabel.gifti import GiftiDataArray, GiftiImage
    con, var = [], []
    for (con_img, var_img) in zip(con_imgs, var_imgs):
        con.append(np.ravel([
            darrays.data for darrays in load(con_img).darrays]))
        var.append(np.ravel([
            darrays.data for darrays in load(var_img).darrays]))

    outputs = []
    intents = ['NIFTI_INTENT_ESTIMATE', 'NIFTI_INTENT_ESTIMATE', 't test']
    arrays = fixed_effects(con, var)
    for array, intent in zip(arrays, intents):
        gii = GiftiImage(
            darrays=[GiftiDataArray().from_array(array, intent)])
        outputs.append(gii)

    return outputs


def fixed_effects_img(con_imgs, var_imgs, mask_img):
    """Compute the fixed effets given images of effects and variance

    Parameters
    ----------
    con_imgs: list of Nifti1Images or strings
              the input contrast images
    var_imgs: list of Nifti1Images or strings
              the input variance images
    mask_img: Nifti1Image or string,
              mask image

    returns
    -------
    ffx_con: Nifti1Image,
             the fixed effects contrast computed within the mask
    ffx_var: Nifti1Image,
             the fixed effects variance computed within the mask
    ffx_stat: Nifti1Image,
             the fixed effects t-test computed within the mask
    """
    import nibabel as nib
    con, var = [], []
    if isinstance(mask_img, str):
        mask_img = nib.load(mask_img)
    mask = mask_img.get_data().astype(np.bool)
    for (con_img, var_img) in zip(con_imgs, var_imgs):
        if isinstance(con_img, str):
            con_img = nib.load(con_img)
        if isinstance(var_img, str):
            var_img = nib.load(var_img)
        con.append(con_img.get_data()[mask])
        var.append(var_img.get_data()[mask])

    arrays = fixed_effects(con, var)
    outputs = []
    for array in arrays:
        vol = mask.astype(np.float)
        vol[mask] = array.ravel()
        outputs.append(nib.Nifti1Image(vol, mask_img.affine))
    return outputs


def fixed_effects(contrasts, variances):
    """Compute the fixed effets given arrays of effects and variance
    """
    tiny = 1.e-16
    con, var = np.asarray(contrasts), np.asarray(variances)
    var = np.maximum(var, tiny)
    prec = 1. / var
    ffx_var = 1. / np.sum(prec, 0)
    ffx_con = np.sum(con * prec, 0) * ffx_var
    ffx_stat = ffx_con / np.sqrt(ffx_var)
    return [ffx_con, ffx_var, ffx_stat]
