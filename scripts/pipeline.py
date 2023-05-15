"""
:Synopsis: script for distortion correction, preproc + stats
on IBC datasets

:Author: DOHMATOB Elvis Dopgima, DENGHIEN Isabelle, THIRION Bertrand

"""
import os
import json
import glob
from pypreprocess.nipype_preproc_spm_utils import (do_subjects_preproc,
                                                   SubjectData)
from pypreprocess.conf_parser import _generate_preproc_pipeline
from joblib import Memory, Parallel, delayed
from ibc_public.utils_pipeline import (
    fixed_effects_analysis, first_level, fsl_topup)
from ibc_public.utils_data import get_subject_session
from script_resample_normalized_data import resample_func_and_anat


def clean_anatomical_images(main_dir):
    """ Removed NaNs from SPM12-supplied images """
    import nibabel as nib
    from numpy import isnan
    subjects = ['sub-%02d' % i for i in range(1, 16)]
    sessions = ['ses-%02d' % i for i in range(1, 50)]
    for subject in subjects:
        for session in sessions:
            anat_img = os.path.join(
                main_dir, 'derivatives', subject, session,
                'anat', 'w%s_%s_T1w.nii') % (subject, session)
            dst = os.path.join(
                main_dir, 'derivatives', subject, session, 'anat',
                'w%s_%s_T1w_nonan.nii.gz') % (subject, session)
            if not os.path.exists(anat_img):
                continue
            # Don't do it again if it was already done
            if os.path.exists(dst):
                continue
            anat_data = nib.load(anat_img).get_data()
            anat_data[isnan(anat_data)] = 0
            anat_img_clean = nib.Nifti1Image(
                anat_data, nib.load(anat_img).affine)
            nib.save(anat_img_clean, dst)


def clean_subject(subject):
    """ Remove sessions with missing data (e.g. onset files)
    before fitting GLM"""
    onsets, rps, funcs, session_ids = [], [], [], []
    if not ('onset' in subject.keys() and
            'realignment_parameters' in subject.keys() and
            'func' in subject.keys() and
            'session_id' in subject.keys()):
        subject['onset'] = onsets
        subject['func'] = funcs
        subject['realignment_parameters'] = rps
        subject['session_id'] = session_ids
        return subject

    # if onset is None for some of the sessions, remove them
    for onset, rp, func, session_id in zip(subject['onset'],
                                           subject['realignment_parameters'],
                                           subject['func'],
                                           subject['session_id']):
        if ((onset is not None)):
            rps.append(rp)
            funcs.append(func)
            session_ids.append(session_id)
            onsets.append(onset)

    subject['onset'] = onsets
    subject['func'] = funcs
    subject['realignment_parameters'] = rps
    subject['session_id'] = session_ids
    subject['high_pass'] = 1. / 128
    return subject


def prepare_derivatives(main_dir):
    import shutil
    source_dir = os.path.join(main_dir, 'sourcedata')
    output_dir = os.path.join(main_dir, 'derivatives')
    subjects = ['sub-%02d' % i for i in range(0, 16)]
    sess = ['ses-%02d' % j for j in range(0, 50)]
    modalities = ['anat', 'fmap', 'func', 'dwi']
    dirs = ([output_dir] +
            [os.path.join(output_dir, subject) for subject in subjects
             if os.path.exists(os.path.join(source_dir, subject))] +
            [os.path.join(output_dir, subject, ses) for subject in subjects
             for ses in sess
             if os.path.exists(os.path.join(source_dir, subject, ses))] +
            [os.path.join(output_dir, subject, ses, modality)
             for subject in subjects
             for ses in sess for modality in modalities
             if os.path.exists(
                os.path.join(source_dir, subject, ses, modality))])

    for dir_ in dirs:
        if not os.path.exists(dir_):
            print(dir_)
            os.mkdir(dir_)

    for subject in subjects:
        for ses in sess:
            tsv_files = glob.glob(
                os.path.join(source_dir, subject, ses, 'func', '*.tsv'))
            dst = os.path.join(output_dir, subject, ses, 'func')
            for tsv_file in tsv_files:
                shutil.copyfile(tsv_file,
                                os.path.join(dst, os.path.basename(tsv_file)))
        highres = glob.glob(
            os.path.join(source_dir, subject, 'ses-*', 'anat', '*'))

        for hr in highres:
            parts = hr.split('/')
            dst = os.path.join(
                output_dir, subject, parts[-3], 'anat', parts[-1])
            if not os.path.isfile(dst[:-3]):
                shutil.copyfile(hr, dst)
                if dst[-3:] == '.gz':
                    os.system('gunzip %s' % dst)


def run_topup(mem, data_dir, subject, ses, acq=None):
    write_dir = os.path.join(data_dir, 'derivatives', subject, ses)
    # gather the BOLD data to be corrected
    functional_data = glob.glob(
        os.path.join(data_dir, 'sourcedata', subject, ses, 'func/*.nii.gz'))
    if functional_data == []:
        return
    if acq == 'mb6':
        functional_data = [
            fd for fd in functional_data if 'RestingState' in fd]
    functional_data.sort()

    # gather the field maps
    if acq == 'mb3':
        field_maps = [
            glob.glob(
                os.path.join(data_dir, 'sourcedata', subject, ses,
                             'fmap/*acq-mb3_dir-ap_epi.nii.gz'))[-1],
            glob.glob(
                os.path.join(data_dir, 'sourcedata', subject, ses,
                             'fmap/*acq-mb3_dir-pa_epi.nii.gz'))[-1]]
    elif acq == 'mb6':
        field_maps = [
            glob.glob(
                os.path.join(data_dir, 'sourcedata', subject, ses,
                             'fmap/*acq-mb6_dir-ap_epi.nii.gz'))[-1],
            glob.glob(
                os.path.join(data_dir, 'sourcedata', subject, ses,
                             'fmap/*acq-mb6_dir-pa_epi.nii.gz'))[-1]]
    elif acq is None:
        field_maps = [
            glob.glob(
                os.path.join(data_dir, 'sourcedata', subject, ses,
                             'fmap/*dir-ap_epi.nii.gz'))[-1],
            glob.glob(
                os.path.join(data_dir, 'sourcedata', subject, ses,
                             'fmap/*dir-pa_epi.nii.gz'))[-1]]
    else:
        raise ValueError('Unknown acq %s' % acq)
    return fsl_topup(field_maps, functional_data, mem, write_dir)


def apply_topup(main_dir, cache_dir, subject_sess=None, acq=None):
    """ Call topup on the datasets """
    mem = Memory(cache_dir)
    if subject_sess is None:
        subject_sess = [('sub-%02d, ses-%02d' % (i, j)) for i in range(0, 50)
                        for j in range(0, 15)]
    Parallel(n_jobs=1)(
        delayed(run_topup)(mem, main_dir, subject_ses[0], subject_ses[1],
                           acq=acq)
        for subject_ses in subject_sess)


def _adapt_jobfile(jobfile, subject, output_name, session=None):
    """ small utility to create temporary jobfile"""
    f1 = open(jobfile, 'r')
    f2 = open(output_name, 'w')
    for line in f1.readlines():
        if session is None:
            f2.write(line.replace('sub-01', subject))
        else:
            f2.write(line.replace('sub-01', subject).replace('ses-*', session))

    f1.close()
    f2.close()
    return output_name


def run_subject_preproc(jobfile, subject, session=None):
    """ Create jobfile and run it on """
    output_name = os.path.join(
        '/tmp', os.path.basename(jobfile)[:-4] + '_%s.ini' % subject)
    _adapt_jobfile(jobfile, subject, output_name, session)
    # Read the jobfile
    list_subjects, params = _generate_preproc_pipeline(output_name)
    # Preproc and Dump data
    subject_data = do_subjects_preproc(output_name, report=True)
    return subject_data


if __name__ == '__main__':
    # correction of distortion_parameters
    # custom solution, to be improved in the future
    main_dir = '/neurospin/ibc/'
    cache_dir = '/neurospin/tmp/ibc'
    prepare_derivatives(main_dir)
    do_topup = True
    protocol =  'abstraction'
    subject_session = [('sub-12', 'ses-46')]
    
    if do_topup:
        acq = None
        if protocol in ['rs']:
            acq = 'mb6'
        elif protocol in ['mtt1', 'mtt2']:
            acq = 'mb3'
        apply_topup(main_dir, cache_dir, subject_session, acq=acq)

    subject_data = []
    jobfile = 'ini_files/IBC_preproc_%s.ini' % protocol
    subject_data_ = Parallel(n_jobs=1)(
        delayed(run_subject_preproc)(jobfile, subject, session)
        for subject, session in subject_session)
    subject_data = subject_data + subject_data_[0]

    list_subject_update = []
    list_files_parameter = []
    for dict_subject in subject_data:
        dict_subject = dict_subject.__dict__
        update_dict_subject_data = {
            k: v for (k, v) in dict_subject.items()
            if v.__class__.__module__ == 'builtins'}
        update_dict_subject_data.pop('nipype_results')
        list_subject_update.append(update_dict_subject_data)

    # FileName for the dumped dictionnary from the preproc
    json_file_name = 'subjects_data.json'
    json_file = open(json_file_name, "w")
    json.dump(list_subject_update, json_file)
    json_file.flush()

    # resampling toward pre-defined shape
    resample_func_and_anat()
    """
    # Load the dump data
    list_subjects_update = json.load(open(json_file_name))
    for subject in list_subjects_update:
        subject = clean_subject(subject)
        print(subject['subject_id'], subject['onset'])
        if len(subject['session_id']) > 0:
            first_level(subject, compcorr=True)
            fixed_effects_analysis(subject)
    """
