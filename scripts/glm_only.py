"""
Synopsis: script for GLM and stats on IBC datasets

Author: THIRION Bertrand, PINHO Ana Luisa 2020

Compatibility: Python 3.5

"""

import os
import glob
from joblib import Parallel, delayed
from pypreprocess.conf_parser import _generate_preproc_pipeline
from ibc_public.utils_pipeline import fixed_effects_analysis, first_level
from pipeline import (clean_subject, clean_anatomical_images, _adapt_jobfile,
                      prepare_derivatives)
from ibc_public.utils_data import get_subject_session


SUBJECTS = [1, 2, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15]

RETINO_REG = dict(
    [(session_id, 'sin_cos_regressors.csv')
        for session_id in [
                'task-WedgeAnti_dir-pa', 'task-WedgeClock_dir-ap', 'task-ContRing_dir-ap',
                'task-WedgeAnti_dir-ap', 'task-ExpRing_dir-pa', 'task-WedgeClock_dir-pa']] +
    [('task-ClipsTrn_run-10', None),
     ('task-ClipsTrn_run-11', None),
     ('task-ClipsTrn_run-12', None)])
IBC = '/neurospin/ibc'
# IBC = '/storage/store2/data/ibc/'


def generate_glm_input(jobfile, smooth=None, lowres=False):
    """ retrun a list of dictionaries that represent the data available
    for GLM analysis"""
    list_subjects, params = _generate_preproc_pipeline(jobfile)
    output = []
    for subject in list_subjects:
        if lowres:
            output_dir = subject.output_dir.replace('derivatives', '3mm')
        elif smooth is not None:
            output_dir = subject.output_dir.replace('derivatives',
                                                    'smooth_derivatives')
        else:
            output_dir = subject.output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        anat = glob.glob(os.path.join(subject.anat_output_dir,
                                      'wsub*_T1w.nii.gz'))[0]
        reports_output_dir = os.path.join(output_dir, 'reports')
        report_log_filename = os.path.join(reports_output_dir,
                                           'report_log.html')
        report_preproc_filename = os.path.join(
            reports_output_dir, 'report_preproc.html')
        report_filename = os.path.join(reports_output_dir, 'report.html')
        tmp_output_dir = os.path.join(output_dir, 'tmp')
        basenames = ['wr' + os.path.basename(func_)[:-3]
                     for func_ in subject.func]
        func = [os.path.join(session_output_dir, basename + '.gz')
                for (session_output_dir, basename) in zip(
                        subject.session_output_dirs, basenames)]
        if lowres:
            func = [f.replace('derivatives', '3mm') for f in func]

        realignment_parameters = [
            os.path.join(session_output_dir, 'rp_' + basename[2:-4] + '.txt')
            for (session_output_dir, basename) in
            zip(subject.session_output_dirs, basenames)]

        hrf_model = subject.hrf_model
        if 'retino' in jobfile or 'stanford1' in jobfile:
            hrf_model = 'spm'

        subject_ = {
            'scratch': output_dir,
            'output_dir': output_dir,
            'session_output_dirs': subject.session_output_dirs,
            'anat_output_dir': subject.anat_output_dir,
            'tmp_output_dir': tmp_output_dir,
            'data_dir': subject.data_dir,
            'subject_id': subject.subject_id,
            'session_id': subject.session_id,
            'TR': subject.TR,
            'drift_model': subject.drift_model,
            'high_pass': 1. / 128,
            'time_units': subject.time_units,
            'hrf_model': hrf_model,
            'anat': anat,
            'onset': subject.onset,
            'report': True,
            'reports_output_dir': reports_output_dir,
            'report_log_filename': report_log_filename,
            'report_preproc_filename': report_preproc_filename,
            'report_filename': report_filename,
            'basenames': basenames,
            'func': func,
            'n_sessions': len(func),
            'realignment_parameters': realignment_parameters,
        }
        output.append(subject_)
    return output


def run_subject_glm(jobfile, protocol, subject, session=None, smooth=None,
                    lowres=False):
    """ Create jobfile and run it """
    if protocol == 'preference' and subject in ['sub-11']:
        jobfile = 'ini_files/IBC_preproc_preference_sub-11.ini'
    elif protocol == 'stanford3' and subject in ['sub-15']:
        jobfile = 'ini_files/IBC_preproc_stanford3_sub-15.ini'
    output_name = os.path.join(
        '/tmp', os.path.basename(jobfile)[:-4] + '_%s.ini' % subject)
    _adapt_jobfile(jobfile, subject, output_name, session)
    list_subjects_update = generate_glm_input(output_name, smooth, lowres)
    clean_anatomical_images(IBC)
    compcorr = True
    if protocol == 'mathlang':
        compcorr = False  # till we orthogonalize comcorr wrt task regressors
    if lowres:
        mask_img = '../ibc_data/gm_mask_3mm.nii.gz'
    else:
        mask_img = '../ibc_data/gm_mask_1_5mm.nii.gz'
    for subject in list_subjects_update:
        subject['onset'] = [onset for onset in subject['onset']
                            if onset is not None]
        clean_subject(subject)
        stop
        if len(subject['session_id']) > 0:
            if protocol == 'clips4':
                first_level(subject, compcorr=compcorr,
                            additional_regressors=RETINO_REG,
                            smooth=smooth, mask_img=mask_img)
            else:
                first_level(subject, compcorr=compcorr, smooth=smooth,
                            mask_img=mask_img)
                fixed_effects_analysis(subject, mask_img=mask_img)


if __name__ == '__main__':
    prepare_derivatives(IBC)
    # protocols = ['rsvp-language', 'hcp1', 'archi', 'screening', 'hcp2']
    # protocols = ['clips4', 'mtt1', 'mtt2', 'preference']
    # protocols = ['biological_motion', 'camcan1', 'camcan2', 'audio1', 'audio2']
    # protocols += ['optimism' 'fbirn', 'enumeration', 'color', 'lyon1', 'lyon2', 'navigation', 'mathlang']
    # protocols = ['self', 'search', 'scene', 'tom', 'stanford1', 'stanford2', 'stanford3']
    # protocols = ['audio1', 'audio2']
    # protocols = ['optimism', 'abstraction', 'leuven', 'abstraction']
    protocols = ['leuven']

    for protocol in protocols:
        jobfile = 'ini_files/IBC_preproc_%s.ini' % protocol
        subject_session = get_subject_session(protocol)
        Parallel(n_jobs=1)(
            delayed(run_subject_glm)(
                jobfile, protocol, subject, session, lowres=True, smooth=5)
            for (subject, session) in subject_session)
    
    smooth = 5
    for protocol in protocols:
        jobfile = 'ini_files/IBC_preproc_%s.ini' % protocol
        subject_session = get_subject_session(protocol)
        Parallel(n_jobs=6)(
            delayed(run_subject_glm)(
                jobfile, protocol, subject, session, smooth=smooth)
            for (subject, session) in subject_session)

    smooth = None
    for protocol in protocols:
        jobfile = 'ini_files/IBC_preproc_%s.ini' % protocol
        subject_session = get_subject_session(protocol)
        Parallel(n_jobs=6)(
            delayed(run_subject_glm)(
                jobfile, protocol, subject, session, smooth=smooth)
            for (subject, session) in subject_session)
