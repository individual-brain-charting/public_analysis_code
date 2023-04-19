"""
:Synopsis: script for GLM and stats only on IBC datasets

:Author: THIRION Bertrand

"""
import os
from pypreprocess.conf_parser import _generate_preproc_pipeline
from joblib import Parallel, delayed
from ibc_public.utils_pipeline import fixed_effects_analysis, first_level

from pipeline import (clean_subject, clean_anatomical_images,
                      _adapt_jobfile, get_subject_session)


SUBJECTS = [1, 2, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15]
retino_sessions = ['task-WedgeAnti_dir-pa',
                   'task-WedgeClock_dir-ap',
                   'task-ContRing_dir-ap',
                   'task-WedgeAnti_dir-ap',
                   'task-ExpRing_dir-pa',
                   'task-WedgeClock_dir-pa']
RETINO_REG = dict([(session_id, 'sin_cos_regressors.csv')
                   for session_id in retino_sessions])
IBC = 'neurospin/ibc'


def generate_glm_input(jobfile, mesh=None):
    """ retrun a list of dictionaries that represent the data available
    for GLM analysis"""
    list_subjects, params = _generate_preproc_pipeline(jobfile)
    output = []
    for subject in list_subjects:
        output_dir = subject.output_dir
        reports_output_dir = os.path.join(output_dir, 'reports')
        basenames = ['wr' + os.path.basename(func_)[:-3]
                     for func_ in subject.func]
        gii_basenames = ['r' + os.path.basename(func_).split('.')[0] +
                         '_{}_lh.gii'.format(mesh) for func_ in subject.func]
        gii_basenames += ['r' + os.path.basename(func_).split('.')[0] +
                          '_{}_rh.gii'.format(mesh) for func_ in subject.func]
        func = [os.path.join(output_dir, 'freesurfer', basename)
                for basename in gii_basenames]
        realignment_parameters = [
            os.path.join(session_output_dir, 'rp_' + basename[2:-4] + '.txt')
            for (session_output_dir, basename) in
            zip(subject.session_output_dirs, basenames)] * 2
        session_ids = [session_id for (session_id, onset) in
                       zip(subject.session_id, subject.onset)
                       if onset is not None]
        onsets = [onset for onset in subject.onset if onset is not None]
        subject_ = {
            'output_dir': output_dir,
            'session_output_dirs': subject.session_output_dirs,
            'subject_id': subject.subject_id,
            'session_id': session_ids * 2,
            'TR': subject.TR,
            'drift_model': subject.drift_model,
            'high_pass': 1. / 128,
            'time_units': subject.time_units,
            'hrf_model': subject.hrf_model,
            'onset': onsets * 2,
            'report': True,
            'reports_output_dir': reports_output_dir,
            'basenames': gii_basenames,
            'func': func,
            'realignment_parameters': realignment_parameters,
        }
        output.append(subject_)
    return output


def run_subject_surface_glm(jobfile, subject, session, protocol, mesh=None, compcorr=True):
    """ Create jobfile and run it """
    output_name = os.path.join(
        '/tmp', os.path.basename(jobfile)[:-4] + '_%s.ini' % subject)
    _adapt_jobfile(jobfile, subject, output_name, session)
    list_subjects_update = generate_glm_input(output_name, mesh)
    clean_anatomical_images(IBC)
    if protocol == 'mathlang':
        compcorr = False
    for subject in list_subjects_update:
        clean_subject(subject)
        if len(subject['session_id']) > 0:
            print(len(subject['session_id']))
        if len(subject['session_id']) > 0:
            if protocol == 'retino':
                subject['onset'] = [''] * len(subject['onset'])
                first_level(subject, compcorr=compcorr,
                            additional_regressors=RETINO_REG,
                            smooth=None, mesh=mesh)
            else:
                first_level(subject, compcorr=compcorr, smooth=None, mesh=mesh)
                fixed_effects_analysis(subject, mesh=mesh)


if __name__ == '__main__':
    """
    protocols = ['stanford3']
    protocols += ['screening', 'rsvp-language', 'hcp1', 'hcp2', 'archi']
    protocols += ['preference', 'mtt1', 'mtt2', 'tom', 'self',
                 'retino']
    protocols += ['mathlang', 'enumeration', 'lyon1', 'lyon2']
    protocols = ['stanford1', 'stanford2, 'stanford3']
    protocols = ['preference', 'audio1', 'audio2']
    protocols = ['enumeration', 'biological_motion', 'reward', 'mathlang', 
                 'navigation', 'search']
    protocols = ['scene', 'color']
    """
    protocols = ['reward']
    for protocol in protocols:
        jobfile = 'ini_files/IBC_preproc_%s.ini' % protocol
        acquisition = protocol
        if protocol == 'retino':
            acquisition = 'clips4'
        subject_session = sorted(get_subject_session(acquisition))
        for mesh in ['fsaverage5', 'individual', 'fsaverage7']:
            Parallel(n_jobs=4)(
                delayed(run_subject_surface_glm)(
                    jobfile, subject, session, protocol, mesh=mesh)
                for (subject, session) in subject_session)
