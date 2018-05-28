"""
:Synopsis: script for GLM and stats on IBC datasets

:Author: THIRION Bertrand

"""

import os
from pypreprocess.conf_parser import _generate_preproc_pipeline
from joblib import Parallel, delayed
from utils_pipeline import fixed_effects_analysis, first_level
import glob
from pipeline import (clean_subject, clean_anatomical_images, _adapt_jobfile,
                      get_subject_session, prepare_derivatives)


SUBJECTS = [1, 2, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14]

RETINO_REG = dict([(session_id, 'sin_cos_regressors.csv')
                   for session_id in ['wedge_anti_pa', 'wedge_clock_ap', 'cont_ring_ap',
                                      'wedge_anti_ap', 'exp_ring_pa', 'wedge_clock_pa']])

def generate_glm_input(jobfile, smooth=None):
    """ retrun a list of dictionaries that represent the data available
    for GLM analysis"""
    list_subjects, params = _generate_preproc_pipeline(jobfile)
    output = []
    for subject in list_subjects:
        if smooth is not None:
            output_dir = subject.output_dir.replace('derivatives',
                                                    'smooth_derivatives')
        else:
            output_dir = subject.output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        anat = glob.glob(os.path.join(subject.anat_output_dir,
                                      'wsub*_T1w_nonan.nii.gz'))[0]
        #gm = glob.glob(os.path.join(subject.anat_output_dir, 'c1sub*_T1w*.nii'))[0]
        #wm = glob.glob(os.path.join(subject.anat_output_dir, 'c2sub*_T1w*.nii'))[0]
        #csf = glob.glob(os.path.join(subject.anat_output_dir, 'c3sub*_T1w*.nii'))[0]
        #mwgm = glob.glob(os.path.join(subject.anat_output_dir, 'mwc1sub*_T1w.nii'))[0]
        #mwwm = glob.glob(os.path.join(subject.anat_output_dir, 'mwc2sub*_T1w.nii'))[0]
        #mwcsf = glob.glob(os.path.join(subject.anat_output_dir, 'mwc3sub*_T1w.nii'))[0]
        reports_output_dir = os.path.join(output_dir, 'reports')
        report_log_filename = os.path.join(reports_output_dir, 'report_log.html')
        report_preproc_filename = os.path.join(
            reports_output_dir, 'report_preproc.html')
        report_filename = os.path.join(reports_output_dir, 'report.html')
        tmp_output_dir = os.path.join(output_dir, 'tmp')
        basenames = ['wr' + os.path.basename(func_)[:-3] for func_ in subject.func]
        func = [os.path.join(session_output_dir, basename + '.gz')
                for (session_output_dir, basename) in zip(
                        subject.session_output_dirs, basenames)]
        realignment_parameters = [
            os.path.join(session_output_dir, 'rp_' + basename[2:-4] + '.txt')
            for (session_output_dir, basename) in
            zip(subject.session_output_dirs, basenames)]

        subject_ = {
            'scratch': output_dir,
            'output_dir':output_dir,
            'session_output_dirs': subject.session_output_dirs,
            'anat_output_dir': subject.anat_output_dir,
            'tmp_output_dir': tmp_output_dir,
            'data_dir': subject.data_dir,
            'subject_id': subject.subject_id,
            'session_id':subject.session_id,
            'TR': subject.TR,
            'drift_model':subject.drift_model,
            'hfcut': subject.hfcut,
            'time_units': subject.time_units,
            'hrf_model': subject.hrf_model,
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
            # 'failed': False,
            # 'tsdiffana':params.tsdiffana,
            # 'warpable': subject.warpable
            # 'isdicom': False,
            # 'parameter_file':,
            # 'mean_realigned_file': None,
            # 'mwgm': mwgm,
            # 'gm': gm,
            # 'wm': wm,
            # 'csf': csf,
            # 'mwcsf': mwcsf,
            # 'mwwm': mwwm,

        }
        output.append(subject_)
    return output

        
def run_subject_glm(jobfile, protocol, subject, session=None, smooth=None):
    """ Create jobfile and run it """
    if protocol == 'preferences' and subject in ['sub-11']:
        jobfile = 'ini_files/IBC_preproc_preferences_sub-11.ini'
    output_name = os.path.join(
        '/tmp', os.path.basename(jobfile)[:-4] + '_%s.ini' % subject)
    _adapt_jobfile(jobfile, subject, output_name, session)
    list_subjects_update = generate_glm_input(output_name, smooth)
    clean_anatomical_images('/neurospin/ibc')
    mask_img = '/neurospin/ibc/smooth_derivatives/group/resampled_gm_mask.nii.gz'
    for subject in list_subjects_update:
        subject['onset'] = [onset for onset in subject['onset'] if onset is not None]
        stop
        clean_subject(subject)
        if len(subject['session_id']) > 0:
            if protocol == 'clips4':
                first_level(subject, compcorr=True, additional_regressors=RETINO_REG,
                            smooth=None)
            else:
                first_level(subject, compcorr=True, smooth=smooth, mask_img=mask_img)
                fixed_effects_analysis(subject, mask_img=mask_img)
    

                
if __name__ == '__main__':
    prepare_derivatives('/neurospin/ibc/')
    smooth = 5 # None  # 
    for protocol in ['enumeration']:  # ['hcp1', 'hcp2', 'language', 'mtt2' 'preferences']
        jobfile = 'ini_files/IBC_preproc_%s.ini' % protocol
        subject_session = get_subject_session(protocol)
        Parallel(n_jobs=1)(
            delayed(run_subject_glm)(jobfile, protocol, subject, session, smooth)
            for (subject, session) in subject_session)
    
    smooth = None
    for protocol in ['enumeration']:
        jobfile = 'ini_files/IBC_preproc_%s.ini' % protocol
        subject_session = get_subject_session(protocol)
        Parallel(n_jobs=1)(
            delayed(run_subject_glm)(jobfile, protocol, subject, session, smooth)
            for (subject, session) in subject_session)

