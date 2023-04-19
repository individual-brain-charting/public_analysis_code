"""
This script does 2 things:
1. Freesurfer segmentation
2. project the coregistered fMRI images to the surface:
the surface is the grey-white matter interface of the subject

The purpose is to perform proper group analysis on the surface on fsaverage,
and use existing  atlases on the surface.

Author: Bertrand Thirion, Isabelle Courcol, 2013 -- 2016

Note
----
First run: export SUBJECTS_DIR=''
"""
import os
import glob
from nipype.caching import Memory
from joblib import Parallel, delayed
from nipype.interfaces.freesurfer import ReconAll, BBRegister
from pipeline import get_subject_session
import nibabel as nib
import numpy as np
from nilearn.image import smooth_img


work_dir = '/neurospin/ibc/derivatives'
subjects = ['sub-%02d' % i for i in [1, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15]]
subjects = ['sub-%02d' % i for i in [2]]
mem = Memory(base_dir='/neurospin/tmp/ibc')

# Step 1: Perform recon-all
os.environ['SUBJECTS_DIR'] = ''


def recon_all(work_dir, subject, high_res=True):
    # create directories in output_dir
    if high_res:
        # high-resolution T1
        anat_img = glob.glob(os.path.join(
            work_dir, subject,
            'ses-*/anat/sub-*_ses-*_acq-highres_T1w.nii*'))[0]
        print(anat_img)
        t1_dir = os.path.dirname(anat_img)
        os.system('recon-all -all -subjid %s -sd %s -hires -i %s '
                  '-expert expert.opts' % (subject, t1_dir, anat_img))
    else:
        # low-resolution T1
        subject_dir = os.path.join(work_dir, subject, 'ses-00')
        t1_dir = os.path.join(subject_dir, 'anat')
        anat_img = glob.glob(os.path.join(t1_dir,
                                          '%s_ses-00_T1w.nii*' % subject))[0]
        # reconall = mem.cache(ReconAll)
        # reconall(subject_id=subject,
        #         directive='all',
        #         subjects_dir=t1_dir,
        #         T1_files=anat_img)
        os.system('recon-all -all -subjid %s -sd %s' % (subject, t1_dir))

"""
Parallel(n_jobs=1)(delayed(recon_all)(work_dir, subject, True)
                   for subject in subjects)
"""

# Step 2: Perform the projection
def project_volume(work_dir, subject, sessions, do_bbr=True):
    t1_dir = os.path.join(work_dir, subject, 'ses-00', 'anat')
    for session in sessions:
        subject_dir = os.path.join(work_dir, subject, session)
        if not os.path.exists(subject_dir):
            continue
        fmri_dir = os.path.join(subject_dir, 'func')
        fs_dir = os.path.join(subject_dir, 'freesurfer')
        fmri_images = glob.glob(os.path.join(fmri_dir, 'rdc*.nii.gz'))

        # --------------------------------------------------------------------
        # run the projection using freesurfer
        os.environ['SUBJECTS_DIR'] = t1_dir
        if not os.path.exists(fs_dir):
            os.mkdir(fs_dir)

        # take the fMRI series
        print("fmri_images", fmri_images)
        for fmri_session in fmri_images:
            basename = os.path.basename(fmri_session).split('.')[0]
            print (basename)
            # output names
            # the .gii files are put in the same directory as the input fMRI
            left_fmri_tex = os.path.join(fs_dir, basename + '_individual_lh.gii')
            right_fmri_tex = os.path.join(fs_dir, basename + '_individual_rh.gii')
            if do_bbr:
                # use BBR registration to finesse the coregistration
                bbreg = BBRegister(
                    subject_id=subject, source_file=fmri_session,
                    init='header', contrast_type='t2')
                bbreg.run()

            # run freesrufer command for projection
            regheader = os.path.join(fmri_dir, basename +
                                     '_bbreg_%s.dat' % subject)
            print(os.system(
                '$FREESURFER_HOME/bin/mri_vol2surf --src %s --o %s '
                '--out_type gii --srcreg %s --hemi lh --projfrac-avg 0 2 0.2'
                % (fmri_session, left_fmri_tex, regheader)))

            print(os.system(
                '$FREESURFER_HOME/bin/mri_vol2surf --src %s --o %s '
                '--out_type gii --srcreg %s --hemi rh --projfrac-avg 0 2 0.2'
                % (fmri_session, right_fmri_tex, regheader)))

            # if there are NaN's in one of the file, redo the projection
            # on the smoothed fMRI
            Xl = nib.load(left_fmri_tex).darrays[0].data
            Xr = nib.load(right_fmri_tex).darrays[0].data
            filename = os.path.join(
                '/neurospin/tmp/bthirion/',
                's' + os.path.basename(fmri_session))
            if np.isnan(Xl).any() or np.isnan(Xr).any():
                smooth_img(fmri_session, 2).to_filename(filename)

            if np.isnan(Xl).any():
                print(os.system(
                    '$FREESURFER_HOME/bin/mri_vol2surf --src %s --o %s '
                    '--out_type gii --srcreg %s --hemi lh '
                    '--projfrac-avg 0 2 0.2'
                    % (filename, left_fmri_tex, regheader)))
                Xl = nib.load(left_fmri_tex).darrays[0].data
            if np.isnan(Xr).any():
                print(os.system(
                    '$FREESURFER_HOME/bin/mri_vol2surf --src %s --o %s '
                    '--out_type gii --srcreg %s --hemi rh '
                    '--projfrac-avg 0 2 0.2'
                    % (filename, right_fmri_tex, regheader)))
                Xr = nib.load(right_fmri_tex).darrays[0].data

            if np.isnan(Xl).any() or np.isnan(Xr).any():
                raise ValueError('persistent NaNs in the data')

            if os.path.exists(filename):
                os.remove(filename)

            # resample to fsaverage
            left_fsaverage_fmri_tex = os.path.join(
                fs_dir, basename + '_fsaverage7_lh.gii')
            right_fsaverage_fmri_tex = os.path.join(
                fs_dir, basename + '_fsaverage7_rh.gii')

            print(os.system(
                '$FREESURFER_HOME/bin/mri_surf2surf --srcsubject %s '
                '--srcsurfval %s --trgsurfval %s --trgsubject ico '
                '--trgicoorder 7 --hemi lh --nsmooth-out 5' %
                (subject, left_fmri_tex, left_fsaverage_fmri_tex)))
            print(os.system(
                '$FREESURFER_HOME/bin/mri_surf2surf --srcsubject %s '
                '--srcsurfval %s --trgsubject ico --trgicoorder 7 '
                '--trgsurfval %s --hemi rh --nsmooth-out 5' %
                (subject, right_fmri_tex, right_fsaverage_fmri_tex)))

            # check possible Nans and if necessary redo the stuff
            Xl = nib.load(left_fsaverage_fmri_tex).darrays[0].data
            Xr = nib.load(right_fsaverage_fmri_tex).darrays[0].data
            if np.isnan(Xl).any() or np.isnan(Xr).any():
                raise ValueError('persistent NaNs in the data')

            # resample to fsaverage5
            left_fsaverage_fmri_tex = os.path.join(
                fs_dir, basename + '_fsaverage5_lh.gii')
            right_fsaverage_fmri_tex = os.path.join(
                fs_dir, basename + '_fsaverage5_rh.gii')

            print(os.system(
                '$FREESURFER_HOME/bin/mri_surf2surf --srcsubject %s '
                '--srcsurfval %s --trgsurfval %s --trgsubject ico '
                '--trgicoorder 5 --hemi lh --nsmooth-out 2' %
                (subject, left_fmri_tex, left_fsaverage_fmri_tex)))
            print(os.system(
                '$FREESURFER_HOME/bin/mri_surf2surf --srcsubject %s '
                '--srcsurfval %s --trgsubject ico --trgicoorder 5 '
                '--trgsurfval %s --hemi rh --nsmooth-out 2' %
                (subject, right_fmri_tex, right_fsaverage_fmri_tex)))

            # check possible Nans and if necessary redo the stuff
            Xl = nib.load(left_fsaverage_fmri_tex).darrays[0].data
            Xr = nib.load(right_fsaverage_fmri_tex).darrays[0].data
            if np.isnan(Xl).any() or np.isnan(Xr).any():
                raise ValueError('persistent NaNs in the data')

            # finally smooth the textures on the individual anat
            print(os.system(
                '$FREESURFER_HOME/bin/mri_surf2surf --srcsubject %s '
                '--srcsurfval %s --trgsurfval %s --trgsubject %s '
                '--hemi lh --nsmooth-out 5' %
                (subject, left_fmri_tex, left_fmri_tex, subject)))
            print(os.system(
                '$FREESURFER_HOME/bin/mri_surf2surf --srcsubject %s '
                '--srcsurfval %s --trgsurfval %s --trgsubject %s'
                ' --hemi rh --nsmooth-out 5' %
                (subject, right_fmri_tex, right_fmri_tex, subject)))


#protocols = ['archi', 'screening', 'rsvp-language']
#protocols += ['preference', 'mtt1', 'mtt2', 'clips4', 'tom', 'self']
#protocols += ['hcp1', 'hcp2', 'lyon1', 'lyon2']
#protocols = ['audio1', 'audio2', 'lpp1', 'lpp2']
# protocols += ['enumeration', 'clips1', 'clips2', 'clips3', 'raiders1',
#              'raiders2', 'BBT1', 'BBT2', 'BBT3', 'self']
#protocols = ['mathlang']
# protocols = ['enumeration']
#protocols = ['stanford1', 'stanford2', 'stanford3']
# protocols = ['biological_motion', 'navigation' , 'camcan1', 'camcan2', 'fbirn', 'search', 'reward', 'scene', 'monkey_kingdom', 'color']

protocols = ['scene']
subject_sessions = sorted(get_subject_session(protocols))

Parallel(n_jobs=6)(
    delayed(project_volume)(work_dir, subject_session[0], [subject_session[1]],
                            do_bbr=True)
    for subject_session in subject_sessions)

