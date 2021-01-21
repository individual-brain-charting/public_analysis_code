
import glob
import os
from nilearn.image import math_img, resample_to_img
from scipy.ndimage import grey_closing
from nibabel import Nifti1Image

derivatives = '/neurospin/ibc/derivatives'
subjects = sorted(glob.glob(os.path.join(derivatives, 'sub-*')))
"""

for subject in subjects:
    subject_id = os.path.basename(subject)
    src = os.path.join(subject, 'ses-00', 'anat',
                       'w%s_ses-00_T1w.nii.gz' % subject_id)
    dst = os.path.join(subject, 'ses-00', 'anat',
                       'w%s_ses-00_T1w_bet.nii.gz' % subject_id)
    wm = os.path.join(subject, 'ses-00', 'anat',
                      'mwc2%s_ses-00_T1w.nii.gz' % subject_id)
    gm = os.path.join(subject, 'ses-00', 'anat',
                      'mwc1%s_ses-00_T1w.nii.gz' % subject_id)
    brain = math_img('i1 + i2', i1=wm, i2=gm)
    brain.to_filename('/tmp/brain_%s.nii.gz' % subject_id)
    large_brain = resample_to_img(brain, src)
    large_brain_data = grey_closing(large_brain.get_data(), 5)
    large_brain = Nifti1Image(large_brain_data, large_brain.get_affine())
    bet_img = math_img('i1 * i2', i1=large_brain, i2=src)
    bet_img.to_filename(dst)
"""

subject_sessions = [('sub-14', 'ses-05'),
                    ('sub-04', 'ses-08'),
                    ('sub-05', 'ses-08'),
                    ('sub-06', 'ses-09'),
                    ('sub-07', 'ses-09'),
                    ('sub-08', 'ses-09'),
                    ('sub-09', 'ses-09'),
                    ('sub-11', 'ses-09'),
                    ('sub-12', 'ses-09'),
                    ('sub-13', 'ses-09'),
                    ('sub-01', 'ses-10'),
                    ('sub-01', 'ses-10')]

for (subject, session) in subject_sessions:
    subject_id = subject
    src = os.path.join(derivatives, subject, session, 'anat',
                       'w%s_%s_acq-highres_T1w.nii.gz' % (subject_id, session))
    dst = os.path.join(derivatives, subject, session, 'anat',
                       'w%s_%s_acq-highres_T1w_bet.nii.gz' %
                       (subject_id, session))
    wm = os.path.join(derivatives, subject, session, 'anat',
                      'mwc2%s_%s_acq-highres_T1w.nii.gz' %
                      (subject_id, session))
    gm = os.path.join(derivatives, subject, session, 'anat',
                      'mwc1%s_%s_acq-highres_T1w.nii.gz' %
                      (subject_id, session))
    brain = math_img('i1 + i2', i1=wm, i2=gm)
    brain.to_filename('/tmp/brain_%s.nii.gz' % subject_id)
    large_brain = resample_to_img(brain, src)
    large_brain_data = grey_closing(large_brain.get_data(), 5)
    large_brain = Nifti1Image(large_brain_data, large_brain.get_affine())
    bet_img = math_img('i1 * i2', i1=large_brain, i2=src)
    bet_img.to_filename(dst)
