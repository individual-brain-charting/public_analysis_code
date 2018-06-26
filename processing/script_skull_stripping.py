
import commands
import glob
import os
from nilearn.image import math_img, resample_to_img
from scipy.ndimage import grey_closing
from nibabel import Nifti1Image

derivatives = '/neurospin/ibc/derivatives'
subjects = sorted(glob.glob(os.path.join(derivatives, 'sub-*')))

for subject in subjects:
    subject_id = os.path.basename(subject)
    src = os.path.join(subject, 'ses-00', 'anat', 'w%s_ses-00_T1w.nii.gz' % subject_id)
    dst = os.path.join(subject, 'ses-00', 'anat', 'w%s_ses-00_T1w_bet.nii.gz' % subject_id)
    #commands.getoutput('fsl5.0-bet %s %s -f 0.3' % (src, dst))
    wm = os.path.join(subject, 'ses-00', 'anat', 'mwc2%s_ses-00_T1w.nii.gz' % subject_id)
    gm = os.path.join(subject, 'ses-00', 'anat', 'mwc1%s_ses-00_T1w.nii.gz' % subject_id)
    brain = math_img('i1 + i2', i1=wm, i2=gm)
    brain.to_filename('/tmp/brain_%s.nii.gz' % subject_id)
    large_brain = resample_to_img(brain, src)
    large_brain_data = grey_closing(large_brain.get_data(), 5)
    large_brain = Nifti1Image(large_brain_data, large_brain.get_affine())
    bet_img = math_img('i1 * i2', i1=large_brain, i2=src)
    bet_img.to_filename(dst)
