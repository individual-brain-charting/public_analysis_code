"""
This script resampled normalized data to a reference shape: (105, 127, 105)
"""
import glob
from nilearn.image import resample_to_img
from joblib import Parallel, delayed
import nibabel as nib


def resample(img, reference):
    rimg = resample_to_img(img, reference)
    rimg.to_filename(img)
    print(img)

reference = '/neurospin/ibc/smooth_derivatives/group/resampled_gm_mask.nii.gz'
imgs = glob.glob('/neurospin/ibc/derivatives/sub-*/ses-*/func/wrdcsub-*.nii.gz')

Parallel(n_jobs=6)(
    delayed(resample)(img, reference) for img in imgs
    if nib.load(img).shape[2] != 105)

reference = '/neurospin/ibc/derivatives/sub-01/ses-10/anat/wsub-01_ses-10_acq-highres_T1w.nii.gz'
imgs = glob.glob('/neurospin/ibc/derivatives/sub-*/ses-*/anat/mwc*sub-*_ses-*_acq-highres_T1w.nii.gz')

reference_shape = nib.load(reference).shape

Parallel(n_jobs=6)(
    delayed(resample)(img, reference) for img in imgs
    if nib.load(img).shape != reference_shape)
