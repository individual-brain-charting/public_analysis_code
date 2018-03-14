"""
This script resampled normalized data to a reference shape: (105, 127, 105)
"""
import glob
from nilearn.image import resample_to_img
from joblib import Parallel, delayed
import nibabel as nib

reference = '/neurospin/ibc/smooth_derivatives/group/resampled_gm_mask.nii.gz'
imgs = glob.glob('/neurospin/ibc/derivatives/sub-*/ses-*/func/wrdcsub-*.nii.gz')

def resample(img, reference):
    rimg = resample_to_img(img, reference)
    rimg.to_filename(img)
    print(img)


Parallel(n_jobs=6)(
    delayed(resample)(img, reference) for img in imgs
    if nib.load(img).shape[2] != 105)
