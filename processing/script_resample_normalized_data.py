"""
This script resampled normalized data to a reference shape: (105, 127, 105)
"""
import glob
from nilearn.image import resample_to_img
from joblib import Parallel, delayed
import nibabel as nib
import os

do_func = True
do_anat = True
do_3mm = True

def resample(img, reference, target=None):
    rimg = resample_to_img(img, reference)
    if target is not None:
        rimg.to_filename(target)
    else:
        rimg.to_filename(img)
    print(img)

reference = '/neurospin/ibc/smooth_derivatives/group/resampled_gm_mask.nii.gz'
imgs = glob.glob('/neurospin/ibc/derivatives/sub-*/ses-*/func/wrdcsub-*.nii.gz')

if do_func:
    Parallel(n_jobs=2)(
        delayed(resample)(img, reference) for img in imgs
        if (nib.load(img).shape[2] != 105)
        and ('RestingState' not in img))

reference = '/neurospin/ibc/derivatives/sub-01/ses-10/anat/wsub-01_ses-10_acq-highres_T1w.nii.gz'
imgs = glob.glob('/neurospin/ibc/derivatives/sub-*/ses-*/anat/mwc*sub-*_ses-*_acq-highres_T1w.nii.gz')
reference_shape = nib.load(reference).shape

if do_anat:
    Parallel(n_jobs=2)(
        delayed(resample)(img, reference) for img in imgs
        if nib.load(img).shape != reference_shape)


reference = '3mm_ref.nii.gz'
imgs = glob.glob('/neurospin/ibc/derivatives/sub-*/ses-*/func/wrdcsub-*.nii.gz')
targets = []
for img in imgs:
    parts = img.split('/')
    subject_dir = os.path.join('/neurospin/ibc/3mm/', parts[-4]) 
    if not os.path.exists(subject_dir):
        print(subject_dir)
        os.mkdir(subject_dir)
    sess_dir = os.path.join(subject_dir, parts[-3]) 
    if not os.path.exists(sess_dir):
        print(sess_dir)
        os.mkdir(sess_dir)
    func_dir = os.path.join(subject_dir, 'func') 
    if not os.path.exists(func_dir):
        print(func_dir)
        os.mkdir(func_dir)
    target = os.path.join(func_dir, os.path.basename(img))
    targets.append(target)

    
if do_3mm:
    Parallel(n_jobs=2)(
        delayed(resample)(img, reference, target) for (img, target) in
        zip(imgs, targets)
        if not os.path.exists(target))
