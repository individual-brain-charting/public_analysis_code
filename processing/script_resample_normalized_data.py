"""
This script resampled normalized data to a reference shape: (105, 127, 105)
"""
import glob
from nilearn.image import resample_to_img
from joblib import Parallel, delayed
import nibabel as nib
import os

SMOOTH_DERIVATIVES = '/neurospin/ibc/smooth_derivatives'
DERIVATIVES = '/neurospin/ibc/derivatives'
THREE_MM = '/neurospin/ibc/3mm'
_package_directory = os.path.dirname(os.path.abspath(__file__))


def _resample(img, reference, target=None):
    rimg = resample_to_img(img, reference)
    if target is not None:
        rimg.to_filename(target)
    else:
        rimg.to_filename(img)
    print(img)


def _prepare_targets(imgs):
    """Prepare target filenames"""
    targets = []
    for img in imgs:
        parts = img.split('/')
        subject_dir = os.path.join(THREE_MM, parts[-4])
        if not os.path.exists(subject_dir):
            print(subject_dir)
            os.mkdir(subject_dir)
        sess_dir = os.path.join(subject_dir, parts[-3])
        if not os.path.exists(sess_dir):
            print(sess_dir)
            os.mkdir(sess_dir)
        func_dir = os.path.join(sess_dir, 'func')
        if not os.path.exists(func_dir):
            print(func_dir)
            os.mkdir(func_dir)
        target = os.path.join(func_dir, os.path.basename(img))
        targets.append(target)
    return targets


def resample_func_data(n_jobs=2):
    reference = os.path.join(
        _package_directory, '../ibc_data', 'gm_mask_1_5mm.nii.gz')
    imgs = glob.glob(os.path.join(DERIVATIVES,
                     'sub-*', 'ses-*', 'func', 'wrdcsub-*.nii.gz'))
    Parallel(n_jobs=n_jobs)(
        delayed(_resample)(img, reference) for img in imgs
        if (nib.load(img).shape[2] != 105)
        and ('RestingState' not in img))


def resample_anat_data(n_jobs=2):
    reference = os.path.join(DERIVATIVES, 'sub-01', 'ses-00', 'anat',
                             'wsub-01_ses-00_T1w.nii.gz')  # FIXME
    imgs = glob.glob(os.path.join(DERIVATIVES, 'sub-*', 'ses-*', 'anat',
                     'mwc*sub-*_ses-*_T1w.nii.gz'))
    reference_shape = nib.load(reference).shape
    Parallel(n_jobs=n_jobs)(
        delayed(_resample)(img, reference) for img in imgs
        if nib.load(img).shape != reference_shape)


def resample_3mm_func_data(n_jobs=2):
    reference = os.path.join(
        _package_directory, '../ibc_data', 'gm_mask_3mm.nii.gz')
    wc = os.path.join(DERIVATIVES, 'sub-*/ses-*/func/wrdcsub-*.nii.gz')
    imgs = glob.glob(wc)
    targets = _prepare_targets(imgs)
    Parallel(n_jobs=n_jobs)(
        delayed(_resample)(img, reference, target) for (img, target) in
        zip(imgs, targets)
        if not os.path.exists(target))


def resample_func_and_anat(n_jobs=4):
    resample_func_data(n_jobs=n_jobs)
    resample_anat_data(n_jobs=n_jobs)
    resample_3mm_func_data(n_jobs=n_jobs)


if __name__ == '__main__':
    resample_func_and_anat()
