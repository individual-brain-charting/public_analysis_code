"""
This script creates diffusion tracts and highres anat images
to later plot with mrview - for IBC subjects
"""
from pypreprocess.nipype_preproc_spm_utils import (SubjectData,
                                                   _do_subject_segment,
                                                   _do_subject_coregister)
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from dipy.io.streamline import load_tck, save_tck, load_trk
from dipy.segment.clustering import QuickBundles

from dipy.align.streamlinear import whole_brain_slr
from dipy.segment.bundles import RecoBundles
from dipy.data import fetch_bundle_atlas_hcp842, get_bundle_atlas_hcp842
from dipy.io.utils import create_tractogram_header

from nilearn.image import math_img
from nilearn.masking import compute_epi_mask

from scipy import ndimage
from nilearn.image.image import new_img_like

def coreg(b0img, t1_img, output_dir, inverse):
    """ Wrapper for pypreprocess's coregister function
    for transforming highres t1 to dwi-space
    """

    data = SubjectData()
    data.anat = t1_img
    data.func = [b0img]
    data.output_dir = output_dir
    coreged = _do_subject_coregister(data,
                                    caching=False,
                                    hardlink_output=False,
                                    coreg_anat_to_func=inverse)
    return coreged

def segment(image, normalize):
    """ Wrapper for pypreprocess's segment function, for segmenting highres
        t1 used for subsequent masking.
    """    

    data = SubjectData()
    data.anat = image
    segmented = _do_subject_segment(data, caching=False,
                    normalize=normalize, hardlink_output=False)
    return segmented

def closing(image, iterations):
    """ Wrapper for scipy's binary_closing function to close a NIFTI image with
    n iterations
    """

    # extract voxel intensity array
    image_data = image.get_fdata()
    # extract the afffine matrix
    image_affine = image.affine
    # scipy closing
    closed_array = ndimage.binary_closing(image_data,
                     iterations=iterations).astype(int)

    closed_nifti_img = new_img_like(image, closed_array, image_affine)

    return closed_nifti_img

def cluster_bundles(workdir, sub, ses, outdir):
    """ This algorithm:
        * removes singleton fibers
        * outputs a colormap for remaining fibers
    """

    f = os.path.join(workdir, f'tracks_{sub}_{ses}_t1.tck')

    ref = os.path.join(workdir,
                    f'{sub}_{ses}_desc-denoise-eddy-correct_dwi.nii.gz')
    tract = load_tck(f, ref)

    atlas_file, atlas_folder = fetch_bundle_atlas_hcp842()
    atlas_file, all_bundles_files = get_bundle_atlas_hcp842()
    sft_atlas = load_trk(atlas_file, "same", bbox_valid_check=False)
    atlas = sft_atlas.streamlines
    atlas_header = create_tractogram_header(atlas_file,
                                            *sft_atlas.space_attributes)

    moved, transform, qb_centroids1, qb_centroids2 = whole_brain_slr(
        atlas, load_tck(f, ref).streamlines,
        x0='affine', verbose=True, progressive=True,
        rng=np.random.RandomState(1984))

    rb = RecoBundles(moved, verbose=True, rng=np.random.RandomState(2001))

    bundle_files = sorted(glob.glob(all_bundles_files))

    clusters = []
    for bf in bundle_files:
        model = load_trk(bf, "same", bbox_valid_check=False).streamlines
        recognized, label = rb.recognize(model_bundle=model,
                                        model_clust_thr=0.05,
                                        reduction_thr=10,
                                        pruning_thr=5,
                                        reduction_distance='mdf',
                                        pruning_distance='mdf',
                                        slr=True)
        clusters.append(label)

    n_fibers= len(moved)
    labels = np.zeros(n_fibers, dtype=int)
    for i, cluster in enumerate(clusters):
        labels[cluster] =  i + 1

    # skip empty clusters
    u, indices = np.unique(labels, return_inverse=True)
    labels_ = np.zeros(n_fibers, dtype=int)
    for i, v in enumerate(u):
        labels_[indices == v] = i

    tract.streamlines = tract.streamlines[labels_ > 0]
    labels_ = labels_[labels_ > 0]
    labels_ -= 1
    unique_labels = np.unique(labels_)
    np.random.seed(1)
    np.random.shuffle(unique_labels)
    labels_ = unique_labels[labels_]

    color_palette_path = os.path.join(outdir, 'palette.txt')
    np.savetxt(color_palette_path, labels_)

    tck_path = os.path.join(outdir, f'bundle-tracks-all_{sub}_{ses}.tck')
    print(save_tck(tract, tck_path, bbox_valid_check=True))

    return color_palette_path, tck_path
    

if __name__ == "__main__":

    DATA_ROOT = '/neurospin/ibc/derivatives'

    # dwi session numbers for each subject
    sub_ses = {'sub-01': 'ses-12', 'sub-04': 'ses-08', 'sub-05': 'ses-08',
               'sub-06': 'ses-09', 'sub-07': 'ses-09', 'sub-08': 'ses-09',
               'sub-09': 'ses-09', 'sub-11': 'ses-09', 'sub-12': 'ses-09',
               'sub-13': 'ses-09', 'sub-14': 'ses-05', 'sub-15': 'ses-12'}

    for sub, ses in sub_ses.items():

        workdir = os.path.join(DATA_ROOT, sub, ses)

        dwi_workdir = os.path.join(DATA_ROOT, sub, ses, 'dwi')

        highres_workdir = os.path.join(DATA_ROOT, sub, ses, 'anat')

        outdir = os.path.join(dwi_workdir, 'tract_plot_files')

        if not os.path.exists(outdir):
                os.makedirs(outdir)

        b0img_pth = os.path.join(dwi_workdir, 
                        f'{sub}_{ses}_desc-denoise-eddy-correct-b0_dwi.nii.gz')

        t1img_pth = os.path.join(highres_workdir,
                         f'{sub}_{ses}_acq-highres_T1w.nii')

        os.system(f'cp {b0img_pth} {outdir}')
        os.system(f'cp {t1img_pth} {outdir}')

        b0img_pth = os.path.join(outdir,
                     f'{sub}_{ses}_desc-denoise-eddy-correct-b0_dwi.nii.gz')
        t1img_pth = os.path.join(outdir, f'{sub}_{ses}_acq-highres_T1w.nii')

        os.system(f'gunzip -df {b0img_pth}')

        b0img_pth = os.path.join(outdir,
                         f'{sub}_{ses}_desc-denoise-eddy-correct-b0_dwi.nii')

        segmented = segment(t1img_pth, False)
        segments = [segmented['gm'], segmented['wm']]
        add = math_img("img1 + img2", img1=segments[0],
                       img2=segments[1])
        if sub=='sub-08':
            full = compute_epi_mask(add, exclude_zeros=True)
        else:
            full = compute_epi_mask(add)
        mask = closing(full, 10)
        mask_pth = os.path.join(outdir, f'{sub}_{ses}_mask.nii')
        # mask.to_filename(mask_pth)
        masked_t1 = math_img("img1 * img2", img1=t1img_pth,
                       img2=mask)
        t1img_pth = os.path.join(outdir, f'{sub}_{ses}_highres_T1w-masked.nii')
        masked_t1.set_data_dtype(float)
        masked_t1.to_filename(t1img_pth)

        coreged = coreg(b0img_pth, t1img_pth, outdir, True)

        t1img_pth_final = os.path.join(outdir,
                                 f'{sub}_{ses}_highres_T1w-masked-coreg.nii.gz')
        os.system(f'gzip -9 -c {t1img_pth} > {t1img_pth_final}')


        color_palette_path, tck_path = cluster_bundles(dwi_workdir,
                                                       sub, ses,  outdir)
