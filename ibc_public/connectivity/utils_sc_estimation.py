"""Utility functions for structural connectivity estimation"""

import os
from nilearn.maskers import NiftiMasker


def antsRegister_b0dwi2mni(mni, b0dwi, tmp_dir):
    """Wrapper function for ANTs antsRegistration to register b0dwi to mni

    Parameters
    ----------
    mni : str
        path to mni nifti file
    b0dwi : str
        path to b0dwi nifti file
    tmp_dir : str
        path to temporary directory to store intermediate files
    """
    cmd1 = (
        f"antsRegistration --verbose 1 --dimensionality 3 --float 0 "
        f"--output [{tmp_dir}/ants_t2,{tmp_dir}/antsWarped_t2.nii.gz,{tmp_dir}/"
        f"antsInverseWarped_t2.nii.gz] "
        f"--interpolation Linear --use-histogram-matching 1 "
        f"--winsorize-image-intensities [0.005,0.995] "
        f"--transform Rigid[0.1] "
        f"--metric CC[{mni},{b0dwi},1,4,Regular,0.1] "
        f"--convergence [1000x500x250x100,1e-6,10] "
        f"--shrink-factors 8x4x2x1 "
        f"--smoothing-sigmas 3x2x1x0vox --transform Affine[0.1] "
        f"--metric CC[{mni},{b0dwi},1,4,Regular,0.2] "
        f"--convergence [1000x500x250x100,1e-6,10] "
        f"--shrink-factors 8x4x2x1 "
        f"--smoothing-sigmas 3x2x1x0vox --transform SyN[0.1,3,0] "
        f"--metric CC[{mni},{b0dwi},1,4] "
        f"--convergence [100x70x50x20,1e-6,10] --shrink-factors 4x2x2x1 "
        f"--smoothing-sigmas 2x2x1x0vox "
        f"-x [reference_mask_t2.nii.gz,input_mask_t2.nii.gz]"
    )
    print(cmd1)
    os.system(cmd1)


def mrconvert_nifti2mif(nifti, mif, tmp_dir):
    """Wrapper function for MRtrix mrconvert to convert nifti to mif

    Parameters

    nifti : str
        path to nifti file
    mif : str
        path to mif file
    tmp_dir : str
        path to temporary directory to store intermediate files
    """
    cmd2 = f"mrconvert {nifti} {tmp_dir}/{mif} --force"
    print(cmd2)
    os.system(cmd2)


def warpinit_create_mni_invidentitywarp(mni_mif, inv_identity_warp, tmp_dir):
    """Wrapper function for MRtrix warpinit to create mni inverse identity warp

    Parameters
    ----------
    mni_mif : str
        path to mni mif file
    inv_identity_warp : str
        path to inverse identity warp file
    tmp_dir : str
        path to temporary directory to store intermediate files
    """
    cmd3 = (
        f"warpinit {tmp_dir}/{mni_mif} "
        f"{tmp_dir}/{inv_identity_warp}'[]'_t2.nii -force"
    )
    print(cmd3)
    os.system(cmd3)


def antsApplyTransforms_invidentitywarp2mni(b0dwi, tmp_dir):
    """Wrapper function for ANTs antsApplyTransforms to apply inverse identity

    Parameters
    ----------
    b0dwi : str
        path to b0dwi nifti file
    tmp_dir : str
        path to temporary directory to store intermediate files
    """
    for warp in range(3):
        cmd4 = (
            f"antsApplyTransforms -d 3 -e 0 "
            f"-i {tmp_dir}/inv_identity_warp{warp}_t2.nii "
            f"-o {tmp_dir}/inv_mrtrix_warp{warp}_t2.nii -r {b0dwi} "
            f"-t '[{tmp_dir}/ants_t20GenericAffine.mat,1]' "
            f"-t {tmp_dir}/ants_t21InverseWarp.nii.gz "
            f"--default-value 2147483647"
        )
        print(cmd4)
        os.system(cmd4)


def warpcorrect(tmp_dir):
    """Wrapper function for MRtrix warpcorrect to correct inverse identity warp

    Parameters
    ----------
    tmp_dir : str
        path to temporary directory to store intermediate files
    """
    cmd5 = (
        f"warpcorrect {tmp_dir}/inv_mrtrix_warp'[]'_t2.nii "
        f"{tmp_dir}/inv_mrtrix_warp_corrected_t2.mif "
        f"-marker 2147483647 -force"
    )
    print(cmd5)
    os.system(cmd5)


def tcktransform_tract2mni(tck, mni_tck, tmp_dir):
    """Wrapper function for MRtrix tcktransform to transform tract to mni space

    Parameters
    ----------
    tck : str
        path to input .tck file
    mni_tck : str
        name of output transformed .tck file in mni space
    tmp_dir : str
        path to temporary directory to store intermediate files
    """
    cmd8 = (
        f"tcktransform {tck} {tmp_dir}/inv_mrtrix_warp_corrected_t2.mif "
        f"{mni_tck} -force"
    )
    print(cmd8)
    os.system(cmd8)


def apply_mask(img, mask, out_masked_img):
    """Wrapper function for NiftiMasker to apply mask to image

    Parameters
    ----------
    img : str
        path to input nifti image
    mask : str
        path to mask for the nifti image
    out_masked_img : str
        name of output masked image
    """
    masker = NiftiMasker(mask)
    masked_img_arr = masker.fit_transform(img)
    masked_img = masker.inverse_transform(masked_img_arr)
    masked_img.to_filename(out_masked_img)


def atlas2dwi(dwi_b0, mni_nifti, atlas_nifti, atlas_in_dwi_space):
    """Wrapper function for ANTs to warp atlas from mni space to dwi space

    Parameters
    ----------
    dwi_b0 : str
        path to dwi b0 nifti file
    mni_nifti : str
        path to mni nifti template
    atlas_nifti : str
        path to atlas nifti file
    atlas_in_dwi_space : str
        name of output atlas in dwi space
    """
    get_warp_cmd = (
        f"ANTS 3 -m PR'[{mni_nifti},{dwi_b0}, 1, 2]'"
        f" -o ANTS -r Gauss'[2,0]' -t SyN'[0.5]' -i 30x99x11"
        f" --use-Histogram-Matching"
    )

    apply_warp_cmd = (
        f"WarpImageMultiTransform 3 {atlas_nifti}"
        f" {atlas_in_dwi_space} -R {dwi_b0}"
        f" -i ANTSAffine.txt ANTSInverseWarp.nii --use-NN"
    )

    print(get_warp_cmd)
    os.system(get_warp_cmd)

    print(apply_warp_cmd)
    os.system(apply_warp_cmd)


def tck2connectome(
    atlas,
    tck,
    connectivity_matrix,
    inverse_connectivity_matrix,
    sift_weights=None,
):
    """Wrapper function for MRtrix tck2connectome to create connectivity matrix
    from tractography

    Parameters
    ----------
    atlas : str
        path to atlas nifti file
    tck : str
        path to input .tck file
    connectivity_matrix : str
        name of output connectivity matrix (.csv)
    inverse_connectivity_matrix : str
        name of output inverse connectivity matrix (.csv)
    sift_weights : str, optional
        path to sift weights in .txt format, by default None
    """
    if sift_weights is None:
        # if no sift weights are given, use tck2connectome without
        # -tck_weights_in parameter
        # -scale_invnodevol scales the connectivity matrix by inverse of the two
        #  ROI volumes involved in the connection
        cmd = (
            f"tck2connectome -force -symmetric -zero_diagonal "
            f"-scale_invnodevol {tck} {atlas} {connectivity_matrix} "
            f"-out_assignments {inverse_connectivity_matrix}"
        )
    else:
        # if sift weights are given, use tck2connectome with
        # -tck_weights_in parameter
        cmd = (
            f"tck2connectome -force -symmetric -zero_diagonal "
            f"-scale_invnodevol -tck_weights_in {sift_weights} "
            f"{tck} {atlas} {connectivity_matrix} -out_assignments "
            f"{inverse_connectivity_matrix}"
        )
    print(cmd)
    os.system(cmd)
