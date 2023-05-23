"""
This script:
1) transforms the diffusion tracts from native space to MNI
Following https://community.mrtrix.org/t/registration-using-transformations-generated-from-other-packages/2259
2) transforms atlas to diffusion native space
3) creates structural connectivity matrix between ROIs from a given
atlas (both in mni and transformed to native space) for given diffusion tracts 
(both transformed to mni and native space) using MRtrix's tck2connectome function.
Connectivity between two ROIs here is measured as per-bundle sum of SIFT2
weights normalised by ROI volume (-scale_invnodevol parameter)
See tck2connectome doc: https://mrtrix.readthedocs.io/en/latest/reference/commands/tck2connectome.html#tck2connectome
Discussion on using original SIFT2 weights (calculated before non-linear 
transformation of tracts): https://community.mrtrix.org/t/are-sift2-weights-still-interpretable-following-non-linear-transformation/6162
"""
import os
from nilearn import datasets
from ibc_public.utils_data import get_subject_session, DERIVATIVES
from nilearn.maskers import NiftiMasker
from joblib import Parallel, delayed


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


def fetch_mni_template(tmp_dir):
    """Fetch MNI template and mask

    Parameters
    ----------
    tmp_dir : str
        where to store the downloaded files
    """
    # download mni templates and mask
    mni_templates_link = (
        "http://www.bic.mni.mcgill.ca/~vfonov/"
        "icbm/2009/mni_icbm152_nlin_sym_09a_nifti.zip"
    )
    os.system(f"wget {mni_templates_link} -P {tmp_dir}")
    # unzip mni templates and mask
    os.system(
        f"unzip -o {tmp_dir}/mni_icbm152_nlin_sym_09a_nifti.zip -d {tmp_dir}"
    )


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


def pipeline(sub, ses, data_root, atlas, mni_nifti):
    """Pipeline for creating connectivity matrices from tractography in MNI
    as well as in individual space

    Parameters
    ----------
    sub : str
        subject id
    ses : str
        session id
    data_root : str
        path to data root directory
    atlas : sklearn.utils.Bunch
        Dictionary-like object, contains:
        - 'maps': `str`, path to nifti file containing the
        3D ~nibabel.nifti1.Nifti1Image (its shape is (182, 218, 182)). The
        values are consecutive integers between 0 and n_rois which can be
        interpreted as indices in the list of labels.
        - 'labels': numpy.ndarray of str, array containing the ROI labels
        including Yeo-network annotation.
        - 'description': `str`, short description of the atlas
          and some references.
        _description_
    mni_nifti : str
        path to mni nifti template

    Returns
    -------
    str
        status message
    """
    # setup tmp dir for saving intermediate files
    tmp_dir = os.path.join(data_root, sub, ses, "dwi", "connectivity_tmp")
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    # directory with dwi data
    dwi_dir = os.path.join(data_root, sub, ses, "dwi")

    ######## start transform diffusion images to diffusion mni space ########
    # path to diffusion b0 image
    b0dwi = os.path.join(
        dwi_dir,
        f"{sub}_{ses}_desc-denoise-eddy-correct-b0_dwi.nii.gz",
    )
    # register b0 to mni t2w template
    antsRegister_b0dwi2mni(mni_nifti, b0dwi, tmp_dir)
    # path to mni t2w template in mif format
    mni_mif = "mni_t2w.mif"
    mrconvert_nifti2mif(mni_nifti, mni_mif, tmp_dir)
    # create mni inverse identity warp
    inv_identity_warp_prefix = "inv_identity_warp"
    warpinit_create_mni_invidentitywarp(
        mni_mif, inv_identity_warp_prefix, tmp_dir
    )
    # apply inverse identity warp to b0
    antsApplyTransforms_invidentitywarp2mni(b0dwi, tmp_dir)
    # correct for warping
    warpcorrect(tmp_dir)
    # path to diffusion tractogram in native space
    dwi_tck = os.path.join(
        dwi_dir,
        f"tracks_{sub}_{ses}_t1.tck",
    )
    # path to diffusion tractogram in mni space
    mni_tck = os.path.join(
        dwi_dir,
        f"mni-tracks_{sub}_{ses}_t2.tck",
    )
    # transform diffusion tractogram to mni space
    tcktransform_tract2mni(dwi_tck, mni_tck, tmp_dir)

    ######## start transform atlas to diffusion native space ########
    # name of atlas in diffusion native space
    atlas_in_dwi_space = os.path.join(
        tmp_dir, (f"{atlas.name}_in_dwi_space.nii.gz")
    )
    # transform atlas to diffusion native space
    atlas2dwi(b0dwi, mni_nifti, atlas.maps, atlas_in_dwi_space)

    ######## start of connectivity matrix creation ########
    # iterate over spaces
    for space in ["MNI152", "individual"]:
        if space == "MNI152":
            # use tractogram in mni space
            tck = mni_tck
        else:
            # use original tractogram in diffusion native space
            tck = dwi_tck
            # use transformed atlas in diffusion native space
            atlas.maps = atlas_in_dwi_space
        # calculate connectivity matrices with and without sift weights
        for sift in ["siftweighted", "nosift"]:
            if sift == "siftweighted":
                # file path for previously calculated sift weights
                sift_weights = os.path.join(
                    dwi_dir,
                    f"sift-track_{sub}_{ses}.txt",
                )
            else:
                sift_weights = None
            # name for output connectivity matrix
            connectivity_matrix = os.path.join(
                tmp_dir,
                f"{sub}_{ses}_Diffusion_connectome_{atlas.name}_{space}_"
                f"{sift}.csv",
            )
            # name for output inverse connectivity matrix
            inverse_connectivity_matrix = os.path.join(
                tmp_dir,
                f"{sub}_{ses}_Diffusion_invconnectome_{atlas.name}_{space}_"
                f"{sift}.csv",
            )
            # calculate and save the connectivity and inv connectivity
            # matrices
            tck2connectome(
                atlas.maps,
                tck,
                connectivity_matrix,
                inverse_connectivity_matrix,
                sift_weights=sift_weights,
            )

    return f"{sub} Done!"


if __name__ == "__main__":
    # cache directory
    cache = "/storage/store/work/haggarwa/"
    # get atlas
    atlas = datasets.fetch_atlas_schaefer_2018(
        data_dir=cache, resolution_mm=1, n_rois=400
    )
    # give atlas a custom name
    atlas["name"] = "schaefer400"
    # download mni templates and mask
    fetch_mni_template(cache)
    # path to mni t2w template
    mni_unmasked = os.path.join(
        cache,
        "mni_icbm152_nlin_sym_09a",
        "mni_icbm152_t2_tal_nlin_sym_09a.nii",
    )
    # path to mni t1w template mask
    mni_mask = os.path.join(
        cache,
        "mni_icbm152_nlin_sym_09a",
        "mni_icbm152_t1_tal_nlin_sym_09a_mask.nii",
    )
    # path to masked mni t2w
    mni_nifti = os.path.join(cache, "mni_t2w.nii.gz")
    # apply mask to mni t2w template
    apply_mask(mni_unmasked, mni_mask, mni_nifti)
    # get sessions with diffusion data
    subject_sessions = sorted(get_subject_session("anat1"))
    sub_ses = {
        subject_session[0]: "ses-12"
        if subject_session[0] in ["sub-01", "sub-15"]
        else subject_session[1]
        for subject_session in subject_sessions
    }
    results = Parallel(n_jobs=6, verbose=1, backend="multiprocessing")(
        delayed(pipeline)(sub, ses, DERIVATIVES, atlas, mni_nifti)
        for sub, ses in sub_ses.items()
    )

    print(results)
