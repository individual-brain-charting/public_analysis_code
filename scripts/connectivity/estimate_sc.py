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
from joblib import Parallel, delayed
from ibc_public.connectivity.utils_sc_estimation import (
    antsRegister_b0dwi2mni,
    mrconvert_nifti2mif,
    warpinit_create_mni_invidentitywarp,
    antsApplyTransforms_invidentitywarp2mni,
    warpcorrect,
    tcktransform_tract2mni,
    atlas2dwi,
    tck2connectome,
    apply_mask,
)


def pipeline(sub, ses, data_root, out_root, atlas, mni_nifti):
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
    out_root : str
        root of the directory to store output files
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
    tmp_dir = os.path.join(out_root, sub, ses, "dwi")
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
        tmp_dir,
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
    # cache and output directory
    cache = OUT_ROOT = "/storage/store2/work/haggarwa/"
    # get atlas
    atlas = datasets.fetch_atlas_schaefer_2018(
        data_dir=cache, resolution_mm=1, n_rois=200
    )
    # give atlas a custom name
    atlas["name"] = "schaefer200"
    # download mni templates and mask
    mni = datasets.fetch_icbm152_2009()
    # path to unmasked mni t2w template
    mni_unmasked = mni["t2"]
    # path to mni mask to remove skull
    mni_mask = mni["mask"]
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
    results = Parallel(n_jobs=12, verbose=2, backend="loky")(
        delayed(pipeline)(sub, ses, DERIVATIVES, OUT_ROOT, atlas, mni_nifti)
        for sub, ses in sub_ses.items()
    )

    print(results)
