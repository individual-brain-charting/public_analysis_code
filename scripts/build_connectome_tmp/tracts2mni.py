import os

def antsRegister_b0dwi2mni(mni, b0dwi, tmp_dir):
    cmd1 = f"antsRegistration --verbose 1 --dimensionality 3 --float 0 --output [{tmp_dir}/ants,{tmp_dir}/antsWarped.nii.gz,{tmp_dir}/antsInverseWarped.nii.gz] --interpolation Linear --use-histogram-matching 1 --winsorize-image-intensities [0.005,0.995] --transform Rigid[0.1] --metric CC[{mni},{b0dwi},1,4,Regular,0.1] --convergence [1000x500x250x100,1e-6,10] --shrink-factors 8x4x2x1 --smoothing-sigmas 3x2x1x0vox --transform Affine[0.1] --metric CC[{mni},{b0dwi},1,4,Regular,0.2] --convergence [1000x500x250x100,1e-6,10] --shrink-factors 8x4x2x1 --smoothing-sigmas 3x2x1x0vox --transform SyN[0.1,3,0] --metric CC[{mni},{b0dwi},1,4] --convergence [100x70x50x20,1e-6,10] --shrink-factors 4x2x2x1 --smoothing-sigmas 2x2x1x0vox -x [reference_mask.nii.gz,input_mask.nii.gz]"
    print(cmd1)
    os.system(cmd1)

def mrconvert_nifti2mif(nifti, mif, tmp_dir):
    cmd2 = f"mrconvert {nifti} {tmp_dir}/{mif} --force"
    print(cmd2)
    os.system(cmd2)

def warpinit_create_mni_invidentitywarp(mni_mif, inv_identity_warp, tmp_dir):
    cmd3 = f"warpinit {tmp_dir}/{mni_mif} {tmp_dir}/{inv_identity_warp}'[]'.nii"
    print(cmd3)
    os.system(cmd3)

def antsApplyTransforms_invidentitywarp2mni(b0dwi, tmp_dir):
    for warp in range(3):
        cmd4 = f"antsApplyTransforms -d 3 -e 0 -i {tmp_dir}/inv_identity_warp{warp}.nii -o {tmp_dir}/inv_mrtrix_warp{warp}.nii -r {b0dwi} -t '[{tmp_dir}/ants0GenericAffine.mat,1]' -t {tmp_dir}/ants1InverseWarp.nii.gz --default-value 2147483647"
        print(cmd4)
        os.system(cmd4)

def warpcorrect(tmp_dir):
    cmd5 = f"warpcorrect {tmp_dir}/inv_mrtrix_warp'[]'.nii {tmp_dir}/inv_mrtrix_warp_corrected.mif -marker 2147483647 -force"
    print(cmd5)
    os.system(cmd5)

def tcktransform_tract2mni(tck, mni_tck, tmp_dir):
    cmd8 = f"tcktransform {tck} {tmp_dir}/inv_mrtrix_warp_corrected.mif {mni_tck} -force"
    print(cmd8)
    os.system(cmd8)


if __name__ == "__main__":

    mni_nifti = 'mni_icbm152_t1_tal_nlin_sym_09a_converted.nii.gz'
    b0dwi = 'sub-04_ses-08_desc-denoise-eddy-correct-b0_dwi.nii.gz'
    tmp_dir = 'tmp'

    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    antsRegister_b0dwi2mni(mni_nifti, b0dwi, tmp_dir)

    mni_mif = 'mni_icbm152_t1_tal_nlin_sym_09a_converted.mif'
    mrconvert_nifti2mif(mni_nifti, mni_mif, tmp_dir)

    inv_identity_warp_prefix = 'inv_identity_warp'
    warpinit_create_mni_invidentitywarp(mni_mif, inv_identity_warp_prefix, tmp_dir)

    antsApplyTransforms_invidentitywarp2mni(b0dwi, tmp_dir)
    warpcorrect(tmp_dir)

    # # check if transformation has worked
    # cmd6 = f"mrtransform mni_icbm152_t1_tal_nlin_sym_09a_converted.nii -warp {tmp_dir}/inv_mrtrix_warp_corrected.mif {tmp_dir}/reference_warped.mif -force"
    # cmd7 = "mrview {tmp_dir}/reference_warped.mif -overlay.load {tmp_dir}/antsInverseWarped.nii.gz -overlay.opacity 0.3"

    # print(cmd6)
    # os.system(cmd6)

    # print(cmd7)
    # os.system(cmd7)

    tck = 'bundle-tracks-all_sub-04_ses-08.tck'
    mni_tck = f"mni_{tck}"
    tcktransform_tract2mni(tck, mni_tck, tmp_dir)


    # cmd9 = "tck2connectome -force -symmetric -zero_diagonal -scale_invnodevol mni-bundle-tracks-all_sub-04_ses-08.tck Schaefer2018_400Parcels_7Networks_order_FSLMNI152_1mm.nii.gz connectome.csv -out_assignments assignments.txt"

    # print(cmd9)
    # os.system(cmd9)