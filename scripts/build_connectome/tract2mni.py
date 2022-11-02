"""
This script transforms the tracts from native space to MNI
Following https://community.mrtrix.org/t/registration-using-transformations-generated-from-other-packages/2259
"""

import os
from nilearn import datasets

def antsRegister_b0dwi2mni(mni, b0dwi, tmp_dir):
    cmd1 = (f"antsRegistration --verbose 1 --dimensionality 3 --float 0 "
            f"--output [{tmp_dir}/ants,{tmp_dir}/antsWarped.nii.gz,{tmp_dir}/"
            f"antsInverseWarped.nii.gz] "
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
            f"-x [reference_mask.nii.gz,input_mask.nii.gz]")
    print(cmd1)
    os.system(cmd1)

def mrconvert_nifti2mif(nifti, mif, tmp_dir):
    cmd2 = f"mrconvert {nifti} {tmp_dir}/{mif} --force"
    print(cmd2)
    os.system(cmd2)

def warpinit_create_mni_invidentitywarp(mni_mif, inv_identity_warp, tmp_dir):
    cmd3 = (f"warpinit {tmp_dir}/{mni_mif}" 
            f"{tmp_dir}/{inv_identity_warp}'[]'.nii -force")
    print(cmd3)
    os.system(cmd3)

def antsApplyTransforms_invidentitywarp2mni(b0dwi, tmp_dir):
    for warp in range(3):
        cmd4 = (f"antsApplyTransforms -d 3 -e 0 "
               f"-i {tmp_dir}/inv_identity_warp{warp}.nii "
               f"-o {tmp_dir}/inv_mrtrix_warp{warp}.nii -r {b0dwi} "
               f"-t '[{tmp_dir}/ants0GenericAffine.mat,1]' "
               f"-t {tmp_dir}/ants1InverseWarp.nii.gz "
               f"--default-value 2147483647")
        print(cmd4)
        os.system(cmd4)

def warpcorrect(tmp_dir):
    cmd5 = (f"warpcorrect {tmp_dir}/inv_mrtrix_warp'[]'.nii "
            f"{tmp_dir}/inv_mrtrix_warp_corrected.mif "
            f"-marker 2147483647 -force")
    print(cmd5)
    os.system(cmd5)

def tcktransform_tract2mni(tck, mni_tck, tmp_dir):
    cmd8 = (f"tcktransform {tck} {tmp_dir}/inv_mrtrix_warp_corrected.mif "
            f"{mni_tck} -force")
    print(cmd8)
    os.system(cmd8)

if __name__ == "__main__":

    DATA_ROOT = '/data/parietal/store2/data/ibc/derivatives/'

    sub_ses = {'sub-01': 'ses-12', 'sub-04': 'ses-08', 'sub-05': 'ses-08',
               'sub-06': 'ses-09', 'sub-07': 'ses-09', 'sub-08': 'ses-09',
               'sub-09': 'ses-09', 'sub-11': 'ses-09', 'sub-12': 'ses-09',
               'sub-13': 'ses-09', 'sub-14': 'ses-05', 'sub-15': 'ses-12'}

    for sub, ses in sub_ses.items():

        tmp_dir = os.path.join(DATA_ROOT, sub, ses, 'dwi', 'tract2mni_tmp')

        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)

        mni = datasets.load_mni152_template(resolution=1)
        mni_nifti = os.path.join(tmp_dir, 'mni_t1w.nii.gz')
        mni.to_filename(mni_nifti)

        b0dwi = os.path.join(DATA_ROOT, sub, ses, 'dwi',
                             (f'{sub}_{ses}_desc-denoise-eddy-correct-b0_dwi'
                              f'.nii.gz'))

        antsRegister_b0dwi2mni(mni_nifti, b0dwi, tmp_dir)

        mni_mif = 'mni_t1w.mif'
        mrconvert_nifti2mif(mni_nifti, mni_mif, tmp_dir)

        inv_identity_warp_prefix = 'inv_identity_warp'
        warpinit_create_mni_invidentitywarp(mni_mif, inv_identity_warp_prefix,
                                            tmp_dir)

        antsApplyTransforms_invidentitywarp2mni(b0dwi, tmp_dir)
        warpcorrect(tmp_dir)

        tck = os.path.join(DATA_ROOT, sub, ses, 'dwi',
                           f'tracks_{sub}_{ses}_t1.tck')
        mni_tck = os.path.join(DATA_ROOT, sub, ses, 'dwi',
                               f'mni-tracks_{sub}_{ses}_t1.tck')
        tcktransform_tract2mni(tck, mni_tck, tmp_dir)
