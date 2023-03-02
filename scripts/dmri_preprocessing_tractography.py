"""
Author: Bertrand Thirion, 2015
Modified by: Swetha Shankar, 2021
"""

import os
import glob
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from ibc_public.utils_data import get_subject_session
from joblib import Memory, Parallel, delayed

source_dir = '/storage/store2/data/ibc/sourcedata'
fs_derivatives_dir = '/storage/store2/data/ibc/derivatives'
derivatives_dir = '/storage/store2/work/sshankar/diffusion/derivatives'
# subjects_sessions = [('sub-01', 'ses-12'), ('sub-04', 'ses-08'), ('sub-05', 'ses-08'), ('sub-06', 'ses-09'), ('sub-07', 'ses-09'), ('sub-08', 'ses-09'), ('sub-09', 'ses-09'), ('sub-11', 'ses-09'), ('sub-12', 'ses-09'), ('sub-13', 'ses-09'), ('sub-14', 'ses-05'), ('sub-15', 'ses-12')]
subjects_sessions = [('sub-14', 'ses-05')]

def concat_images(in_imgs, out_img):
    nib.nifti1.save(nib.funcs.concat_images(in_imgs, axis=3), out_img)

def concat_bvals(in_bvals, out_bvals):
    bvals_ = np.loadtxt(in_bvals[0], dtype=int)
    for i in range(len(in_bvals)-1):
        bv = np.loadtxt(in_bvals[i+1], dtype=int)
        bvals_ = np.concatenate((bvals_,bv))
    np.savetxt(out_bvals, bvals_, fmt='%d', newline=' ')

def concat_bvecs(in_bvecs, out_bvecs):
    bvecs_ = np.loadtxt(in_bvecs[0])
    for i in range(len(in_bvecs)-1):
        bv = np.loadtxt(in_bvecs[i+1])
        bvecs_ = np.concatenate((bvecs_,bv), axis=1)
    np.savetxt(out_bvecs, bvecs_)

def denoise_dwi(in_dn, out_dn):
    cmd = 'dwidenoise %s %s' % (in_dn, out_dn)
    print(cmd)
    os.system(cmd)

def degibbs_dwi(in_dg, out_dg):
    cmd = 'mrdegibbs %s %s' % (in_dg, out_dg)
    print(cmd)
    os.system(cmd)

def collate_b0s(b0_imgs, vols, merged_b0_img):
    for i, vol in enumerate(vols):
        if i==0:
            cmd = "fslroi %s %s %d 1" % (b0_imgs, merged_b0_img, vol)
            print(cmd)
            os.system(cmd)
        else:
            cmd = "fslroi %s temp_vol %d 1" % (b0_imgs, vol)
            print(cmd)
            os.system(cmd)
            cmd = "fslmerge -t %s %s temp_vol" % (merged_b0_img, merged_b0_img)
            print(cmd)
            os.system(cmd)

    cmd = "rm temp_vol*"
    print(cmd)
    os.system(cmd)

def make_acq_param_file(acq_params_file):
    acqp = np.array([[0.0, 1.0, 0.0, 0.1],
                    [0.0, 1.0, 0.0, 0.1],
                    [0.0, -1.0, 0.0, 0.1],
                    [0.0, -1.0, 0.0, 0.1]])
    np.savetxt(acq_params_file, acqp, fmt='%0.1f', delimiter=' ')

def calc_topup(merged_b0_img, acq_params_file, topup_results_basename, iout_file):
    cmd = "topup --imain=%s --datain=%s --config=b02b0.cnf --out=%s --iout=%s" % (
        merged_b0_img, acq_params_file, topup_results_basename, iout_file)
    print(cmd)
    os.system(cmd)

def extract_brain(in_brain, out_brain, threshold, r):
    if r is None:
        cmd = "bet %s %s -f %s -R -m" % (in_brain, out_brain, str(threshold))
    else:
        cmd = "bet %s %s -f %s -r %s -R -m" % (in_brain, out_brain, str(threshold), str(r))
    print(cmd)
    os.system(cmd)

def calc_mask(hifi_file, hifi_brain, threshold=0.5, r=None):
    cmd = "fslmaths %s -Tmean temp" % (hifi_file)
    print(cmd)
    os.system(cmd)
    extract_brain('temp', hifi_file, threshold, r)

def make_acqdir_file(index_file, nvols):
    inds = np.concatenate((np.ones(nvols*2, dtype=int), np.ones(nvols*2, dtype=int)*3))
    np.savetxt(index_file, inds, fmt='%d')

def run_eddy(eddy_in, mask_img, acq_params_file, index_file, out_bvecs, out_bvals, topup_results_basename, eddy_out):
    cmd = "eddy_cuda9.1 --imain=%s --mask=%s --acqp=%s --index=%s --bvecs=%s --bvals=%s --topup=%s --repol --out=%s" % (
        eddy_in, mask_img, acq_params_file, index_file, out_bvecs, out_bvals, topup_results_basename, eddy_out)

    print(cmd)
    os.system(cmd)

def bias_correct():
    # Correct for negative values and values close to 0 prior to bias correction
    # For more details see ANTs documentation for N4BiasFieldCorrection
    eddy_out = glob.glob('%s/eddy_denoise.nii.gz' % dwi_dir)[0]
    img_ = nib.load(eddy_out)
    data_ = img_.get_fdata()
    min_val = np.min(np.min(np.min(data_)))
    data_ = data_ + np.abs(min_val) + (0.1 * np.abs(min_val))

    nonneg_file = os.path.join(dwi_dir, 'nn_eddy_denoise.nii.gz')
    nib.save(nib.Nifti1Image(data_, img_.affine), nonneg_file)
    eddy_out = glob.glob('%s/nn_eddy_denoise.nii.gz' % dwi_dir)[0]
    bf_out = os.path.join(dwi_dir, 'ants_bf_nn_eddy_denoise.nii.gz')
    cmd = "dwibiascorrect ants %s %s -mask %s -fslgrad %s %s" % (
        eddy_out, bf_out, mask_img, out_bvecs, out_bvals)
    cmd = "N4BiasFieldCorrection -d 4 -i %s -w %s -o %s -s 2 -b [150] -c [200x200,0.0]" % (
        eddy_out, mask_img, bf_out)
    print(cmd)
    os.system(cmd)

def convert_mif(in_file, out_file):
    cmd = 'mrconvert %s %s' % (in_file, out_file)
    print(cmd)
    os.system(cmd)

def calc_resp_fn(eddy_mif, wm_out, gm_out, csf_out, bvecs, bvals, mask, voxels_out, algorithm='dhollander'):
    cmd = 'dwi2response %s %s %s %s %s -fslgrad %s %s -mask %s -voxels %s' % (algorithm, eddy_mif, wm_out, gm_out, csf_out, bvecs, bvals, mask, voxels_out)
    print(cmd)
    os.system(cmd)

def create_fod(eddy_mif, bvecs, bvals, mask, wm_out, wm_fod, gm_out, gm_fod, csf_out, csf_fod):
    # Uses multi-shell multi-tissue constrained spherical deconvolution
    cmd = 'dwi2fod msmt_csd %s -fslgrad %s %s -mask %s -force %s %s %s %s %s %s' % (eddy_mif, bvecs, bvals, mask, wm_out, wm_fod, gm_out, gm_fod, csf_out, csf_fod)
    print(cmd)
    os.system(cmd)

def remove_nans(t1_in, t1_out):
    cmd = 'fslmaths %s -nan %s' % (t1_in, t1_out)
    print(cmd)
    os.system(cmd)

def seg_anat(t1_mif, seg_out, algorithm='fsl'):
    cmd = '5ttgen %s %s %s' % (algorithm, t1_mif, seg_out)
    print(cmd)
    os.system(cmd)

def mif_to_nii(seg_out_mif, seg_out_nii):
    cmd = 'mrconvert %s %s' % (seg_out_mif, seg_out_nii)
    print(cmd)
    os.system(cmd)

def extract_gm_vol(seg_out, gm_vol):
    cmd = 'fslroi %s %s 0 1' % (seg_out, gm_vol)
    print(cmd)
    os.system(cmd)

def xform_dwi_t1(mean_b0, t1_nifti, transform_mat):
    cmd = 'flirt -in %s -ref %s -omat %s' % (mean_b0, t1_nifti, transform_mat)
    print(cmd)
    os.system(cmd)

def convert_xform_mat(transform_mat, mean_b0, t1_nifti, transform_mat_mrtrix):
    cmd = 'transformconvert %s %s %s flirt_import %s' % (transform_mat, mean_b0, t1_nifti, transform_mat_mrtrix)
    print(cmd)
    os.system(cmd)

def reg_to_dwi(seg_out, transform_mat_mrtrix, seg_out_coreg, dtype='float32'):
    cmd = 'mrtransform %s -linear %s -inverse -datatype %s %s' % (seg_out, transform_mat_mrtrix, dtype, seg_out_coreg)
    print(cmd)
    os.system(cmd)

def gen_gmwm_boundary(seg_out_coreg, gmwm_bound):
    cmd = '5tt2gmwmi %s %s' % (seg_out_coreg, gmwm_bound)
    print(cmd)
    os.system(cmd)

def gen_streamlines(seg_out_coreg, gmwm_bound, wm_fod, track_out, max_len=250, cutoff=0.06, nstreamlines=10000000):
    cmd = 'tckgen -act %s -backtrack -seed_gmwmi %s -maxlength %s -cutoff %s -select %s %s %s' % (seg_out_coreg, gmwm_bound, str(max_len), str(cutoff), str(nstreamlines), wm_fod, track_out)
    print(cmd)
    os.system(cmd)

def sift_tracts(seg_out_coreg, out_mu, out_coeff, track_out, wm_fod, out_sift):
    cmd = 'tcksift2 -act %s -out_mu %s -out_coeffs %s %s %s %s' % (seg_out_coreg, out_mu, out_coeff, track_out, wm_fod, out_sift)
    print(cmd)
    os.system(cmd)

def labelconvert(fs_label, fs_clut, mrtrix_clut, parcels):
    cmd = 'labelconvert %s %s %s %s' % (fs_label, fs_clut, mrtrix_clut, parcels)
    print(cmd)
    os.system(cmd)

def gen_connectome(out_sift, track_out, coreg_parcel, coreg_parcel_csv, inverse_coreg_parcel):
    cmd = 'tck2connectome -symmetric -zero_diagonal -scale_invnodevol -tck_weights_in %s %s %s %s -out_assignment %s' % (out_sift, track_out, coreg_parcel, coreg_parcel_csv, inverse_coreg_parcel)
    print(cmd)
    os.system(cmd)

def plot_connectome(coreg_parcel_csv, out_img, format):
    img = np.loadtxt(coreg_parcel_csv, delimiter=',')
    plt.figure(figsize=(12, 12))
    plt.imshow(img, interpolation='nearest', vmin=0, vmax=1, cmap=plt.cm.Reds_r)
    plt.colorbar()
    plt.savefig(out_img, format=format)

def run_dmri_preproc(subject_session):
    subject, session = subject_session

    src_dir = os.path.join(source_dir, subject, session)
    fs_src_dir = os.path.join(fs_derivatives_dir, subject, 'ses-00', 'anat')
    dest_dir = os.path.join(derivatives_dir, subject, session)

    src_anat_dir = os.path.join(src_dir, 'anat')
    src_dwi_dir = os.path.join(src_dir, 'dwi')
    src_fmap_dir = os.path.join(src_dir, 'fmap')

    dest_dwi_dir = os.path.join(dest_dir, 'dwi')
    dest_fmap_dir = os.path.join(dest_dir, 'fmap')
    dest_anat_dir = os.path.join(dest_dir, 'anat')

    if not os.path.exists(dest_dwi_dir):
        os.makedirs(dest_dwi_dir)

    if not os.path.exists(dest_fmap_dir):
        os.makedirs(dest_fmap_dir)

    if not os.path.exists(dest_anat_dir):
        os.makedirs(dest_anat_dir)

    # Concatenate all DWI images (4 runs in our case)
    dwi_imgs = sorted(glob.glob(os.path.join(src_dwi_dir, '*run*dwi.nii.gz')))
    out_concat = os.path.join(dest_dwi_dir, '%s_%s_dwi.nii.gz' % (subject, session))
    concat_images(dwi_imgs, out_concat)

    # Concatenate the bval and bvec files as well
    in_bvals = sorted(glob.glob(os.path.join(src_dwi_dir, '*run*dwi.bval')))
    out_bvals = os.path.join(dest_dwi_dir, '%s_%s_dwi.bvals' % (subject, session))
    concat_bvals(in_bvals, out_bvals)

    in_bvecs = sorted(glob.glob(os.path.join(src_dwi_dir, '*run*dwi.bvec')))
    out_bvecs = os.path.join(dest_dwi_dir, '%s_%s_dwi.bvecs' % (subject, session))
    concat_bvecs(in_bvecs, out_bvecs)

    # Denoise images using MP-PCA
    out_dn = os.path.join(dest_dwi_dir, '%s_%s_desc-denoise_dwi.nii.gz' % (subject, session))
    denoise_dwi(out_concat, out_dn)

    # Remove Gibbs ringing artifacts
    # The recommendation is to do this if not using partial Fourier acquisition, but we do use it.
    # The images look a little blurred and of lower intensity than the denoised images
    # out_dg = os.path.join(dest_dwi_dir, 'dg_%s_%s_dwi.nii.gz' % (subject, session))
    # degibbs_dwi(out_dn, out_dg)

    # Run FSL topup - it's a 2-step process
    # 1. Collect all the b=0 volumes in one file and use that as input to topup
    b0_imgs = sorted(glob.glob('%s/%s_%s_desc-denoise_dwi.nii.gz' % (dest_dwi_dir, subject, session)))[0]
    merged_b0_img = os.path.join(dest_dwi_dir, '%s_%s_desc-collated-b0s_dwi.nii.gz' % (subject, session))
    vols = [0, 61, 122, 183]
    collate_b0s(b0_imgs, vols, merged_b0_img)

    # 2. Calculate distortion from the collated b0 images
    acq_params_file = os.path.join(dest_fmap_dir, '%s_%s_desc-b0_acq_param_dwi.txt' % (subject, session))
    topup_results_basename = os.path.join(dest_fmap_dir, '%s_%s_desc-topup-results' % (subject, session))
    iout_file = os.path.join(dest_dwi_dir, '%s_%s_desc-unwarped-nomotion-b0_dwi' % (subject, session))
    make_acq_param_file(acq_params_file)
    calc_topup(merged_b0_img, acq_params_file, topup_results_basename, iout_file)

    # Create mask file for use with eddy
    topup_brain = os.path.join(dest_dwi_dir, '%s_%s_desc-unwarped-nomotion-b0_dwi.nii.gz' % (subject, session))
    if subject == 'sub-08':
        threshold = 0.8
        r = 105
    elif subject == 'sub-12':
        threshold = 0.75
        r = None
    elif subject == 'sub-14':
        threshold = 0.75
        r = 105
    else:
        threshold = 0.5
        r = None

    calc_mask(iout_file, topup_brain, threshold, r)

    # Create a text file that contains, for each volume in the concatenated dwi images file,
    # the corresponding line of the acquisition parameters file.
    # The way the data has been concatenated, we have 2 AP runs followed by 2 PA runs,
    # each with 61 volumes. Thus, the text file will have 244 lines, the first 122 will
    # say "1" and the last 122 will say "3"
    index_file = os.path.join(dest_dwi_dir, '%s_%s_desc-acqdir-index_dwi.txt')
    nvols = 61
    make_acqdir_file(index_file, nvols)

    # Now run eddy to correct eddy current distortions
    mask_img = glob.glob('%s/%s_%s_hifi*mask.nii.gz' % (dest_dwi_dir, subject, session))[0]
    eddy_in = out_dn
    eddy_out = os.path.join(dest_dwi_dir, '%s_%s_desc-denoise-eddy-correct_dwi.nii.gz' % (subject, session))
    run_eddy(eddy_in, mask_img, acq_params_file, index_file, out_bvecs, out_bvals, topup_results_basename, eddy_out)

    # Once again extract the b0 volumes, this time from the eddy corrected images,
    # create a mean volume, and a mask of the mean volume
    b0_imgs = glob.glob('%s/%s_%s_desc-denoise-eddy-correct_dwi.nii.gz' % (dest_dwi_dir, subject, session))[0]
    merged_b0_img = os.path.join(dest_dwi_dir, '%s_%s_desc-denoise-eddy-correct-b0_dwi.nii.gz' % (subject, session))
    collate_b0s(b0_imgs, vols, merged_b0_img)
    b0_brain = os.path.join(dest_dwi_dir, '%s_%s_desc-denoise-eddy-correct-b0_dwi' % (subject, session))
    calc_mask(merged_b0_img, b0_brain, threshold, r)

    # Bias field correction
    # Bias field correction doesn't work very well via dwibiascorrect, and
    # fails when I try running N4BiasFieldCorrection. Skipping this step for now.
    # bias_correct()

    ### Preprocessing is now complete, start the tractography part.

    # Convert DWI files to mif format
    eddy_in = glob.glob('%s/%s_%s_desc-denoise-eddy-correct_dwi.nii.gz' % (dest_dwi_dir, subject, session))[0]
    eddy_mif = os.path.join(dest_dwi_dir, '%s_%s_desc-denoise-eddy-correct_dwi.mif' % (subject, session))
    convert_mif(eddy_in, eddy_mif)

    # Derive response functions for the different tissue types from diffusion data using the Dhollander algorithm.
    # The wm, gm and csf txt files contain the response functions for those tissue types.
    # These are all generated using the dwi2response function
    algorithm = 'dhollander'
    bvecs = glob.glob('%s/%s_%s_dwi.bvecs' % (dest_dwi_dir, subject, session))[0]
    bvals = glob.glob('%s/%s_%s_dwi.bvals' % (dest_dwi_dir, subject, session))[0]
    mask = glob.glob('%s/%s_%s_desc-denoise-eddy-correct_dwi_mask.nii.gz' % (dest_dwi_dir, subject, session))[0]
    wm_out = os.path.join(dest_dwi_dir, '%s_%s_desc-wm-resp-fn_dwi.txt' % (subject, session))
    gm_out = os.path.join(dest_dwi_dir, '%s_%s_desc-gm-resp-fn_dwi.txt' % (subject, session))
    csf_out = os.path.join(dest_dwi_dir, '%s_%s_desc-csf-resp-fn_dwi.txt' % (subject, session))
    voxels_out = os.path.join(dest_dwi_dir, '%s_%s_desc-resp-fn-voxel-selection_dwi.mif' % (subject, session))
    calc_resp_fn(eddy_mif, wm_out, gm_out, csf_out, bvecs, bvals, mask, voxels_out, algorithm)
    # Use mrview to visualize the voxels file to make sure voxels are in the correct tissue group.
    # Red markers should be in CSF, Green markers should be in gray matter, Blue markers should be in white matter
    # View the basis functions files using shview.

    # Using the response functions we can create the FODs, or fiber orientation densities. These are estimates of the amount of diffusion in the 3 orthogonal directions.
    wm_fod = os.path.join(dest_dwi_dir, 'wm-fod_%s_%s_dwi.mif' % (subject, session))
    gm_fod = os.path.join(dest_dwi_dir, 'gm-fod_%s_%s_dwi.mif' % (subject, session))
    csf_fod = os.path.join(dest_dwi_dir, 'csf-fod_%s_%s_dwi.mif' % (subject, session))
    create_fod(eddy_mif, bvecs, bvals, mask, wm_out, wm_fod, gm_out, gm_fod, csf_out, csf_fod)

    # Remove NaNs from the anatomical image
    t1_nifti = glob.glob('%s/*T1w.nii.gz' % (fs_src_dir))[0]
    t1_nonan = os.path.join(dest_anat_dir, '%s_%s_T1w.nii.gz' % (subject, session))
    remove_nans(t1_nifti, t1_nonan)

    # Convert the anatomical to MRtrix format
    t1_mif = os.path.join(dest_anat_dir, '%s_%s_T1w.mif' % (subject, session))
    convert_mif(t1_nonan, t1_mif)

    # Segment anatomical into individual tissue types using FSL (other options available)
    seg_out = os.path.join(dest_dwi_dir, 'seg_%s_%s_T1w.mif' % (subject, session))
    algorithm = 'fsl'
    seg_anat(t1_mif, seg_out, algorithm)

    # Extract the gray matter volume from the segmented volumes after converting the segmented file
    # from mif to nii.gz
    seg_out_nii = os.path.join(dest_dwi_dir, 'seg_%s_%s_T1w.nii.gz' % (subject, session))
    mif_to_nii(seg_out, seg_out_nii)
    gm_vol = os.path.join(dest_dwi_dir, 'seg_%s_%s_gm.nii.gz' % (subject, session))
    extract_gm_vol(seg_out_nii, gm_vol)

    # Align the mean b0 image to the anatomical using FLIRT
    mean_b0 = glob.glob('%s/b0s_eddy_dn_%s_%s_dwi.nii.gz' % (dest_dwi_dir, subject, session))[0]
    transform_mat = os.path.join(dest_dwi_dir, '%s_%s_dwi-to-T1.mat' % (subject, session))
    xform_dwi_t1(mean_b0, gm_vol, transform_mat)

    # Convert the transformation matrix to MRtrix format
    transform_mat_mrtrix = os.path.join(dest_dwi_dir, '%s_%s_dwi-to-T1.txt' % (subject, session))
    convert_xform_mat(transform_mat, mean_b0, gm_vol, transform_mat_mrtrix)

    # Apply the transformation matrix to the non-coregistered segmentation data:
    seg_out_coreg = os.path.join(dest_dwi_dir, 'coreg-seg_%s_%s_gm.mif' % (subject, session))
    dtype = 'float32'
    reg_to_dwi(seg_out, transform_mat_mrtrix, seg_out_coreg, dtype)

    # Generate GM/WM boundary
    gmwm_bound = os.path.join(dest_dwi_dir, 'gmwm-bound-coreg_%s_%s.mif' % (subject, session))
    gen_gmwm_boundary(seg_out_coreg, gmwm_bound)

    # Create streamlines
    track_out = os.path.join(dest_dwi_dir, 'tracks_%s_%s_t1.tck' % (subject, session))
    max_len = 250
    cutoff = 0.06
    nstreamlines = 10000000
    gen_streamlines(seg_out_coreg, gmwm_bound, wm_fod, track_out, max_len, cutoff, nstreamlines)

    # Start work on building the connectome

    # Remove over- and under-fitted white matter tracts
    out_mu = os.path.join(dest_dwi_dir, 'sift-mu_%s_%s.txt' % (subject, session))
    out_coeff = os.path.join(dest_dwi_dir, 'sift-coeffs_%s_%s.txt' % (subject, session))
    out_sift = os.path.join(dest_dwi_dir, 'sift-track_%s_%s.txt' % (subject, session))
    sift_tracts(seg_out_coreg, out_mu, out_coeff, track_out, wm_fod, out_sift)

    # recon-all has been run on all subjects already, using one of the output files (aparc+aseg.mgz) here
    # Converting labels
    fs_dir = os.path.join('/storage/store2/data/ibc/derivatives/%s/%s/anat/%s/mri' % (subject, 'ses-00', subject))
    fs_label = glob.glob('%s/aparc+aseg.mgz' % (fs_dir))[0]
    fs_clut = glob.glob('/usr/local/freesurfer/FreeSurferColorLUT.txt')[0]
    mrtrix_clut = '/storage/store2/work/sshankar/miniconda3/pkgs/mrtrix3-3.0.3-h2bc3f7f_0/share/mrtrix3/labelconvert/fs_default.txt'
    parcels = os.path.join(dest_dwi_dir, '%s_%s_parcels.mif' % (subject, session))
    labelconvert(fs_label, fs_clut, mrtrix_clut, parcels)

    # Coregister the parcellation
    coreg_parcel = os.path.join(dest_dwi_dir, 'coreg_%s_%s_parcels.mif' % (subject, session))
    dtype = 'uint32'
    reg_to_dwi(parcels, transform_mat_mrtrix, coreg_parcel, dtype)

    # Creating the connectome
    coreg_parcel_csv = os.path.join(dest_dwi_dir, 'coreg_%s_%s_parcels.csv' % (subject, session))
    inverse_coreg_parcel = os.path.join(dest_dwi_dir, 'inv-coreg_%s_%s_parcels.csv' % (subject, session))
    gen_connectome(out_sift, track_out, coreg_parcel, coreg_parcel_csv, inverse_coreg_parcel)

    # Plot the connectome
    out_img = os.path.join(dest_dwi_dir, '%s_%s_connectome.png' % (subject, session))
    format = 'png'
    plot_connectome(coreg_parcel_csv, out_img, format)

for subject_session in subjects_sessions:
    run_dmri_preproc(subject_session)

# Parallel(n_jobs=1)(
#     delayed(run_dmri_preproc)(subject_session)
#     for subject_session in subjects_sessions)
