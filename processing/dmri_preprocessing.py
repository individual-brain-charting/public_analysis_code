"""
This script performs the pre-processing of dMRI data:
1. topup-based distortion correction
2. Motion correction
3. Eddy current correction

Currently draft version with hard coded paths etc.
Author: Bertrand Thirion, 2015
"""
import os
import glob
from joblib import Memory, Parallel, delayed
import numpy as np
import nibabel as nib
from nilearn.masking import compute_epi_mask
from ibc_public.utils_pipeline import fsl_topup
from dipy.reconst.dti import TensorModel, fractional_anisotropy
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
from dipy.direction import peaks_from_model
from dipy.tracking.eudx import EuDX
from dipy.data import get_sphere
from dipy.viz.colormap import line_colors
from dipy.core.gradients import gradient_table
from dipy.segment.quickbundles import QuickBundles
from mayavi import mlab
from ibc_public.utils_data import get_subject_session


source_dir = '/neurospin/ibc/sourcedata'
derivatives_dir = '/neurospin/ibc/derivatives'
do_topup = False
do_edc = 0
subjects_sessions = [('sub-13', 'ses-09')]  # get_subject_session('anat1')


def concat_images(imgs, out):
    """ nib.concat_images does not work, dunno why"""
    data = []
    for img in imgs:
        img_ = nib.load(img)
        data.append(img_.get_data())

    data = np.concatenate(data, 3)
    nib.save(nib.Nifti1Image(data, img_.affine), out)


def eddy_current_correction(img, file_bvals, file_bvecs, write_dir, mem,
                            acqp='b0_acquisition_params_AP.txt'):
    """ Perform Eddy current correction on diffusion data
    Todo: do topup + eddy in one single command
    """
    import nibabel as nib
    bvals = np.loadtxt(file_bvals)
    mean_img = get_mean_unweighted_image(nib.load(img), bvals)
    mean_img.to_filename(os.path.join(write_dir, 'mean_unweighted.nii.gz'))
    mask = compute_epi_mask(mean_img)
    mask_file = os.path.join(write_dir, 'mask.nii.gz')
    nib.save(mask, mask_file)
    corrected = os.path.join(os.path.dirname(img),
                             'ed' + os.path.basename(img))
    index = np.ones(len(bvals), np.uint8)
    index_file = os.path.join(write_dir, 'index.txt')
    np.savetxt(index_file, index)
    cmd = 'fsl5.0-eddy_correct --acqp=%s --bvals=%s --bvecs=%s --imain=%s '\
          '--index=%s --mask=%s --out=%s' % (
           acqp, file_bvals, file_bvecs, img, index_file, mask_file, corrected)
    cmd = 'fsl5.0-eddy_correct %s %s %d' % (img, corrected, 0)
    print(cmd)
    os.system(cmd)
    return nib.load(corrected)


def length(streamline):
    """ Compute the length of streamlines"""
    n = streamline.shape[0] // 2
    return np.sqrt((
        (streamline[0] - streamline[n]) ** 2 +
        (streamline[-1] - streamline[n]) ** 2).sum())


def filter_according_to_length(streamlines, threshold=30):
    """Remove streamlines shorter than the predefined threshold """
    print(len(streamlines))
    for i in range(len(streamlines) - 1, 0, -1):
        if length(streamlines[i]) < threshold:
            streamlines.pop(i)

    print(len(streamlines))
    return streamlines


def adapt_ini_file(template, subject, session):
    """ Adapt an ini file by changing the subject and session"""
    output_name = os.path.join(
        '/tmp', os.path.basename(template)[:- 4] + '_' + subject + '_'
        + session + '.ini')
    f1 = open(template, 'r')
    f2 = open(output_name, 'w')
    for line in f1.readlines():
        f2.write(line.replace('sub-01', subject).replace('ses-01', session))

    f1.close()
    f2.close()
    return output_name


def get_mean_unweighted_image(img, bvals):
    """ Create an average diffusion image from the most weakly weighted images
    for registration"""
    X = img.get_data().T[bvals < 50].T
    return nib.Nifti1Image(X.mean(-1), img.affine)


def visualization(streamlines_file):
    # clustering of fibers into bundles and visualization thereof
    streamlines = np.load(streamlines_file)['arr_0']
    qb = QuickBundles(streamlines, dist_thr=10., pts=18)
    centroids = qb.centroids
    colors = line_colors(centroids).astype(np.float)
    mlab.figure(bgcolor=(0., 0., 0.))
    for streamline, color in zip(centroids, colors):
        mlab.plot3d(streamline.T[0], streamline.T[1], streamline.T[2],
                    line_width=1., tube_radius=.5, color=tuple(color))

    figname = streamlines_file[:-3] + 'png'
    mlab.savefig(figname)
    print(figname)
    mlab.close()


def tractography(img, gtab, mask, dwi_dir, do_viz=True):
    data = img.get_data()
    # dirty imputation
    data[np.isnan(data)] = 0
    # Diffusion model
    csd_model = ConstrainedSphericalDeconvModel(gtab, response=None)

    sphere = get_sphere('symmetric724')
    csd_peaks = peaks_from_model(
        model=csd_model, data=data, sphere=sphere, mask=mask,
        relative_peak_threshold=.5, min_separation_angle=25,
        parallel=False)

    # FA values to stop the tractography
    tensor_model = TensorModel(gtab, fit_method='WLS')
    tensor_fit = tensor_model.fit(data, mask)
    fa = fractional_anisotropy(tensor_fit.evals)
    stopping_values = np.zeros(csd_peaks.peak_values.shape)
    stopping_values[:] = fa[..., None]

    # tractography
    streamline_generator = EuDX(stopping_values,
                                csd_peaks.peak_indices,
                                seeds=10**6,
                                odf_vertices=sphere.vertices,
                                a_low=0.1)

    streamlines = [streamline for streamline in streamline_generator]
    streamlines = filter_according_to_length(streamlines)
    np.savez(os.path.join(dwi_dir, 'streamlines.npz'), streamlines)

    #  write the result as images
    hdr = nib.trackvis.empty_header()
    hdr['voxel_size'] = img.header.get_zooms()[:3]
    hdr['voxel_order'] = 'LAS'
    hdr['dim'] = fa.shape[:3]

    csd_streamlines_trk = ((sl, None, None) for sl in streamlines)
    csd_sl_fname = os.path.join(dwi_dir, 'csd_streamline.trk')
    nib.trackvis.write(csd_sl_fname, csd_streamlines_trk, hdr,
                       points_space='voxel')
    fa_image = os.path.join(dwi_dir, 'fa_map.nii.gz')
    nib.save(nib.Nifti1Image(fa, img.affine), fa_image)
    if 1:
        visualization(os.path.join(dwi_dir, 'streamlines.npz'))

    return streamlines


def run_dmri_pipeline(subject_session, do_topup=True, do_edc=True):
    subject, session = subject_session
    data_dir = os.path.join(source_dir,  subject, session, 'dwi')
    write_dir = os.path.join(derivatives_dir, subject, session)
    dwi_dir = os.path.join(write_dir, 'dwi')
    # Apply topup to the images
    input_imgs = sorted(glob.glob('%s/sub*.nii.gz' % data_dir))
    dc_imgs = sorted(glob.glob(os.path.join(dwi_dir, 'dcsub*run*.nii.gz')))
    mem = Memory('/neurospin/tmp/bthirion/cache_dir')
    if len(dc_imgs) < len(input_imgs):
        se_maps = [
            os.path.join(source_dir, subject, session, 'fmap',
                         '%s_%s_dir-ap_epi.nii.gz' % (subject, session)),
            os.path.join(source_dir, subject, session, 'fmap',
                         '%s_%s_dir-pa_epi.nii.gz' % (subject, session))]

        if do_topup:
            fsl_topup(se_maps, input_imgs, mem, write_dir, 'dwi')

    # Then proceeed with Eddy current correction
    # get the images
    dc_imgs = sorted(glob.glob(os.path.join(dwi_dir, 'dc*run*.nii.gz')))
    dc_img = os.path.join(dwi_dir, 'dc%s_%s_dwi.nii.gz' % (subject, session))
    concat_images(dc_imgs, dc_img)

    # get the bvals/bvec
    file_bvals = sorted(glob.glob('%s/sub*.bval' % data_dir))
    bvals = np.concatenate([np.loadtxt(fbval) for fbval in sorted(file_bvals)])
    bvals_file = os.path.join(dwi_dir, 'dc%s_%s_dwi.bval' % (subject, session))
    np.savetxt(bvals_file, bvals)
    file_bvecs = sorted(glob.glob('%s/sub*.bvec' % data_dir))
    bvecs = np.hstack([np.loadtxt(fbvec) for fbvec in sorted(file_bvecs)])
    bvecs_file = os.path.join(dwi_dir, 'dc%s_%s_dwi.bvec' % (subject, session))
    np.savetxt(bvecs_file, bvecs)

    # Get eddy-preprocessed images
    # eddy_img = nib.load(glob.glob(os.path.join(dwi_dir, 'eddc*.nii*'))[-1])

    # Get eddy-preprocessed images
    eddy_img = mem.cache(eddy_current_correction)(
        dc_img, bvals_file, bvecs_file, dwi_dir, mem)

    # load the data
    gtab = gradient_table(bvals, bvecs, b0_threshold=10)
    # Create a brain mask

    from dipy.segment.mask import median_otsu
    b0_mask, mask = median_otsu(eddy_img.get_data(), 2, 1)
    if subject == 'sub-13':
        from nilearn.masking import compute_epi_mask
        from nilearn.image import index_img
        imgs_ = [index_img(eddy_img, i)
                 for i in range(len(bvals)) if bvals[i] < 50]
        mask_img = compute_epi_mask(imgs_, upper_cutoff=.8)
        mask_img.to_filename('/tmp/mask.nii.gz')
        mask = mask_img.get_data()
    # do the tractography
    streamlines = tractography(eddy_img, gtab, mask, dwi_dir)
    return streamlines


Parallel(n_jobs=1)(
    delayed(run_dmri_pipeline)(subject_session, do_topup, do_edc)
    for subject_session in subjects_sessions)

# mlab.show()
