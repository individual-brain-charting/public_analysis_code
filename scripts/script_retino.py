"""
Script used to perform retinotopic analysis based on the IBC low-level visual
protocol.

Author: Bertrand Thirion, 2015--2017
"""
import os
import glob
from os.path import join as pjoin
import numpy as np
from nilearn.image import mean_img, math_img
from ibc_public.utils_retino import phase_maps
import nibabel as nib
from nilearn.plotting import plot_stat_map, plot_surf_stat_map
import matplotlib.pyplot as plt
from nilearn.input_data import NiftiMasker
from nilearn.image import resample_img
from nibabel import load, save
from nibabel.gifti import GiftiDataArray, GiftiImage
from ibc_public.utils_data import DERIVATIVES
from nilearn.glm import fdr_threshold

data_dir = DERIVATIVES
do_surface = True

# find the subjects / sessions with the retinotopy protocol
cos_imgs = glob.glob(pjoin(data_dir, 'sub-*', 'ses-*', 'res_task*',
                           'z_score_maps', 'cos.nii.gz'))

subjects_sessions = []
for cos_img in cos_imgs:
    subject, session = cos_img.split('/')[-5:-3]
    subjects_sessions.append((subject + '_' +session))
subjects_sessions = np.unique(subjects_sessions)

# list all the acquisitions to take into account
acqs = ['res_stats_%s' % acq for acq in [
    'WedgeAnti_pa', 'WedgeAnti_ap', 'WedgeClock_ap', 'WedgeClock_pa',
    'ExpRing_pa', 'ContRing_ap']]
mesh = 'fsaverage7'
if do_surface:
    acqs = ['res_task-{}_space-{}_dir-{}'.format(acq[:-3], mesh, acq[-2:]) for acq in [
        'WedgeAnti_pa', 'WedgeAnti_ap', 'WedgeClock_ap', 'WedgeClock_pa',
        'ExpRing_pa', 'ContRing_ap']]

THRESHOLD = 4.
alpha = .05

#######################################################################################
# stuff for volume data

ref_affine = np.array([[-1.5, 0., 0., 78.],
                       [0. , 1.5, 0., -112.],
                       [0. , 0., 1.5, -70.],
                       [0. , 0., 0., 1.]])
ref_shape = (105, 127, 105)


#######################################################################################
# stuff for surface plotting

def make_paths(subject):
    surf_dir = os.path.join(
        DERIVATIVES, subject, 'ses-00', 'anat', 'fsaverage', 'surf')
    lh_pial = os.path.join(surf_dir, 'lh.pial')
    lh_white = os.path.join(surf_dir, 'lh.white')
    rh_pial = os.path.join(surf_dir, 'rh.pial')
    rh_white = os.path.join(surf_dir, 'rh.white')
    sulc_left =  os.path.join(surf_dir, 'lh.sulc')
    sulc_right =  os.path.join(surf_dir, 'rh.sulc')
    lh_inflated = os.path.join(surf_dir, 'lh.inflated')
    rh_inflated = os.path.join(surf_dir, 'rh.inflated')
    return(lh_pial, lh_white, rh_pial, rh_white, sulc_left, sulc_right,
           lh_inflated, rh_inflated)
    

def get_retino_coefs(work_dir, mesh, hemi):
    """Reads files containing retinotopic coefficients"""
    #
    cos_wedge_clock = np.mean([np.ravel([
        darrays.data for darrays in load(z_map).darrays]) for z_map in (
            pjoin(work_dir, 'res_task-WedgeClock_space-{}_dir-pa'.format(mesh),
                  'z_score_maps', 'cos_%s.gii' % hemi),
            pjoin(work_dir, 'res_task-WedgeClock_space-{}_dir-ap'.format(mesh),
                  'z_score_maps', 'cos_%s.gii' % hemi))], 0)
    sin_wedge_clock = np.mean([np.ravel([
        darrays.data for darrays in load(z_map).darrays]) for z_map in (
            pjoin(work_dir, 'res_task-WedgeClock_space-{}_dir-pa'.format(mesh),
                  'z_score_maps', 'sin_%s.gii' % hemi),
            pjoin(work_dir, 'res_task-WedgeClock_space-{}_dir-ap'.format(mesh),
                  'z_score_maps', 'sin_%s.gii' % hemi))], 0)
    cos_wedge_anti = np.mean([np.ravel([
        darrays.data for darrays in load(z_map).darrays]) for z_map in (
            pjoin(work_dir, 'res_task-WedgeAnti_space-{}_dir-pa'.format(mesh),
                  'z_score_maps', 'cos_%s.gii' % hemi),
            pjoin(work_dir, 'res_task-WedgeAnti_space-{}_dir-ap'.format(mesh),
                  'z_score_maps', 'cos_%s.gii' % hemi))], 0)
    sin_wedge_anti = np.mean([np.ravel([
        darrays.data for darrays in load(z_map).darrays]) for z_map in (
            pjoin(work_dir, 'res_task-WedgeAnti_space-{}_dir-pa'.format(mesh),
                  'z_score_maps', 'sin_%s.gii' % hemi),
            pjoin(work_dir, 'res_task-WedgeAnti_space-{}_dir-ap'.format(mesh),
                  'z_score_maps', 'sin_%s.gii' % hemi))], 0)
    retino_imgs = {
        'cos_wedge_pos': cos_wedge_anti,
        'sin_wedge_pos': sin_wedge_anti,
        'sin_wedge_neg': sin_wedge_clock,
        'cos_wedge_neg': cos_wedge_clock,
        'cos_ring_pos': pjoin(
            work_dir, 'res_task-ExpRing_space-{}_dir-pa'.format(mesh), 'z_score_maps',
            'cos_%s.gii' % hemi),
        'sin_ring_pos': pjoin(
            work_dir, 'res_task-ExpRing_space-{}_dir-pa'.format(mesh), 'z_score_maps',
            'sin_%s.gii' % hemi),
        'sin_ring_neg': pjoin(
            work_dir, 'res_task-ContRing_space-{}_dir-ap'.format(mesh), 'z_score_maps',
            'sin_%s.gii' % hemi),
        'cos_ring_neg': pjoin(
            work_dir, 'res_task-ContRing_space-{}_dir-ap'.format(mesh), 'z_score_maps',
            'cos_%s.gii' % hemi)
    }
    retino_coefs = {}
    for key in retino_imgs.keys():
        if isinstance(retino_imgs[key], np.ndarray):
            retino_coefs[key] = retino_imgs[key]
        else:
            retino_coefs[key] = np.ravel([
                darrays.data for darrays in load(retino_imgs[key]).darrays])
    return retino_coefs


def faces_2_connectivity(faces, identity=True, normalize=True):
    from scipy.sparse import coo_matrix, dia_matrix
    n_features = len(np.unique(faces))
    edges = np.vstack((faces.T[:2].T, faces.T[1:].T, faces.T[0:3:2].T))
    weight = np.ones(edges.shape[0])
    connectivity = coo_matrix(
        (weight, (edges.T[0], edges.T[1])),
        (n_features, n_features))  # .tocsr()
    # Making it symmetrical
    connectivity = (connectivity + connectivity.T) / 2
    if identity:
        connectivity = connectivity +  coo_matrix(
                (np.ones(n_features), (np.arange(n_features), np.arange(n_features))),
                (n_features, n_features))  
    if normalize:
        connectivity.data /= np.array(connectivity.sum(1)[connectivity.nonzero()[0].T]).T[0]
    return connectivity

# parameters to smooth the stat map
from nilearn.surface import load_surf_mesh
from nilearn.datasets import fetch_surf_fsaverage
fsaverage = fetch_surf_fsaverage('fsaverage7')
mesh_ = fsaverage['pial_right']
_, faces = load_surf_mesh(mesh_)
smoothing_kernel_rh = faces_2_connectivity(faces)
mesh_ = fsaverage['pial_left']
_, faces = load_surf_mesh(mesh_)
smoothing_kernel_lh = faces_2_connectivity(faces)
n_smoothing = 0

for subject_session in subjects_sessions:
    subject, session = subject_session.split('_')
    # subject, session = subject_session
    work_dir = pjoin(data_dir, subject, session)
    # result directory
    ses_dir = pjoin(DERIVATIVES, subject, session)
    if not os.path.exists(ses_dir):
        os.mkdir(ses_dir)
    task_dir = pjoin(ses_dir, 'res_task-Retinotopy_space-MNI152_dir-ffx')
    if do_surface:
        task_dir = pjoin(ses_dir, 'res_task-Retinotopy_space-{}_dir-ffx'.format(mesh))
    if not os.path.exists(task_dir):
        os.mkdir(task_dir)
    write_dir = pjoin(task_dir, 'stat_maps')
    if not os.path.exists(write_dir):
        os.mkdir(write_dir)

    # Compute the fixed effects across sessions
    # quick and dirty approach: sum z maps
    if do_surface:
        for hemi in ['lh', 'rh']:
            z_maps = [
                pjoin(work_dir, acq, 'z_score_maps',
                      'effects_interest_%s.gii' % hemi)
                for acq in acqs]

            mean_z = np.mean([np.ravel([
                darrays.data for darrays in load(z_map).darrays])
                              for z_map in z_maps], 0)
            n_maps = len(z_maps)
            fixed_effects = mean_z * np.sqrt(n_maps)
            bh = fdr_threshold(fixed_effects, alpha)
            mask = fixed_effects > bh
            print(subject, bh, np.sum(mask))
            smoothing_kernel = smoothing_kernel_lh
            if hemi == 'rh':
                 smoothing_kernel = smoothing_kernel_rh
            print(fixed_effects.max())
            for  _ in range(n_smoothing):
                fixed_effects = smoothing_kernel.dot(fixed_effects)
            fixed_effects[np.isnan(fixed_effects)] = 0
            mask = fixed_effects > bh
            print(fixed_effects.max(),np.sum(mask))
            gii = GiftiImage(
                darrays=[GiftiDataArray(fixed_effects.astype('float32'), intent='t test')])
                       # GiftiDataArray().from_array(fixed_effects, 't test')])
            gii.to_filename(pjoin(write_dir, 'retinotopicity_%s.gii' % hemi))
            
            
            # todo: plot on a surface
            """
            lh_pial, lh_white, rh_pial, rh_white, sulc_left, sulc_right,
           lh_inflated, rh_inflated = make_paths(subject)
            output_file = pjoin(write_dir, 'retinotopicity_%s.png' % hemi)
            if hemi == 'lh':
                plot_surf_stat_map(
                    lh_inflated, fixed_effects, bg_map=sulc_left,
                    output_file=output_file,
                    hemi='left', view='medial', bg_on_data=True, darkness=1,
                    alpha=1,
                    threshold=THRESHOLD)
            else:
                plot_surf_stat_map(
                    rh_inflated, fixed_effects, bg_map=sulc_right,
                    output_file=output_file,
                    hemi='right', view='medial', bg_on_data=True, darkness=1,
                    alpha=1,
                    threshold=THRESHOLD)
            """
            retino_coefs = get_retino_coefs(work_dir, mesh, hemi)
            
            phase_wedge, phase_ring, phase_hemo = phase_maps(
                retino_coefs,
                offset_ring=np.pi,
                offset_wedge=0,
                do_wedge=True,
                do_ring=True,
            )
            phase_wedge[mask == 0] = 0
            phase_ring[mask == 0] = 0
            phase_hemo[mask == 0] = 0
            GiftiImage(
                darrays=[
                    GiftiDataArray(phase_wedge.astype('float32'), intent='NIFTI_INTENT_ESTIMATE')]).\
                    to_filename(pjoin(write_dir, 'phase_wedge_%s.gii' % hemi))
            GiftiImage(
                darrays=[
                    GiftiDataArray(phase_ring.astype('float32'), intent='NIFTI_INTENT_ESTIMATE')]).\
                    to_filename(pjoin(write_dir, 'phase_ring_%s.gii' % hemi))
            GiftiImage(
                darrays=[
                    GiftiDataArray(phase_hemo.astype('float32'), intent='NIFTI_INTENT_ESTIMATE')]).\
                    to_filename(pjoin(write_dir, 'phase_hemo_%s.gii' % hemi))
    else:
        z_maps = [pjoin(work_dir, acq, 'z_score_maps', 'effects_interest.nii.gz')
                  for acq in acqs]
        mean_z = resample_img(
            mean_img(z_maps), target_affine=ref_affine, target_shape=ref_shape)
        n_maps = len(z_maps)
        fixed_effects = math_img('im * %d' % np.sqrt(n_maps), im=mean_z)
        fixed_effects.to_filename(pjoin(write_dir, 'retinotopicity.nii.gz'))

        anat = pjoin(
            data_dir, subject, 'ses-00', 'anat', 'w%s_ses-00_T1w_nonan.nii.gz' % subject)
        plot_stat_map(fixed_effects, threshold=THRESHOLD, bg_img=anat, dim=0,
                      output_file=pjoin(write_dir, 'retinotopicity.png'))
        mask = fixed_effects.get_data() > THRESHOLD
        mask_img = nib.Nifti1Image(mask.astype('uint8'), fixed_effects.affine)
        # to be completed with region size thresholding
        masker = NiftiMasker(mask_img=mask_img).fit()

        # take all images corresonding to sine regressors
        cos_wedge_clock = mean_img((
            pjoin(work_dir, 'res_stats_WedgeClock_pa', 'z_score_maps', 'cos.nii.gz'),
            pjoin(work_dir, 'res_stats_WedgeClock_ap', 'z_score_maps', 'cos.nii.gz')))

        sin_wedge_clock = mean_img((
            pjoin(work_dir, 'res_stats_WedgeClock_pa', 'z_score_maps', 'sin.nii.gz'),
            pjoin(work_dir, 'res_stats_WedgeClock_ap', 'z_score_maps', 'sin.nii.gz')))

        cos_wedge_anti = mean_img((
            pjoin(work_dir, 'res_stats_WedgeAnti_pa', 'z_score_maps', 'cos.nii.gz'),
            pjoin(work_dir, 'res_stats_WedgeAnti_ap', 'z_score_maps', 'cos.nii.gz')))

        sin_wedge_anti = mean_img((
            pjoin(work_dir, 'res_stats_WedgeAnti_pa', 'z_score_maps', 'sin.nii.gz'),
            pjoin(work_dir, 'res_stats_WedgeAnti_ap', 'z_score_maps', 'sin.nii.gz')))

        retino_imgs = {
            'cos_wedge_pos': cos_wedge_anti,
            'sin_wedge_pos': sin_wedge_anti,
            'sin_wedge_neg': sin_wedge_clock,
            'cos_wedge_neg': cos_wedge_clock,
            'cos_ring_pos': pjoin(
                work_dir, 'res_stats_ExpRing_pa', 'z_score_maps', 'cos.nii.gz'),
            'sin_ring_pos': pjoin(
                work_dir, 'res_stats_ExpRing_pa', 'z_score_maps', 'sin.nii.gz'),
            'sin_ring_neg': pjoin(
                work_dir, 'res_stats_ContRing_ap', 'z_score_maps', 'sin.nii.gz'),
            'cos_ring_neg': pjoin(
                work_dir, 'res_stats_ContRing_ap', 'z_score_maps', 'cos.nii.gz')
        }
 
        retino_coefs = {}
        for key in retino_imgs.keys():
            retino_coefs[key] = masker.transform(retino_imgs[key])

        phase_wedge, phase_ring, phase_hemo = phase_maps(
            retino_coefs, offset_ring=np.pi, offset_wedge=0.,
            do_wedge=True, do_ring=True,
        )

        phase_wedge_img = masker.inverse_transform(phase_wedge)
        phase_ring_img = masker.inverse_transform(phase_ring)
        phase_hemo_img = masker.inverse_transform(phase_hemo)

        phase_wedge_img.to_filename(pjoin(write_dir, 'phase_wedge.nii.gz'))
        phase_ring_img.to_filename(pjoin(write_dir, 'phase_ring.nii.gz'))
        phase_hemo_img.to_filename(pjoin(write_dir, 'phase_hemo.nii.gz'))

        plot_stat_map(phase_wedge_img, title='polar angle', cmap='hsv',
                      bg_img=anat, dim=1,
                      output_file=pjoin(write_dir, 'phase_wedge.png'))
        plot_stat_map(phase_ring_img, title='eccentricity', cmap='hsv',
                      bg_img=anat, dim=1,
                      output_file=pjoin(write_dir, 'phase_ring.png'))
        plot_stat_map(phase_hemo_img, title='hemodynamics', cmap='hsv',
                      bg_img=anat, dim=1,
                      output_file=pjoin(write_dir, 'phase_hemo.png'))




"""
for i, subject_session in enumerate(subjects_sessions[2:3]):
    subject, session = subject_session.split('_')
    write_dir = pjoin(
        DERIVATIVES, subject, session,
        'res_task-Retinotopy_space-{}_dir-ffx'.format(mesh), 'stat_maps')
    rh_mesh = fsaverage.infl_right
    lh_mesh = fsaverage.infl_left
    rh_sulc = fsaverage.sulc_right
    lh_sulc = fsaverage.sulc_left
    # lh_mesh = pjoin(DERIVATIVES, subject, 'ses-00', 'anat', subject, 'surf', 'lh.inflated')
    # lh_sulc = pjoin(DERIVATIVES, subject, 'ses-00', 'anat', subject, 'surf', 'lh.sulc')
    # rh_mesh = pjoin(DERIVATIVES, subject, 'ses-00', 'anat', subject, 'surf', 'rh.inflated')
    # rh_sulc = pjoin(DERIVATIVES, subject, 'ses-00', 'anat', subject, 'surf', 'rh.sulc')
    for j, stat in enumerate(['phase_wedge', 'phase_ring']):
        lh = os.path.join(write_dir, '%s_lh.gii' % stat)
        rh = os.path.join(write_dir, '%s_rh.gii' % stat)
        output_file = os.path.join(write_dir, '%s.png' % stat)

        x1 = np.ravel([darrays.data for darrays in load(lh).darrays])
        x2 = np.ravel([darrays.data for darrays in load(rh).darrays])
        #x1[x1 == 0] = np.nan
        #x2[x2 == 0] = np.nan
        fig = plot_surf_stat_map(
            rh_mesh, x2, hemi='right',
            title='{}, right hemisphere'.format(stat), colorbar=True,
            threshold=1.e-6, bg_map=rh_sulc, cmap='coolwarm',
            engine='plotly'  # Specify the plotting engine here
        )
        fig.show()
        fig = plot_surf_stat_map(
            lh_mesh, x1, hemi='left',
            title='{}, left hemisphere'.format(stat), colorbar=True,
            threshold=1.e-6, bg_map=lh_sulc, cmap='coolwarm',
            engine='plotly'  # Specify the plotting engine here
        )
        fig.show()
"""
        
plt.show(block=False)

"""
from nilearn import surface, plotting
fig = plt.figure(figsize=(6, 20))
fig, axes = plt.subplots(nrows=12, ncols=2, subplot_kw={'projection': '3d'})

for i, subject_session in enumerate(subjects_sessions):
    subject, session = subject_session.split('_')
    write_dir = pjoin(
        DERIVATIVES, subject, session,
        'res_task-Retinotopy_space-fsaverage7_dir-ffx', 'stat_maps')

    for j, stat in enumerate(['phase_wedge', 'phase_ring']):
        lh = os.path.join(write_dir, '%s_lh.gii' % stat)
        rh = os.path.join(write_dir, '%s_rh.gii' % stat)
        output_file = os.path.join(write_dir, '%s.png' % stat)

        x1 = np.ravel([darrays.data for darrays in load(lh).darrays])
        x2 = np.ravel([darrays.data for darrays in load(rh).darrays])
        x = np.hstack((x1, x2))
        x[x == 0] = np.nan
        
        # ax = plt.subplot(12, 2, i * 2 + j + 1, subplot_kw={'projection': '3d'})
        ax = axes[i, j]
        bg_map = np.sign(surface.load_surf_data(fsaverage['curv_left']))
        bg_map_rescaled = (bg_map + 1) / 4 + 0.25

        plotting.plot_surf_stat_map(
            fsaverage['flat_left'],
            stat_map=x1,
            hemi='left',
            view='dorsal',
            bg_map=bg_map_rescaled,
            bg_on_data=True,
            axes=ax,
            colorbar=False,
            cmap='coolwarm',
        )
    #ax = plt.axes([.46, .96 - i * 1. * .0815, .08, .04])
    #ax.text(.01, .2, subject)
    #ax.axis('off')

plt.subplots_adjust(left=.01, right=.99, top=.99, bottom=.01, hspace=.01, wspace=.01)

output_file = pjoin(DERIVATIVES, 'group', 'retino', 'retino.svg')

fig.savefig(output_file)


"""


import cortex
mesh = 'fsaverage7'
plt.figure(figsize=(6, 20))
for i, subject_session in enumerate(subjects_sessions):
    subject, session = subject_session.split('_')
    write_dir = pjoin(
        DERIVATIVES, subject, session,
        'res_task-Retinotopy_space-{}_dir-ffx'.format(mesh), 'stat_maps')

    for j, stat in enumerate(['phase_wedge', 'phase_ring']):
        lh = os.path.join(write_dir, '%s_lh.gii' % stat)
        rh = os.path.join(write_dir, '%s_rh.gii' % stat)
        output_file = os.path.join(write_dir, '%s.png' % stat)

        x1 = np.ravel([darrays.data for darrays in load(lh).darrays])
        x2 = np.ravel([darrays.data for darrays in load(rh).darrays])
        x = np.hstack((x1, x2))
        x[x == 0] = np.nan
        
        ax = plt.subplot(12, 2, i * 2 + j + 1)

        vertex_data = cortex.Vertex(x, 'fsaverage')
        fig = cortex.quickshow(vertex_data,
                               with_colorbar=False,
                               with_rois=False,
                               with_labels=False,
                               with_curvature=True,
                               curvature_contrast=0.5,
                               curvature_brightness=0.5,
                               curvature_threshold=True,
                               fig = ax
        )

        #fig.set_size_inches((8, 4.5))
        #fig.savefig(output_file)
    #
    ax = plt.axes([.46, .96 - i * 1. * .0815, .08, .04])
    ax.text(.01, .2, subject)
    ax.axis('off')

plt.subplots_adjust(left=.01, right=.99, top=.99, bottom=.01, hspace=.01, wspace=.01)

output_file = pjoin(DERIVATIVES, 'group', 'retino', 'retino.svg')

fig.savefig(output_file)
