"""
This script is meant to perform anatomical mapping on the cortical surface
using tools from CARET and Freesurfer
# Second thought: Use only Freesurfer
a. Freesurfer segmentation on T1
b. bbreg and registration of T2
c. projection of T1 and T2, computation of the ratio.

Author: Bertrand Thirion
"""
import os
from nipype.interfaces.freesurfer import BBRegister
from joblib import Memory, Parallel, delayed
import glob
import numpy as np
import nibabel as nib

data_dir = '/neurospin/ibc/derivatives'
subjects = ['sub-%02d' % i for i in [1, 2, 4, 5, 6, 7, 8, 9, 11, 13, 14, 12, 15]]
subjects = ['sub-%02d' % i for i in [12]]
os.environ['SUBJECTS_DIR'] = ''


def smooth_data_as_texture(data, subject, hemi):
    """To smooth the data, save them as texture,
        surfs2surf and extract the data """
    from nibabel.gifti import read, write, GiftiImage, GiftiDataArray as gda
    file_raw = '/tmp/data.gii'
    file_smooth = '/tmp/smooth_data.gii'
    write(GiftiImage(darrays=[gda(data=data.astype('float32'))]), file_raw)
    os.system(
        '$FREESURFER_HOME/bin/mri_surf2surf' +
        ' --srcsubject %s' % subject +
        ' --srcsurfval %s' % file_raw +
        ' --trgsurfval %s' % file_smooth +
        ' --trgsubject %s' % subject +
        ' --hemi %s' % hemi + ' --nsmooth-out 2')
    return read(file_smooth).darrays[0].data


def read_data(tex):
    from nibabel.gifti import read
    return read(tex).darrays[0].data


def closing(image):
    """Numerical closing of the image

    Parameters
    ----------
    image: string,
           input image

    returns
    -------
    filename: string,
              path of closed image
    """
    from scipy.ndimage.morphology import grey_closing
    # from nilearn.plotting import plot_anat
    import nibabel as nib
    data = nib.load(image).get_fdata()
    data_ = grey_closing(data, size=(3, 3, 3))
    img = nib.Nifti1Image(data_, nib.load(image).affine)
    print(np.sum((data - data_) ** 2))
    filename = os.path.join(
        os.path.dirname(image), 'analysis', os.path.basename(image)[:-4] +
        '_closed.nii.gz')
    img.to_filename(filename)
    return filename


def median_filter(image):
    """Numerical filter of the image

    Parameters
    ----------
    image: string,
           input image

    returns
    -------
    filename: string,
              path of closed image
    """
    from scipy.ndimage import median_filter
    # from nilearn.plotting import plot_anat
    import nibabel as nib
    data = nib.load(image).get_fdata()
    data_ = median_filter(data, size=(3, 3, 3))
    img = nib.Nifti1Image(data_, nib.load(image).affine)
    print(np.sum((data - data_) ** 2) / np.sum(data ** 2))
    filename = os.path.join(
        os.path.dirname(image), 'analysis', os.path.basename(image)[:-4] +
        '_median.nii.gz')
    if not os.path.exists(os.path.dirname(filename)):
        os.mkdir(os.path.dirname(filename))
    img.to_filename(filename)
    return filename


def project_volume(work_dir, subject, do_bbr=True):
    # first find the session where T1w and T2w files could be
    ref_file = sorted(glob.glob(os.path.join(
        work_dir, subject, 'ses-*', 'anat', 'sub*-highres_T1w.nii')))[-1]
    # session = ref_file.split('/')[-3]
    anat_dir = os.path.dirname(ref_file)

    write_dir = os.path.join(anat_dir, 'analysis')
    if not os.path.exists(write_dir):
        os.mkdir(write_dir)
    os.environ['SUBJECTS_DIR'] = anat_dir
    data = {'fsaverage5': {},
            'fsaverage7': {}
    }
    for modality in ['T1w', 'T2w']:
        if modality == 'T1w':
            image = ref_file
        else:
            image = sorted(glob.glob(os.path.join(
                work_dir, subject, 'ses-*', 'anat', '*-highres_T2w.nii')))[0]

        image_ = median_filter(image)
        
        # --------------------------------------------------------------------
        # run the projection using freesurfer
        print("image", image)
        basename = os.path.basename(image).split('.')[0]

        if modality == 'T1w':
            bbreg = BBRegister(subject_id=subject, source_file=image,
                               init='header', contrast_type='t1')
        else:
            # use BBR registration to finesse the coregistration
            bbreg = BBRegister(subject_id=subject, source_file=image,
                               init='header', contrast_type='t2')

        regheader = os.path.join(os.path.dirname(image), basename + '_bbreg_%s.dat'
                                 % subject)
        bbreg.run()

        # output names
        # the .gii files will be put in the same directory as the input
        left_tex = os.path.join(write_dir, basename + '_space-individual_lh.gii')
        right_tex = os.path.join(write_dir, basename + '_space-individual_rh.gii')

        # run freesrufer command for projection
        os.system(
            '$FREESURFER_HOME/bin/mri_vol2surf --src %s --o %s '
            '--out_type gii --srcreg %s --hemi lh --projfrac-avg 0 1 0.1'
            % (image_, left_tex, regheader))
        
        os.system(
            '$FREESURFER_HOME/bin/mri_vol2surf --src %s --o %s '
            '--out_type gii --srcreg %s --hemi rh --projfrac-avg 0 1 0.1'
            % (image_, right_tex, regheader))

        # resample to fsaverage7
        left_smooth_tex = os.path.join(
            write_dir, basename + '_space-fsaverage7_lh.gii')
        right_smooth_tex = os.path.join(
            write_dir, basename + '_space-fsaverage7_rh.gii')

        os.system(
            '$FREESURFER_HOME/bin/mri_surf2surf --srcsubject %s '
            '--srcsurfval %s --trgsurfval %s --trgsubject ico '
            '--trgicoorder 7 --hemi lh' %
            (subject, left_tex, left_smooth_tex))
        os.system(
            '$FREESURFER_HOME/bin/mri_surf2surf --srcsubject %s '
            '--srcsurfval %s --trgsubject ico --trgicoorder 7 '
            '--trgsurfval %s --hemi rh' %
            (subject, right_tex, right_smooth_tex))

        data['fsaverage7'][modality] = {
            'lh': nib.load(left_smooth_tex).darrays[0].data,
            'rh': nib.load(right_smooth_tex).darrays[0].data
        }

        # resample to fsaverage5
        left_smooth_tex = os.path.join(
            write_dir, basename + '_space-fsaverage5_lh.gii')
        right_smooth_tex = os.path.join(
            write_dir, basename + '_space-fsaverage5_rh.gii')
        
        os.system(
            '$FREESURFER_HOME/bin/mri_surf2surf --srcsubject %s '
            '--srcsurfval %s --trgsurfval %s --trgsubject ico '
            '--trgicoorder 5 --hemi lh' %
            (subject, left_tex, left_smooth_tex))
        os.system(
            '$FREESURFER_HOME/bin/mri_surf2surf --srcsubject %s '
            '--srcsurfval %s --trgsubject ico --trgicoorder 5 '
            '--trgsurfval %s --hemi rh' %
            (subject, right_tex, right_smooth_tex))
            
        data['fsaverage5'][modality] = {
            'lh': nib.load(left_smooth_tex).darrays[0].data,
            'rh': nib.load(right_smooth_tex).darrays[0].data
        }
        
    from nibabel.gifti import write, GiftiImage, GiftiDataArray as gda
    session = os.path.basename(ref_file).split('_')[1]
    for mesh in ['fsaverage5', 'fsaverage7']:
        for hemi in ['lh', 'rh']:
            ratio = data[mesh]['T1w'][hemi] / data[mesh]['T2w'][hemi]
            file_ratio = os.path.join(
                write_dir,
                '{}_{}_T1T2Ratio_space-{}_{}.gii'.format(
                    subject, session, mesh, hemi))
            write(
                GiftiImage(darrays=[gda(data=ratio.astype('float32'))]),
                file_ratio)

"""
Parallel(n_jobs=1)(
    delayed(project_volume)(data_dir, subject)
    for subject in subjects)
"""

###########################################################################
from nilearn.plotting import plot_surf, view_surf, show
from nilearn import datasets
from ibc_public.utils_data import get_subject_session
from ibc_public.utils_data import DERIVATIVES
import os

fsaverage = datasets.fetch_surf_fsaverage('fsaverage5')
# dir_ = '/neurospin/ibc/derivatives/sub-12/ses-*/anat/analysis/'

subject_sessions = get_subject_session('anat1')
interactive = False

for subject_session in subject_sessions:
    subject, session = subject_session
    dir_ = os.path.join(DERIVATIVES, subject, session, 'anat', 'analysis')
    if interactive:
        wc = os.path.join(dir_, '*_*_T1T2Ratio_space-fsaverage5_lh.gii')
        textures = glob.glob(wc)
        for texture in textures:
            tex = nib.load(texture).darrays[0].data
            view_surf(
                fsaverage['infl_left'], surf_map=tex, bg_map=None, vmin=1, vmax=2
            ).open_in_browser()

        wc = os.path.join(dir_, '*_*_T1T2Ratio_space-fsaverage5_rh.gii')
        textures = glob.glob(wc)
        for texture in textures:
            tex = nib.load(texture).darrays[0].data
            view_surf(
                fsaverage['infl_right'], surf_map=tex, bg_map=None, vmin=1, vmax=2
            ).open_in_browser()
    else:
        wc = os.path.join(dir_, '*_*_T1T2Ratio_space-fsaverage5_lh.gii')
        textures = glob.glob(wc)
        tex = nib.load(textures[0]).darrays[0].data
        plot_surf(
            fsaverage['infl_left'],
            surf_map=tex, bg_map=None, hemi='left', view='lateral',
            vmin=1, vmax=2, engine='matplotlib')
        plot_surf(
            fsaverage['infl_left'],
            surf_map=tex, bg_map=None, hemi='left', view='medial',
            vmin=1, vmax=2, engine='matplotlib')
        show()



