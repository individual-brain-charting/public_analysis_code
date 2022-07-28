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
subjects = ['sub-%02d' % i for i in [1, 4, 5, 6, 7, 8, 9, 13, 14]]
# subjects = ['sub-%02d' % i for i in [11, 12, 15]] # 
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
    data = {}
    for modality in ['T1w', 'T2w']:
        if modality == 'T1w':
            image = ref_file
        else:
            image = sorted(glob.glob(os.path.join(
                work_dir, subject, 'ses-*', 'anat', '*-highres_T2w.nii')))[-1]

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

        if 1:
            # output names
            # the .gii files will be put in the same directory as the input
            left_tex = os.path.join(write_dir, basename + '_individual_lh.gii')
            right_tex = os.path.join(write_dir, basename + '_individual_rh.gii')

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
                write_dir, basename + '_fsaverage7_lh.gii')
            right_smooth_tex = os.path.join(
                write_dir, basename + '_fsaverage7_rh.gii')

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

            # resample to fsaverage5
            left_smooth_tex = os.path.join(
                write_dir, basename + '_fsaverage5_lh.gii')
            right_smooth_tex = os.path.join(
                write_dir, basename + '_fsaverage5_rh.gii')

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
            
            data[modality] = {
                'lh': nib.load(left_smooth_tex).darrays[0].data,
                'rh': nib.load(right_smooth_tex).darrays[0].data
                }
        else:
            from surfer import project_volume_data
            data[modality] = {}
            for hemi in ['lh', 'rh']:
                data_ = project_volume_data(
                    image_, hemi, regheader, projarg=[0, 1., .1],
                    smooth_fwhm=0)
                data[modality][hemi] = data_

    # reset subject_dir to set fsaverage
    os.environ['SUBJECTS_DIR'] = os.path.join(
        work_dir, subject, 'ses-00', 'anat')
    for hemi in ['lh', 'rh']:
        ratio = data['T1w'][hemi] / data['T2w'][hemi]
        from nibabel.gifti import write, GiftiImage, GiftiDataArray as gda
        file_ratio = os.path.join(write_dir, 't1_t2_ratio_%s.gii' % hemi)
        write(GiftiImage(darrays=[gda(data=ratio.astype('float32'))]),
              file_ratio)


Parallel(n_jobs=1)(
    delayed(project_volume)(data_dir, subject)
    for subject in subjects)


###########################################################################
from nilearn.plotting import plot_surf, view_surf, show
from nilearn import datasets
import os
import nibabel as nib
import glob

fsaverage = datasets.fetch_surf_fsaverage('fsaverage5')

dir_ = '/neurospin/ibc/derivatives/sub-*/ses-*/anat/analysis/'
if 0:
    wc = os.path.join(dir_, 't1_t2_ratio_lh.gii')
    textures = glob.glob(wc)
    for texture in textures:
        tex = nib.load(texture).darrays[0].data
        view_surf(
            fsaverage['infl_left'], surf_map=tex, bg_map=None, vmin=1, vmax=2
        ).open_in_browser()

    wc = os.path.join(dir_, 't1_t2_ratio_rh.gii')
    textures = glob.glob(wc)
    for texture in textures:
        tex = nib.load(texture).darrays[0].data
        view_surf(
            fsaverage['infl_right'], surf_map=tex, bg_map=None, vmin=1, vmax=2
        ).open_in_browser()

if 1:
    wc = os.path.join(dir_, 't1_t2_ratio_lh.gii')
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



