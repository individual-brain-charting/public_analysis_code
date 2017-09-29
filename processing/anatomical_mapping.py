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
import commands
from nipype.interfaces.freesurfer import BBRegister
from joblib import Memory, Parallel, delayed
from surfer import Brain, project_volume_data
import glob
import mayavi.mlab as mlab
import numpy as np

data_dir = '/neurospin/ibc/derivatives'
subjects = ['sub-%02d' % i for i in [5]]  #[1, 2, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14]]
os.environ['SUBJECTS_DIR'] = ''


def smooth_data_as_texture(data, subject, hemi):
    """To smooth the data, save them as texture, surfs2surf and extract the data """
    from nibabel.gifti import read, write, GiftiImage, GiftiDataArray as gda
    file_raw = '/tmp/data.gii'
    file_smooth = '/tmp/smooth_data.gii'
    write(GiftiImage(darrays=[gda(data=data.astype('float32'))]), file_raw)
    print(commands.getoutput(
        '$FREESURFER_HOME/bin/mri_surf2surf' +
        ' --srcsubject %s' % subject +
        ' --srcsurfval %s' % file_raw +
        ' --trgsurfval %s' % file_smooth +
        ' --trgsubject %s' % subject +
        ' --hemi %s' % hemi + ' --nsmooth-out 2'))
    return read(file_smooth).darrays[0].data


def read_data(tex):
    from nibabel.gifti import read
    return read(tex).darrays[0].data

def project_volume(work_dir, subject, do_bbr=True):
    # first find the session where T1w and T2w files could be
    ref_file = glob.glob(os.path.join(work_dir, subject, 'ses-*', 'anat',
                                      '*-highres_T2w.nii*'))[-1]
    session = ref_file.split('/')[-3]
    anat_dir = os.path.dirname(ref_file)
    
    write_dir = os.path.join(anat_dir, 'analysis')
    if not os.path.exists(write_dir):
        os.mkdir(write_dir)
    os.environ['SUBJECTS_DIR'] = anat_dir
    data = {}
    smooth_data = {}
    for modality in ['T1w', 'T2w']:
        image = glob.glob(os.path.join(anat_dir, '%s_%s_acq-highres_%s.nii*' % (
            subject, session, modality)))[-1]

        # --------------------------------------------------------------------
        # run the projection using freesurfer
        print("image", image)
        basename = os.path.basename(image).split('.')[0]
        # output names
        # the .gii files will be put in the same directory as the input fMRI
        left_tex = os.path.join(write_dir, basename + '_lh.gii')
        right_tex = os.path.join(write_dir, basename + '_rh.gii')
        
        if modality == 'T1w':
            bbreg = BBRegister(subject_id=subject, source_file=image,
                               init='header', contrast_type='t1')
        else:
            # use BBR registration to finesse the coregistration
            bbreg = BBRegister(subject_id=subject, source_file=image,
                               init='header', contrast_type='t2')

        regheader = os.path.join(anat_dir, basename + '_bbreg_%s.dat' % subject)
        bbreg.run()
        
        if 0:
            # run freesrufer command for projection
            print(commands.getoutput(
                '$FREESURFER_HOME/bin/mri_vol2surf --src %s --o %s '\
                '--out_type gii --srcreg %s --hemi lh --projfrac-avg 0 1 0.1'
                % (image, left_tex, regheader)))

            print(commands.getoutput(
                '$FREESURFER_HOME/bin/mri_vol2surf --src %s --o %s '\
                '--out_type gii --srcreg %s --hemi rh --projfrac-avg 0 1 0.1'
                % (image, right_tex, regheader)))

            # resample to fsaverage
            left_smooth_tex = os.path.join(write_dir, basename + 'fsaverage_lh.gii')
            right_smooth_tex = os.path.join(write_dir, basename + 'fsaverage_rh.gii')

            print(commands.getoutput(
                '$FREESURFER_HOME/bin/mri_surf2surf --srcsubject %s --srcsurfval '\
                '%s --trgsurfval %s --trgsubject ico --trgicoorder 7 '\
                '--hemi lh' %
                (subject, left_tex, left_smooth_tex)))
            print(commands.getoutput(
                '$FREESURFER_HOME/bin/mri_surf2surf --srcsubject %s --srcsurfval '\
                '%s --trgsubject ico --trgicoorder 7 --trgsurfval %s '\
                '--hemi rh' %
                (subject, right_tex, right_smooth_tex)))

    views = ['lat', 'med']
    funcs = [left_smooth_tex, right_smooth_tex]
    for hemi, func in zip(['lh', 'rh'], funcs):
        data_ = 
        brain = Brain("fsaverage", hemi, "inflated",  title='', views=views)
        brain.add_overlay(func, hemi=hemi, min=1, max=2)
        # img = brain.save_montage('/tmp/trial_%s.png' % hemi, order=views, colorbar=None)
            
        """
        data[modality] = {}
        smooth_data[modality] = {} 
        for hemi in ['lh', 'rh']:
            data_ = project_volume_data(
                image, hemi, regheader, projarg=[0, 1., .1], smooth_fwhm=0)
            smooth_data_ = project_volume_data(
                image, hemi, regheader, projarg=[0, 1., .1], smooth_fwhm=5)
            diff = np.abs(smooth_data_ - data_)
            thr = np.percentile(diff, 90)
            data_[diff > thr] = smooth_data_[diff > thr]
            smooth_data_ = smooth_data_as_texture(smooth_data_, subject, hemi)
            data[modality][hemi] = data_
            smooth_data[modality][hemi] = smooth_data_
        """
    """
    for hemi in ['lh']:
        fig = mlab.figure()
        ratio = data['T1w'][hemi] / data['T2w'][hemi]
        brain = Brain(subject, hemi, "white", title='Ratio', figure=fig)
        brain.add_data(ratio, min=1., max=2., colormap='jet')
        brain.show_view("lateral")

        fig = mlab.figure()
        ratio = smooth_data['T1w'][hemi] / smooth_data['T2w'][hemi]
        brain = Brain(subject, hemi, "white", title='Ratio', figure=fig)
        brain.add_data(ratio, min=1., max=2., colormap='jet')
        brain.show_view("lateral")
    """ 
        
        
Parallel(n_jobs=1)(
    delayed(project_volume)(data_dir, subject)
    for subject in subjects)
