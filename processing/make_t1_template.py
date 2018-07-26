"""
This script is meant to create a nice(r)
t1 image template for better rendering
"""
import os
import glob
from pypreprocess.nipype_preproc_spm_utils import (do_subjects_preproc,
                                                   SubjectData)
from nilearn.image import mean_img
import nibabel as nib

# Set firectories first
data_dir = '/neurospin/ibc/derivatives'
scratch = '/neurospin/tmp/ibc'
ref_img = os.path.join(data_dir, 'sub-01/ses-00/mask.nii.gz')
output_dir = os.path.join(data_dir, 'group', 'anat')
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# glob for subject ids
subject_id_wildcard = 'sub-*'
subject_ids = [os.path.basename(x)
               for x in glob.glob(os.path.join(data_dir, subject_id_wildcard))]

anats = glob.glob(
    os.path.join(
    data_dir, 'sub*', 'ses-*', 'anat', 'sub-*_ses-*_acq-highres_T1w.nii'))
for anat in anats:
    img = nib.load(anat)
    nib.Nifti1Image(img.get_data().astype('int32'), img.affine).to_filename(anat)

# producer subject data
def subject_factory():
    anats = glob.glob(
        os.path.join(
            data_dir, 'sub*', 'ses-*', 'anat', 'sub-*_ses-*_acq-highres_T1w.nii'))
    subject_sessions = [(anat.split('/')[-4], anat.split('/')[-3]) for anat in anats]
    subject_sessions = [('sub-01', 'ses-12')]
    for subject_session in subject_sessions:
        subject, session = subject_session
        subject_data = SubjectData(isdicom=False, scratch=scratch, session_output_dirs=[], n_sessions=0)
        subject_data.subject_id = subject
        subject_data.anat = os.path.join(data_dir, subject, session, 'anat',
                                         '%s_%s_acq-highres_T1w.nii' % (subject, session))
        subject_data.func = []
        subject_data.output_dir = os.path.join(
            data_dir, subject, session, 'anat', 'dartel')
        # yield data for this subject
        yield subject_data
    
# do preprocessing proper
report_filename = os.path.join(output_dir, '_report.html')

do_subjects_preproc(
    subject_factory(),
    dataset_id='ibc',
    output_dir=output_dir,
    do_report=True,
    do_dartel=True,
    dataset_description="ibc",
    report_filename=report_filename,
    do_shutdown_reloaders=True,)


# Create mean images for masking and display
wanats = sorted(glob.glob(os.path.join(data_dir, 'sub-*', 'ses-*', 'anat', 'dartel',
                                'w*_ses-*_acq-highres_T1w.nii.gz')))
template = mean_img(wanats)
template.to_filename(os.path.join(output_dir, 'highres_T1avg.nii.gz'))

mgms = sorted(glob.glob(os.path.join(data_dir, 'sub-*', 'ses-*', 'anat', 'dartel',
                                     'mwc1*_ses-*_acq-highres_T1w.nii.gz')))

# take a reference functional image
ref_image = nib.load(ref_img)
ref_affine = ref_image.affine
ref_shape = ref_image.shape
mean_gm = mean_img(
    mgms, target_affine=ref_affine, target_shape=ref_shape) 
gm_mask = nib.Nifti1Image((mean_gm.get_data() > .25).astype('uint8'),
                          ref_affine)
mean_gm.to_filename(os.path.join(output_dir, 'mean_highres_gm.nii.gz'))
gm_mask.to_filename(os.path.join(output_dir, 'highres_gm_mask.nii.gz'))
