
import commands
import glob
import os

derivatives = '/neurospin/ibc/derivatives'
subjects = sorted(glob.glob(os.path.join(derivatives, 'sub-*')))

for subject in subjects:
    subject_id = os.path.basename(subject)
    src = os.path.join(subject, 'ses-00', 'anat', 'w%s_ses-00_T1w.nii.gz' % subject_id)
    dst = os.path.join(subject, 'ses-00', 'anat', 'w%s_ses-00_T1w_bet.nii.gz' % subject_id)
    commands.getoutput('fsl5.0-bet %s %s -f 0.3' % (src, dst))
    
