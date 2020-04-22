"""
Analysis of the coverage of brain maps.

Authors: Bertrand Thirion, 2017
"""
import numpy as np
import os
import glob
import pandas as pd
from joblib import Memory, Parallel, delayed
import nibabel as nib
from nilearn.input_data import NiftiMasker
from nilearn import plotting

DERIVATIVES = '/neurospin/ibc/derivatives'
SMOOTH_DERIVATIVES = '/neurospin/ibc/smooth_derivatives'
SUBJECTS = [os.path.basename(full_path) for full_path in
            sorted(glob.glob(os.path.join(DERIVATIVES, 'sub-*')))]
CONDITIONS = pd.DataFrame().from_csv('../processing/conditions.tsv', sep='\t')
cache = '/neurospin/tmp/bthirion'
mem = Memory(cachedir=cache, verbose=0)

def stouffer(x):
    return x.mean(0) * np.sqrt(x.shape[0])

def eoi_parser(derivatives=DERIVATIVES):
    """Generate a dataframe that contains all the data corresponding
    to the archi, hcp and rsvp_language acquisitions"""
    paths = []
    subjects = []
    sessions = []
    modalities = []
    contrasts = []
    tasks = []
    acquisitions = []

    # T1 images
    imgs_ = sorted(glob.glob(os.path.join(
        derivatives, 'sub-*/ses-*/anat/wsub*_T1w_nonan.nii.gz')))
    for img in imgs_:
        session = img.split('/')[-3]
        subject = img.split('/')[-4]
        paths.append(img)
        sessions.append(session)
        subjects.append(subject)
        modalities.append('T1')
        contrasts.append('t1')
        tasks.append('')
        acquisitions.append('')
        
    # fixed-effects activation images
    task_list = CONDITIONS.task.unique()
    for acq in ['ap', 'pa', 'ffx']:
        for task in task_list:
            for subject in SUBJECTS:
                imgs_ = glob.glob(os.path.join(
                    derivatives, '%s/*/res_stats_%s_%s/z_score_maps/effects_interest.nii.gz' %
                    (subject, task, acq)))
                imgs_.sort()

                task_ = task
                if task_ == 'language_':
                    task_ = 'rsvp_language'

                for img in imgs_:
                    session = img.split('/')[5]
                    paths.append(img)
                    sessions.append(session)
                    subjects.append(img.split('/')[4])
                    modalities.append('bold')
                    contrasts.append('effects_interest')
                    tasks.append(task_)
                    acquisitions.append(acq)
    
    # create a dictionary with all the information
    db_dict = dict(
        path=paths,
        subject=subjects,
        modality=modalities,
        contrast=contrasts,
        session=sessions,
        task=tasks,
        acquisition=acquisitions,
    )
    # create a FataFrame out of the dictionary and write it to disk
    db = pd.DataFrame().from_dict(db_dict)
    return db

if __name__ == '__main__':
    db = eoi_parser(derivatives=SMOOTH_DERIVATIVES)
    mask_gm = nib.load(os.path.join(DERIVATIVES, 'group', 'anat', 'gm_mask.nii.gz'))
    masker = NiftiMasker(mask_img=mask_gm).fit()
    df = db[db.modality == 'bold']
    X = masker.transform(df.path.values)

    # per-subject EoI
    for subject in SUBJECTS:
        anat = db[db.modality == 'T1'][db.subject == subject].path.values[0]
        z = stouffer(X[df.subject.values == subject])
        plotting.plot_stat_map(masker.inverse_transform(z), bg_img=anat, threshold=5.)

    z = stouffer(X)
    plotting.plot_stat_map(masker.inverse_transform(z), threshold=5.,
                           display_mode='x', cut_coords=5,
                           output_file=os.path.join('output/coverage.pdf'))
    plotting.show()
