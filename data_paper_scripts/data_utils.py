"""
Common utilisities to all scripts
"""

import glob
import os
import pandas as pd

if 1:
    ibc = '/neurospin/ibc'
else:
    ibc = '/storage/store/data/ibc'
    
DERIVATIVES = os.path.join(ibc, 'derivatives')
SMOOTH_DERIVATIVES = os.path.join(ibc, 'smooth_derivatives')
SUBJECTS = [os.path.basename(full_path) for full_path in
            sorted(glob.glob(os.path.join(DERIVATIVES, 'sub-*')))]
CONDITIONS = pd.DataFrame().from_csv('conditions.tsv', sep='\t')
CONTRASTS = pd.DataFrame().from_csv('main_contrasts.tsv', sep='\t')


def data_parser(derivatives=DERIVATIVES, conditions=CONDITIONS,
                subject_list=SUBJECTS):
    """Generate a dataframe that contains all the data corresponding
    to the archi, hcp and rsvp_language acquisitions

    Parameters
    ----------
    derivatives: string, optional
        path toward a valid BIDS derivatives directory
    
    conditions: pandas DataFrame, optional,
        dataframe describing the conditions under considerations

    subject_list: list, optional,
        list of subjects to be included in the analysis

    Returns
    -------
    db: pandas DataFrame,
        "database" yielding informations (path,
        subject, modality, contrast, session, task, acquisition) 
        on the images under consideration
    """
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
        
    # gm images
    imgs_ = sorted(glob.glob(os.path.join(
        derivatives, 'sub-*/ses-*/anat/mwcc1sub*_T1w.nii.gz')))
    for img in imgs_:
        session = img.split('/')[-3]
        subject = img.split('/')[-4]
        paths.append(img)
        sessions.append(session)
        subjects.append(subject)
        modalities.append('T1')
        contrasts.append('gm')
        tasks.append('')
        acquisitions.append('')
        
    # wm image
    imgs_ = sorted(glob.glob(os.path.join(
        derivatives, 'sub-*/ses-*/anat/mwc2sub*_T1w.nii.gz')))
    for img in imgs_:
        session = img.split('/')[-3]
        subject = img.split('/')[-4]
        paths.append(img)
        sessions.append(session)
        subjects.append(subject)
        modalities.append('T1')
        contrasts.append('wm')
        tasks.append('')
        acquisitions.append('')

    # fixed-effects activation images
    con_df = conditions
    if 'condition' in conditions.keys():
        contrast_name = con_df.condition
    elif 'contrast' in conditions.keys():
        contrast_name = con_df.contrast

    for acq in ['ap', 'pa', 'ffx']:
        for subject in subject_list:
            for i in range(len(con_df)):
                contrast = contrast_name[i]
                task = con_df.task[i]
                task_name = task
                if task == 'language_':
                    task = 'language_*'
                    task_name = 'rsvp_language'

                wildcard = os.path.join(
                    derivatives, '%s/*/res_stats_%s_%s/stat_maps/%s.nii.gz' %
                    (subject, task, acq, contrast))
                imgs_ = glob.glob(wildcard)
                if len(imgs_) == 0:
                    stop
                imgs_.sort()
                # some renaming
                contrast_id = contrast
                if ((contrast_id == 'probe') and
                    (task_name == 'rsvp_language')):
                    contrast_id = 'language_probe'                    

                for img in imgs_:
                    session = img.split('/')[-4]
                    paths.append(img)
                    sessions.append(session)
                    subjects.append(img.split('/')[-5])
                    modalities.append('bold')
                    contrasts.append(contrast_id)
                    tasks.append(task_name)
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
