"""
Analysis of the coverage of brain maps.

Authors: Bertrand Thirion, Ana Luisa Pinho 2020

Compatibility: Python 3.5

"""

import os
import glob
import json
import warnings

from joblib import Memory

import numpy as np
import pandas as pd

import nibabel as nib
from nilearn.input_data import NiftiMasker
from nilearn import plotting


# ############################### INPUTS ######################################

DERIVATIVES = '/neurospin/ibc/derivatives'
SMOOTH_DERIVATIVES = '/neurospin/ibc/smooth_derivatives'

# ### First + Second Releases (cumulative) ###
# Extract data from dataframe only referring to a pre-specified subset of
# participants
# Tasks terminology must conform with the one used in all_contrasts.tsv and
# main_contrasts.tsv

# sub_no = [1, 2, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15]
# task_list = ['archi_standard', 'archi_spatial',
#              'archi_social', 'archi_emotional',
#              'hcp_emotion', 'hcp_gambling', 'hcp_motor', 'hcp_language',
#              'hcp_relational', 'hcp_social', 'hcp_wm',
#              'rsvp_language',
#              'mtt_we', 'mtt_sn',
#              'preference',
#              'theory_of_mind', 'emotional_pain', 'pain_movie',
#              'vstm', 'enumeration',
#              'self', 'bang']
# output_fname = 'coverage_cumulative.pdf'


# ### Second Release (delta) ###

sub_no = [1, 4, 5, 6, 7, 9, 11, 12, 13, 14, 15]
task_list = ['mtt_we', 'mtt_sn',
             'preference',
             'theory_of_mind', 'emotional_pain', 'pain_movie',
             'vstm', 'enumeration',
             'self', 'bang']
output_fname = 'coverage_delta.png'


# ### Fmap of single task ###

# sub_no = [1, 4, 5, 6, 7, 9, 11, 12, 13, 14, 15]
# task_list = ['pain_movie']
# output_fname = 'fmap_%s.pdf' % task_list[0]


cache = '/neurospin/tmp/agrilopi/brain_coverage_2'
# cache = os.path.join(os.getcwd(), 'figures')
# if not os.path.exists(cache):
#     os.makedirs(cache)

# #############################################################################

# Define subjects' paths
sub_path = [os.path.join(DERIVATIVES, 'sub-%02d' % s) for s in sub_no]
SUBJECTS = [os.path.basename(full_path) for full_path in sub_path]

# BIDS conversion of task names
# Load dictionary file
with open(os.path.join('bids_postprocessed.json'), 'r') as f:
    task_dic = json.load(f)

TASK_BATTERIES = [task_dic[tkey] for tkey in task_list]
TASK_BATTERIES = [item for sublist in TASK_BATTERIES for item in sublist]

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

    for sbj in SUBJECTS:
        # T1 images
        t1_relat_path = 'sub-*/ses-00/anat/w%s_ses-00_T1w_nonan.nii.gz' % sbj
        t1_abs_path = os.path.join(derivatives, t1_relat_path)
        t1_imgs_ = glob.glob(os.path.join(t1_abs_path))
        if not t1_imgs_:
            msg = 'w%s_T1w_nonan.nii.gz file not found!' % (sbj)
            warnings.warn(msg)
        for img in t1_imgs_:
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
    # for acq in ['ap', 'pa', 'ffx']:
    for acq in ['ap', 'pa']:
        for task in TASK_BATTERIES:
            for sbj in SUBJECTS:
                res_stats = '%s/*/res_stats_%s*_%s*' % (sbj, task, acq)
                imgs__path = os.path.join(derivatives, res_stats,
                                          'z_score_maps',
                                          'effects_interest.nii.gz')
                imgs_ = glob.glob(imgs__path)
                if not imgs_:
                    # Add exception for 'bang' task, since 'ap' was
                    # never part of acq planning
                    if acq == 'ap' and task == 'bang':
                        pass
                    else:
                        task_ = task
                        if task == 'language':
                            task_ = 'rsvp_language'
                        msg = 'effects_interest.nii.gz for task %s %s in ' \
                              '%s not found!' % (task_, acq, sbj)
                        warnings.warn(msg)
                imgs_.sort()
                for img in imgs_:
                    session = img.split('/')[5]
                    paths.append(img)
                    sessions.append(session)
                    subjects.append(img.split('/')[4])
                    modalities.append('bold')
                    contrasts.append('effects_interest')
                    tasks.append(task)
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
    # create a DataFrame out of the dictionary and write it to disk
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
        print(anat)
        z = stouffer(X[df.subject.values == subject])
        plotting.plot_stat_map(masker.inverse_transform(z), bg_img=anat, threshold=5.)

    z = stouffer(X)
    brain_covg = plotting.plot_stat_map(masker.inverse_transform(z),
                                        threshold=5., display_mode='x', cut_coords=5)
    brain_covg.savefig(os.path.join(cache, output_fname), dpi=1200)
