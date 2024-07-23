"""
Analysis of the coverage of brain maps.
Authors: Bertrand Thirion, Ana Luisa Pinho 2020
Compatibility: Python 3.5
"""

# %%
# Importing neeeded stuff

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

from ibc_public.utils_data import (data_parser, DERIVATIVES,
                                   SMOOTH_DERIVATIVES, CONDITIONS)
# %%
# ############################### INPUTS ######################################

# ### Third Release ###

sub_no = [4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15]
task_list = ['MathLanguage', 'SpatialNavigation', 'EmoReco', 'EmoMem',
             'StopNogo', 'Catell', 'FingerTapping', 'VSTMC',
             'BiologicalMotion1', 'BiologicalMotion2',
             'Checkerboard','FingerTap','ItemRecognition', 'BreathHolding'
             ]
output_fname = 'coverage_third.png'

cache = '/storage/store3/work/aponcema/IBC_paperFigures/brain_coverage3/'\
        'cache_brain_coverage3_7jun'

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

df_conds = CONDITIONS
trick_cond = (df_conds['task'] == 'Catell') & (df_conds['contrast'] == 'easy_oddball')
df_conds.loc[trick_cond, 'contrast'] = 'easy'
trick_cond = (df_conds['task'] == 'Catell') & (df_conds['contrast'] == 'hard_oddball')
df_conds.loc[trick_cond, 'contrast'] = 'hard'

trick_cond = df_conds['contrast'].isna()
df_conds.loc[trick_cond, 'contrast'] = 'null'

mem = Memory(location=cache, verbose=0)
# %%

def stouffer(x):
    return x.mean(0) * np.sqrt(x.shape[0])
# %%

if __name__ == '__main__':
    db = data_parser(derivatives=SMOOTH_DERIVATIVES, subject_list = SUBJECTS,
                     task_list=TASK_BATTERIES)
    mask_gm = nib.load(os.path.join(DERIVATIVES, 'group', 'anat',
                                    'gm_mask.nii.gz'))
    masker = NiftiMasker(mask_img=mask_gm).fit()
    # %%
    df = db[db.modality == 'bold']
    X = masker.transform(df.path.values)
# %%
    # per-subject EoI
    for subject in SUBJECTS:
        anat = db[db.modality == 'T1'][db.subject == subject].path.values[0]
        print(anat)
        z = stouffer(X[df.subject.values == subject])
        plotting.plot_stat_map(masker.inverse_transform(z),
                               bg_img=anat, threshold=5.)

    z = stouffer(X)
    brain_covg = plotting.plot_stat_map(masker.inverse_transform(abs(z)),
                                        threshold=5., display_mode='x',
                                        cut_coords=5)
    brain_covg.savefig(os.path.join(cache, output_fname), dpi=1200)