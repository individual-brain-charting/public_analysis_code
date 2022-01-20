"""
Label individual topographies and
generation of functional fingerprints for each topography

Author: Bertrand Thirion, Ana Luisa Pinho 2020

Compatibility: Python 3.5

"""

import os
import json
import numpy as np
import pandas as pd

from utils_dictionary import (make_dictionary, dictionary2labels)
from ibc_public.utils_data import (horizontal_fingerprint, make_surf_db,
                                   DERIVATIVES, ALL_CONTRASTS)

from collections import Counter


def flatten(li):
    return sum(([x] if not isinstance(x, list) else flatten(x)
                for x in li), [])


def relabel(repeated_labels, r=1):
    for repeated_label in repeated_labels:
        # Indexes for each repeated label
        idx_labels_repetition = [index
                                 for index,value in enumerate(best_labels)
                                 if value == repeated_label]
        # Get the corresponding components in the dictionary
        dictionary_repetition = [dictionary[idx_comp]
                                 for idx_comp in idx_labels_repetition]
        # Get the max loading of the components with repeated labels
        loadings_repetition = [np.max(abs(comp_repeat))
                               for comp_repeat in dictionary_repetition]
        # Get idx of the component with the maximum loading
        idx_component_repetition_max = np.argmax(loadings_repetition)
        # Get the idx of the label that corresponds
        # to the max of the components with repeated labels
        idx_max = idx_labels_repetition[idx_component_repetition_max]
        # Substitute
        for idx in idx_labels_repetition:
            if idx != idx_max:
                if dictionary[idx][components_argsorted[idx][r]] < 0:
                    best_labels[idx] = negative_labels[
                        components_argsorted[idx][r]]
                else:
                    best_labels[idx] = positive_labels[
                        components_argsorted[idx][r]]


# #### INPUTS ####

#input_dir = './'
write_dir = '../../../admin/papers/descriptive_paper/supplementary_material/' + \
            'components/'
dictionary = np.load(os.path.join('dictionary.npz'), allow_pickle=True)['loadings']
contrasts = np.load(os.path.join('dictionary.npz'), allow_pickle=True)['contrasts']
contrasts = [x if x != 'jabberworcky-pseudo' else 'jabberwocky-pseudo'
             for x in contrasts]

task_list = ['ArchiEmotional', 'ArchiSocial', 'ArchiSpatial',
             'ArchiStandard', 'HcpEmotion', 'HcpGambling',
             'HcpLanguage', 'HcpMotor', 'HcpRelational',
             'HcpSocial', 'HcpWm',
             'RSVPLanguage']

# BIDS conversion of task names
# Load dictionary file
with open(os.path.join('bids_postprocessed.json'), 'r') as f:
    task_dic = json.load(f)

TASKS = [task_dic[tkey] for tkey in task_list]
TASKS = flatten(TASKS)

path = os.path.join(write_dir, 'labels.png')

df_all_contrasts = pd.read_csv(ALL_CONTRASTS, sep='\t')

negative_labels = [df_all_contrasts['negative label']\
               [df_all_contrasts.contrast == c].values[0]
               for c in contrasts]
positive_labels = [df_all_contrasts['positive label']\
                [df_all_contrasts.contrast == c].values[0]
                for c in contrasts]

# Takes for each component the index of the maximum loading
labels_argmax = [np.argmax(abs(component)) for component in dictionary]
# Sort loading of all components
components_argsorted= [np.argsort(abs(comp))[::-1] for comp in dictionary]


# #### FIRST LABELING ####
# Assigns the corresponding contrast label
best_labels = []
for a, label_arg in enumerate(labels_argmax):
    if dictionary[a][label_arg] < 0:
        best_labels.append(negative_labels[label_arg])
    else:
        best_labels.append(positive_labels[label_arg])


# #### RELABEL REPEATED LABELS ####
# Identify repeated labels
count=0
while len(best_labels) != len(set(best_labels)):
    repetitions = [k for (k,v) in Counter(best_labels).items() if v > 1]
    relabel(repetitions, r=count)
    count = count + 1
    index = int()
    # Remove 'any motion' label
    if np.any(np.array(best_labels) == 'any motion'):
        index = np.where(np.array(best_labels) == 'any motion')[0][0]
    if index:
        if dictionary[index][components_argsorted[index][count]] < 0:
            best_labels[index] = negative_labels[components_argsorted[index] \
                                                 [count]]
        else:
            best_labels[index] = positive_labels[components_argsorted[index] \
                                                 [count]]

# get contrasts
# db = make_surf_db(main_dir=DERIVATIVES)
# df = db[db.task.isin(task_list)]
# df = df.sort_values(by=['subject', 'task', 'contrast'])
# contrasts = df.contrast.unique()
# best_labels = [
#     'math',  # 'mental subtraction'
#     'saccades',  # 'random motion'
#     'consonant strings',
#     '2-back',  # 'math'
#     'random motion',  # 'saccades'
#     'place image',  # 'reward',#
#     'face image',
#     'button presses 1',
#     'visual matching',  # 'reward', #
#     'left foot',
#     'button presses 2',  # 'math',#
#     'fixation cross',
#     'tongue',
#     'social',
#     '0-back',  # 'silence', #
#     'sentence listening',
#     'fixation',  # '0-back',#
#     'story',  #
#     'read sentence',  # 'story',#
#     'read words',  # 'story'# #
#     ]

# Create png's with list of contrast labels
_ = dictionary2labels(dictionary, task_list, path,
                      facecolor=[.5, .5, .5], contrasts=contrasts,
                      best_labels=best_labels)
0/0
# Create png's with finger prints of cognitive components
filenames = [filename.replace(" ", "_") for filename in best_labels]
for i in range(20):
    output_file = os.path.join(write_dir, 'component_%s.png' % filenames[i])
    horizontal_fingerprint(dictionary[i], best_labels[i],
                           negative_labels, positive_labels,
                           output_file, wc=True, dpi=600)
