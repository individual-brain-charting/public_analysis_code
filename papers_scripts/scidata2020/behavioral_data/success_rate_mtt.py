# -*- coding: utf-8 -*-
"""
Compute success rates for the mtt-task performances of the IBC participants

author: Ana Luisa Pinho
e-mail: ana.pinho@inria.fr

Compatibility: Python 3.5

Date: September 2019
"""

import os
# import sys
import glob
import csv
import numpy as np

# Add momentarily the parent dir to the path in order to call 'scores' module
# new_path = os.path.abspath(os.pardir)
# if new_path not in sys.path:
#     sys.path.append(new_path)

from behav_utils import calc_score


def mtt_scores_extractor(participants, dir_path, events = None):
    all_pt_scores = []
    # For each participant...
    for participant in participants:
        # For each session...
        all_sess_scores = []
        only_scores = []
        for session in ['we', 'sn']:
            log_path = os.path.abspath(os.path.join(dir_path,
                                                    'sub-' + \
                                                    '%02d' % participant,
                                                    'mtt', 'log_' + session))
            # Load the files
            log_files = glob.glob(os.path.join(log_path, '*.xpd'))
            log_files.sort()
            # Handle exception for participant 7, session 'we':
            # - two acqs and thus two log files;
            # - the last acq. and thus the last log file is the good one.
            if participant == 7 and session == 'we':
                # Take the last element as a list
                log_files = log_files[len(log_files)-1:]
            # For every log file:
            runs = []
            all_scores = []
            for ll, log_fname in enumerate(log_files):
                print(log_fname)
                # Handle exception for participant 15:
                # - no data from session 'sn' was recorded.
                if participant == 15 and session == 'sn':
                    runs = np.arange(3)
                    runs = list(map(str, runs))
                    all_scores = [0] * len(runs)
                    break
                log_file = [line for line in csv.reader(open(log_fname),
                                                        delimiter=',')]
                # Retrieve table from log_file
                for r, row in enumerate(log_file):
                    if row[0] == str(participant):
                        break
                # Handle exception for participant 15, session 'we':
                # - not many trials were registered for the first runs;
                # - no trial in this run has a record of the sub's responses.
                if participant == 15 and session == 'we' and ll == 0:
                    discarded_trials = 0
                else:
                    discarded_trials = len(log_file[r:]) % 200
                # Discard last trials from log files of acq. when interrupted
                if  discarded_trials == 0:
                    data_list = log_file[r:]
                else:
                    data_list = log_file[r:][:-discarded_trials]
                # Extract arrays
                runn = []
                trial = []
                answers = []
                right_answers = []
                flag = 0
                for d, data in enumerate(data_list):
                    # To take into account answers given during the events
                    if events is not None and data[8] in events and \
                       data[12] != 'None':
                        # If the answer is given during the first half of
                        # the event, it is considered an answer of
                        # the previous event
                        # (except for the first event of the trial)
                        if len(trial) > 0 and int(data[11]) < 1000:
                            answers[-1] = data[12]
                        # If in the second half,
                        # it is considered an answer of the current event
                        else:
                            answers.append(data[12])
                            flag = 1
                    # Read answers of response trials from data list
                    if data[8] == 'response':
                        runn.append(data[1])
                        trial.append(data[2])
                        right_answers.append(data[13])
                        if flag == 0:
                            answers.append(data[12])
                        flag = 0
                    # Compute scores of every run
                    if (len(runn) > 0 and data[1] != runn[-1]) or \
                       d == len(data_list) - 1:
                        if session == 'we':
                            converted_right_answers = np.where(
                                np.isin(right_answers, ['before', 'west'],
                                        ['after', 'east']), 'y', 'b')
                        elif session == 'sn':
                            converted_right_answers = np.where(
                                np.isin(right_answers, ['before', 'south'],
                                        ['after', 'north']), 'y', 'b')
                        score = round(calc_score(converted_right_answers,
                                                 answers), 2)
                        all_scores.append(score)
                        runs.append(runn[-1])
                        runn = []
                        answers = []
                        right_answers = []
                    # Empty trial array in the end of every trial
                    if len(trial) == 4:
                        trial = []
            # Compute mean of scores in all runs for each session
            # of each participants
            only_scores.extend(all_scores)
            if np.trim_zeros(all_scores):
                score_mean = np.rint(np.mean(np.trim_zeros(all_scores)))
            else:
                score_mean = np.rint(np.mean(all_scores))
            all_scores.append(score_mean)
            # Scores from both sessions for each participant
            all_sess_scores.extend(all_scores)
        # Compute and append total average (i.e. for both scores)
        only_scores = np.trim_zeros(only_scores)
        total_score_mean = np.rint(np.mean(only_scores))
        all_sess_scores.append(total_score_mean)
        all_sess_scores = ["%d" % s for s in all_sess_scores]
        # Scores from all participants
        all_pt_scores.append(all_sess_scores)
    return all_pt_scores, runs


# %%
# ========================== GENERAL PARAMETERS ===============================

# Inputs
pt_list = [1, 4, 5, 6, 7, 9, 11, 12, 13, 14, 15]

HERE = os.path.dirname(__file__)
intermediate_folder = 'neurospin_data/info/'
parent_dir = os.path.abspath(os.path.join(HERE, os.pardir,
                                          intermediate_folder))
main_dir = '/home/analu/mygit/ibc_ghub/' + \
           'admin/papers/data_paper2/behavioral_results'

EVENTS_WE = ['maladie_inconnue', 'accostage_falaise', 'provision_deau',
             'perte_de_nourriture', 'eboulis_contournes', 'attaque_crocodile',
             'minerai_embarque', 'pommade_anti_moustique', 'recifs_evites',
             'peche_de_poisson', 'soiree_musicale', 'lavage_des_vetements',
             'traces_danimaux', 'arrivee_sur_la_plage', 'decouverte_de_fer',
             'analyse_de_la_roche']
EVENTS_SN = ['priere_aux_morts', 'fumee_observee', 'discussions_tranquilles',
             'esprit_du_marais', 'oiseaux_pour_augures',
             'roseaux_pour_offices', 'onguent_contre_parasite',
             'arret_pour_manger', 'vivres_charges', 'position_des_astres',
             'bain_purificateur', 'procession_terminee', 'commerce_local',
             'arc_en_ciel_observe', 'construction_de_radeaux',
             'retour_au_foyer']

EVENTS = EVENTS_WE + EVENTS_SN

# Outputs
pt_full = ['one', 'four', 'five', 'six', 'seven', 'nine', 'eleven', 'twelve',
           'thirteen', 'fourteen', 'fifteen']

HEADER = ['sub' + sub for sub in pt_full]
HEADER.insert(0, 'run')
HEADER.insert(0, 'session')

# %%
# =========================== COMPUTE SCORES ==================================

# With no corrections
# participants_scores, runs_id = mtt_scores_extractor(pt_list, parent_dir)
# output_filename = 'success_rate_mtt_no_corrections.csv'
# group_csv = 'group_values_mtt_no_corr.csv'

# # With corrections
participants_scores, runs_id = mtt_scores_extractor(pt_list, parent_dir,
                                                    events = EVENTS)
output_filename = 'success_rate_mtt_with_corrections.csv'
group_csv = 'group_values_mtt_with_corr.csv'

# %%
# ========================== CREATE CSV FILE  =================================

# Prepare some labeling arrays for table
runs_id = [r for r in runs_id] * 2
runs_id.insert(3, 'Mean')
runs_id.insert(7, 'Mean')
runs_id.insert(len(runs_id), 'Total')
session_names = [''] * len(runs_id)
session_names[1] = 'MTT WE'
session_names[5] = 'MTT SN'

# Replace '0's by 'n/a' in scores of participants
participants_scores = [np.where(np.array(s) == '0', 'n/a', s)
                       for s in participants_scores]

# Stack all arrays in a table
table = np.vstack((HEADER, np.vstack((session_names, runs_id,
                                      participants_scores)).T))

# Save table in the output file
parent_dir = '/home/analu/mygit/ibc_ghub'
output_path = os.path.join(parent_dir,
                           'admin/papers/data_paper2/behavioral_results')

output = os.path.join(output_path, output_filename)
with open(output, 'w') as fp:
    a = csv.writer(fp, delimiter=',')
    a.writerows(table)

# %%
# ========================== GROUP MEANS  =====================================

mtt_1 = [int(i[3]) for i in participants_scores]
mtt_2 = [int(j[7]) for j in participants_scores if not j[7] == 'n/a']
mtt_total = mtt_1 + mtt_2

mtt_1_mean = "%d" % np.rint(np.mean(mtt_1))
mtt_1_std = "%d" % np.rint(np.std(mtt_1))
mtt_2_mean = "%d" % np.rint(np.mean(mtt_2))
mtt_2_std = "%d" % np.rint(np.std(mtt_2))
mtt_total_mean = "%d" % np.rint(np.mean(mtt_total))
mtt_total_std = "%d" % np.rint(np.std(mtt_total))

group_table = np.vstack((['session', 'groupmean', 'groupstd'],
                         ['MTT WE', mtt_1_mean, mtt_1_std],
                         ['MTT SN', mtt_2_mean, mtt_2_std],
                         ['Total', mtt_total_mean, mtt_total_std]))

group_path = os.path.join(main_dir, group_csv)
# Save table in the output file
with open(group_path, 'w') as fp:
    a = csv.writer(fp, delimiter=',')
    a.writerows(group_table)
