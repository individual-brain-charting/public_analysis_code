# -*- coding: utf-8 -*-
"""
Compute success rates for the performances of the IBC participants
during the recognition phase of the self task

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

from behav_utils import calc_score, generate_csv


def self_scores_extractor(participants, dir_path):
    all_pt_scores = []
    # For each participant...
    for participant in participants:
        log_path = os.path.abspath(os.path.join(dir_path,
                                                'sub-' + '%02d' % participant,
                                                'self'))
        # Load the files
        log_files = glob.glob(os.path.join(log_path, '*.xpd'))
        log_files.sort()
        # For every log file:
        runs = []
        all_scores = []
        trials_number = []
        for log_fname in log_files:
            print(log_fname)
            log_file = [line for line in csv.reader(open(
                log_fname, encoding='iso-8859-1'), delimiter=',')]
            # Retrieve table from log_file
            for r, row in enumerate(log_file):
                if row[0] == str(participant):
                    break
            # Discard last trials from log files of acq. when interrupted
            if len(log_file[r:]) < 533:
                discarded_trials = len(log_file[r:]) % 123
            else:
                discarded_trials = len(log_file[r:]) % 533
            if discarded_trials == 0:
                data_list = log_file[r:]
            else:
                data_list = log_file[r:][:-discarded_trials]
            # Extract data
            runn = []
            trial_type = []
            answers = []
            right_answers = []
            for d, data in enumerate(data_list):
                # Retrieve answers from "recognition" trials
                if data[5] == 'recognition':
                    runn.append(data[1])
                    trial_type.append(data[6])
                    answers.append(data[10])
                # Compute the scores for every run
                if (len(runn) > 0 and data[1] != runn[-1]) or \
                   d == len(data_list) - 1:
                    right_answers = np.where(np.array(trial_type) == '0',
                                             'g', 'y')
                    score = round(calc_score(right_answers, answers), 2)
                    trials_number.append(str(len(answers)))
                    all_scores.append(score)
                    runs.append(runn[-1])
                    runn = []
                    trial_type = []
                    answers = []
                    right_answers = []
            # Compute mean of scores in all runs for each session
            # of each participant
            score_mean = np.rint(np.mean(all_scores))
            all_scores.append(score_mean)
        # Append total average per participant
        all_scores = ["%d" % s for s in all_scores]
        all_pt_scores.append(all_scores)
    return all_pt_scores, runs, trials_number


# # %%
# # ========================== GENERAL PARAMETERS =============================

# Inputs
pt_list = [1, 4, 5, 6, 7, 9, 11, 12, 13, 14, 15]

HERE = os.path.dirname(__file__)
intermediate_folder = 'neurospin_data/info/'
parent_dir = os.path.abspath(os.path.join(HERE, os.pardir,
                                          intermediate_folder))

task_name = 'self'

# Outputs
pt_full = ['one', 'four', 'five', 'six', 'seven', 'nine', 'eleven', 'twelve',
           'thirteen', 'fourteen', 'fifteen']

HEADER = ['sub' + sub for sub in pt_full]
HEADER.insert(0, 'trials')
HEADER.insert(0, 'run')

main_dir = '/home/analu/mygit/ibc_ghub/' + \
           'admin/papers/data_paper2/behavioral_results'

# # %%
# # =========================== COMPUTE SCORES ================================

participants_scores, runs_id, trials = self_scores_extractor(pt_list,
                                                             parent_dir)
csv_file = 'success_rate_' + task_name + '.csv'
output_path = os.path.join(main_dir, csv_file)
generate_csv(participants_scores, runs_id, HEADER, output_path,
             n_trials = trials)
