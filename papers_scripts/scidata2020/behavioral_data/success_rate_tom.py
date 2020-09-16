# -*- coding: utf-8 -*-
"""
Compute success rates for the performances of the IBC participants
for the theory-of-mind task

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


def tom_scores_extractor(participants, dir_path, correct_answers_1,
                         correct_answers_2):
    all_pt_scores = []
    # For each participant...
    for participant in participants:
        log_path = os.path.abspath(os.path.join(dir_path,
                                                'sub-' + '%02d' % participant,
                                                'tom/tom'))
        # Load the files
        log_files = glob.glob(os.path.join(log_path, '*.mat'))
        log_files.sort()
        # For every log file, i.e. for every run:
        runs = []
        all_scores = []
        for ll, log_fname in enumerate(log_files):
            print(log_fname)
            data_list = [line for line in csv.reader(open(log_fname),
                                                     delimiter=',')]
            answers = []
            counter_onset = 0
            # Read it line by line...
            for dt, data in enumerate(data_list):
                # ...and retrieve answers
                if data_list[dt - 4 - counter_onset] == ["# name: key"]:
                    if len(data) == 0:
                        break
                    else:
                        counter_onset = counter_onset + 1
                    answer = data[0].strip()
                    answers.append(answer)
            # Estimate score for the present run
            if participant == 9:
                score = round(calc_score(correct_answers_2[ll], answers), 2)
            else:
                score = round(calc_score(correct_answers_1[ll], answers), 2)
            all_scores.append(score)
            runs.append(str(ll))
        # Compute mean of scores in all runs for each participant
        score_mean = np.rint(np.mean(np.trim_zeros(all_scores)))
        all_scores.append(score_mean)
        # Append total average per participant
        all_scores = ["%d" % s for s in all_scores]
        all_pt_scores.append(all_scores)
    return all_pt_scores, runs


# # %%
# # ========================== GENERAL PARAMETERS =============================

# Inputs
pt_list = [1, 4, 5, 6, 7, 9, 11, 12, 13, 14, 15]

HERE = os.path.dirname(__file__)
intermediate_folder = 'neurospin_data/info/'
parent_dir = os.path.abspath(os.path.join(HERE, os.pardir,
                                          intermediate_folder))

all_right_answers = [['False', 'True', 'True', 'True', 'True', 'True', 'False',
                      'False', 'True', 'False'],
                     ['False', 'False', 'False', 'True', 'False', 'True',
                      'False', 'True', 'False', 'True']]

converted_right_answers_vs1 = []
converted_right_answers_vs2 = []
for right_answ in all_right_answers:
    cright_answ_1 = []
    cright_answ_2 = []
    cright_answ_1 = np.where(np.array(right_answ) == 'True', '89', '71')
    cright_answ_2 = np.where(np.array(right_answ) == 'True', '30', '43')
    converted_right_answers_vs1.append(cright_answ_1)
    converted_right_answers_vs2.append(cright_answ_2)

task_name = 'tom'

# Outputs
pt_full = ['one', 'four', 'five', 'six', 'seven', 'nine', 'eleven', 'twelve',
           'thirteen', 'fourteen', 'fifteen']

HEADER = ['sub' + sub for sub in pt_full]
HEADER.insert(0, 'run')

main_dir = '/home/analu/mygit/ibc_ghub/' + \
           'admin/papers/data_paper2/behavioral_results'

# # %%
# # =========================== COMPUTE SCORES ================================

participants_scores, runs_id = tom_scores_extractor(pt_list, parent_dir,
                                                converted_right_answers_vs1,
                                                converted_right_answers_vs2)

# Replace '0's by 'n/a' in scores of participants
participants_scores = [np.where(np.array(s) == '0', 'n/a', s)
                       for s in participants_scores]

csv_file = 'success_rate_' + task_name + '.csv'
output_path = os.path.join(main_dir, csv_file)
generate_csv(participants_scores, runs_id, HEADER, output_path)
