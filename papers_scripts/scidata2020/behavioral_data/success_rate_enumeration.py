# -*- coding: utf-8 -*-
"""
Compute success rates for the performances of the IBC participants
for the Enumeration task

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


def enumeration_scores_extractor(participants, dir_path, numerosity = None):
    all_pt_scores = []
    all_pt_means = []
    # For each participant...
    for participant in participants:
        log_path = os.path.abspath(os.path.join(dir_path,
                                                'sub-' + '%02d' % participant,
                                                'knops/enumeration'))
        # Load the files
        log_files = glob.glob(os.path.join(log_path, '*.dat'))
        log_files.sort()
        # For every log file:
        runs = []
        all_scores = []
        count = 0
        for log_fname in log_files:
            print(log_fname)
            log_file = [line for line in csv.reader(open(log_fname),
                                                    delimiter='\t')]
            # Discard rows pertaining to trials from a run that didn't finish
            discarded_trials = len(log_file[1:]) % 96
            if  discarded_trials == 0:
                data_list = log_file[1:]
            else:
                data_list = log_file[1:][:-discarded_trials]
            # Start reading the log files row-by-row
            correct_answers = []
            answers = []
            for dt, data in enumerate(data_list):
                if numerosity is not None:
                    if numerosity == int(data[6]):
                        correct_answers.append(data[6])
                        answers.append(data[11])
                else:
                    correct_answers.append(data[6])
                    answers.append(data[11])
                # Compute scores for each run
                if dt in [95, 191]:
                    score = round(calc_score(correct_answers, answers), 0)
                    all_scores.append(score)
                    runs.append(str(count))
                    count = count + 1
                    # Clean arrays
                    correct_answers = []
                    answers = []
        # Compute mean of scores in all runs for each participant
        score_mean = np.rint(np.mean(all_scores))
        all_scores.append(score_mean)
        all_pt_means.append(score_mean)
        # Append total average per participant
        all_scores = ["%d" % s for s in all_scores]
        all_pt_scores.append(all_scores)
    # Compute mean and standard deviation for all participants
    group_mean = np.rint(np.mean(all_pt_means))
    group_mean = "%d" % group_mean
    group_std = np.rint(np.std(all_pt_means))
    group_std = "%d" % group_std
    return all_pt_scores, runs, group_mean, group_std


# # %%
# # ========================== GENERAL PARAMETERS =============================

# Inputs
pt_list = [1, 4, 5, 6, 7, 9, 11, 12, 13, 14, 15]

HERE = os.path.dirname(__file__)
intermediate_folder = 'neurospin_data/info/'
parent_dir = os.path.abspath(os.path.join(HERE, os.pardir,
                                          intermediate_folder))

task_name = 'enumeration'

# Outputs
pt_full = ['one', 'four', 'five', 'six', 'seven', 'nine', 'eleven', 'twelve',
           'thirteen', 'fourteen', 'fifteen']

HEADER = ['sub' + sub for sub in pt_full]
HEADER.insert(0, 'run')
HEADER.insert(0, 'numerosity')

main_dir = '/home/analu/mygit/ibc_ghub/' + \
           'admin/papers/data_paper2/behavioral_results'

# # %%
# # ================= COMPUTE SCORES AND GENERATE CSV FILES ===================

# All numerosities
print('Numerosities all together')
participants_scores, runs_id, gmean_all, gstd_all = \
    enumeration_scores_extractor(pt_list, parent_dir)

# Create csv file with individual rates for numerosities all together
csv_file = 'success_rate_' + task_name + '_' + 'all' + '.csv'
output_path = os.path.join(main_dir, csv_file)
generate_csv(participants_scores, runs_id, HEADER, output_path,
             numerosity='All numerosities')

# Start table with group means and stds
group_table = [['All numerosities', gmean_all, gstd_all]]

# Every numerosity
num = [1, 2, 3, 4, 5, 6, 7, 8]
for n in num:
    print(n)
    participants_scores, runs_id, gmean, gstd = \
        enumeration_scores_extractor(pt_list, parent_dir, numerosity = n)
    csv_file = 'success_rate_' + task_name + '_' + str(n) + '.csv'
    output_path = os.path.join(main_dir, csv_file)
    generate_csv(participants_scores, runs_id, HEADER, output_path,
                 numerosity=str(n))
    group_table.append([str(n), gmean, gstd])

# Create csv file with group rates for all numerosities
group_table = np.vstack((['numerosity', 'groupmean', 'groupstd'],
                         group_table))
group_csv = 'group_values_' + task_name + '.csv'
group_path = os.path.join(main_dir, group_csv)
# Save table in the output file
with open(group_path, 'w') as fp:
    a = csv.writer(fp, delimiter=',')
    a.writerows(group_table)
