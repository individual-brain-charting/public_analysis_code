# -*- coding: utf-8 -*-

import csv
import numpy as np


def calc_score(answer_list, pt_fdbk_list):
    """
    Calculate the final score of the participant based on the number of
    correct answers.
    """
    answer = np.array(answer_list)
    pt_fdbk = np.array(pt_fdbk_list)
    score_list = np.where(answer == pt_fdbk, 1, 0)
    score_list = [float(i) for i in score_list]
    score = np.sum(score_list)
    total_score = (score / answer.size) * 100
    return total_score


def generate_csv(pt_scores, runs_number, header, output, numerosity=None,
                 n_trials=None):
    """
    Generate csv file to be imported by data_paper2.tex
    """
    # Prepare some labeling arrays for table
    runs_number = [r for r in runs_number]
    runs_number.insert(len(runs_number), 'Mean')
    if numerosity is not None:
        if len(runs_number) % 2 == 0:
            row = len(runs_number) // 2 - 1
        else:
            row = len(runs_number) // 2
        num_column = [''] * len(runs_number)
        num_column[row] = numerosity
        # Stack all arrays in a table
        table = np.vstack((header, np.vstack((num_column, runs_number,
                                              pt_scores)).T))
    elif n_trials is not None:
        # Stack all arrays in a table
        n_trials.insert(len(runs_number), '-')
        table = np.vstack((header, np.vstack((runs_number, n_trials,
                                              pt_scores)).T))
    else:
        # Stack all arrays in a table
        table = np.vstack((header, np.vstack((runs_number, pt_scores)).T))
    # Save table in the output file
    with open(output, 'w') as fp:
        a = csv.writer(fp, delimiter=',')
        a.writerows(table)
