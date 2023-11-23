"""
Utilities to run FastSRM in the IBC dataset

Author: Ana Luisa Pinho

Created: October 2020
Last revision: May 2021

Compatibility: Python 3.9.1
"""

import os
import glob
import re
import numpy as np

from ibc_public.utils_data import DERIVATIVES


def flatten(li):
    return sum(([x] if not isinstance(x, list) else flatten(x)
                for x in li), [])


def reshape_preprocdata(participants_list, tasks, preprocdata,
                        input_type='vol'):
    """
    Return list of lists of lists: (n_subjects, n_tasks, n_runs)
    """
    files = []
    for p in participants_list:
        pt_files = []
        for t in tasks:
            task_files = []
            if input_type == 'vol':
                fname = 'wrdcsub-%02d_ses-*_task-%s' % (p, t) + \
                        '_dir-*_run-*_bold_masked.npy'
                fpath = os.path.join(preprocdata, fname)
                match_expression = '.*run-(..)_bold_masked.npy'
            else:
                assert input_type == 'surf'
                if preprocdata == DERIVATIVES:
                    fname = 'rdcsub-%02d_ses-*_task-%s' % (p, t) + \
                            '_dir-*_run-*_bold_fsaverage_*'
                    fpath = os.path.join(preprocdata, 'sub-*/ses-*/freesurfer',
                                         fname)
                else:
                    fname = 'rdcsub-%02d_ses-*_task-%s' % (p, t) + \
                            '_dir-*_run-*_bold_fsaverage*'
                    fpath = os.path.join(preprocdata, fname)
                match_expression = '.*run-(..)_bold_fsaverage*'
            # List paths
            task_files = glob.glob(fpath)
            # Sort in ascending order by run number
            run_numbers = [int(re.match(match_expression,
                                        task_file).groups()[0])
                           for task_file in task_files]
            indices_order = flatten([
                [r for r, run_number in enumerate(run_numbers, 1)
                 if run_number == j]
                for j in np.arange(1, len(run_numbers) + 1)])
            task_files_sorted = [task_files[k-1] for k in indices_order]
            # Append list of runs for one task
            pt_files.append(task_files_sorted)
        files.append(pt_files)
    return files


def stacker(subjects_set, tasks_set, data_paths):
    """
    Inputs a list of lists of lists w/ shape (n_subjects, n_tasks, n_runs)
    Outputs numpy array of paths w/ shape (n_subjects, n_runs)
    Note: runs follow order of tasks and run number within tasks
    """
    output_array = []
    for ss, subject_set in enumerate(subjects_set):
        task_array = []
        for task_set in tasks_set:
            task_array.extend(data_paths[subject_set][task_set])
        if ss < len(subjects_set) - 1:
            output_array.append(task_array)
        elif ss == len(subjects_set) - 1 and ss != 0:
            output_array = np.vstack((output_array, task_array))
        else:
            assert len(subjects_set) - 1 == 0
            output_array = np.array(task_array)

    return output_array
