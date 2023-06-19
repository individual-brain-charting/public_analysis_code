import numpy as np
from nilearn import connectome
import pandas as pd
import os
from ibc_public.utils_data import get_subject_session
from glob import glob
import numpy as np


def get_ses_modality(task):
    """Get session numbers and modality for given task

    Parameters
    ----------
    task : str
        name of the task

    Returns
    -------
    sub_ses : dict
        dictionary with subject as key and session number as value
    modality : str
        modality of the task
    """
    if task == "GoodBadUgly":
        # session names with movie task data
        session_names = ["BBT1", "BBT2", "BBT3"]
    elif task == "MonkeyKingdom":
        # session names with movie task data
        session_names = ["monkey_kingdom"]
    elif task == "Raiders":
        # session names with movie task data
        session_names = ["raiders1", "raiders2"]
    elif task == "RestingState":
        # session names with RestingState state task data
        session_names = ["mtt1", "mtt2"]
    elif task == "DWI":
        # session names with diffusion data
        session_names = ["anat1"]
    # get session numbers for each subject
    # returns a list of tuples with subject and session number
    subject_sessions = sorted(get_subject_session(session_names))
    # convert the tuples to a dictionary with subject as key and session
    # number as value
    sub_ses = {}
    # for dwi, with anat1 as session_name, get_subject_session returns wrong
    # session number for sub-01 and sub-15
    # setting it to ses-12 for these subjects
    if task == "DWI":
        modality = "structural"
        sub_ses = {
            subject_session[0]: "ses-12"
            if subject_session[0] in ["sub-01", "sub-15"]
            else subject_session[1]
            for subject_session in subject_sessions
        }
    else:
        # for fMRI tasks, for one of the movies, ses no. 13 pops up for sub-11
        # and sub-12, so skipping that
        modality = "functional"
        for subject_session in subject_sessions:
            if (
                subject_session[0] in ["sub-11", "sub-12"]
                and subject_session[1] == "ses-13"
            ):
                continue
            # initialize a subject as key and an empty list as the value
            # and populate the list with session numbers
            # try-except block is used to avoid overwriting the list for subject
            try:
                sub_ses[subject_session[0]]
            except KeyError:
                sub_ses[subject_session[0]] = []
            sub_ses[subject_session[0]].append(subject_session[1])
    return sub_ses, modality


def get_all_subject_connectivity(
    sub_ses,
    task_modality,
    conn_measure,
    data_root,
    average_runs=False,
):
    """Get all subject connectivity for given task and connectivity measure.
    Handles path retrieval based on task modality and connectivity measure.

    Parameters
    ----------
    sub_ses : dict
        dictionary with subject as key and session number as value
    task_modality : str
        modality of the task
    conn_measure : str
        name of the connectivity measure
    data_root : str
        path to the root directory of the data
    average_runs : bool, optional
        whether to average the connectivity matrices across runs, by default
        False

    Returns
    -------
    list
        a list with flattened connectivity matrices for each session/run of
        all given subjects, sesssions and connectivity measure
    """

    if task_modality == "functional":
        ses_mod_dir = "func"
    elif task_modality == "structural":
        ses_mod_dir = "dwi"

    all_subject_connectivity = []

    for sub, sess in sub_ses.items():
        # if it is pearsons correlation, we calculated it for each run
        if conn_measure.split("_")[0] == "Pearsons":
            sess = sorted(sess)
            subject_connectivity = []
            for ses in sess:
                file_loc = os.path.join(data_root, sub, ses, ses_mod_dir)
                runs = glob(os.path.join(file_loc, f"*{conn_measure}*"))
                assert len(runs) > 0
                for run in runs:
                    pearson_mat = pd.read_csv(
                        run, index_col=False, header=None
                    )
                    pearson_mat_flat = connectome.sym_matrix_to_vec(
                        pearson_mat.to_numpy(), discard_diagonal=True
                    )
                    subject_connectivity.append(pearson_mat_flat)
            if average_runs:
                subject_connectivity = np.mean(subject_connectivity, axis=0)
        # if it is any other connectivity measure, we have only one estimate
        # over all runs
        else:
            if task_modality == "functional":
                sess = sorted(sess)
                ses = sess[-1]
            elif task_modality == "structural":
                ses = sess
            file_loc = os.path.join(data_root, sub, ses, ses_mod_dir)
            subject_connectivity = glob(
                os.path.join(file_loc, f"*{conn_measure}*")
            )
            assert len(subject_connectivity) == 1
            subject_connectivity = pd.read_csv(
                subject_connectivity[0], index_col=False, header=None
            )
            subject_connectivity = connectome.sym_matrix_to_vec(
                subject_connectivity.to_numpy(), discard_diagonal=True
            )
        all_subject_connectivity.append(subject_connectivity)

    return all_subject_connectivity
