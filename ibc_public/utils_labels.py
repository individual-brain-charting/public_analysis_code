import json
import sys
import warnings

import numpy as np
import pandas as pd

import os

_package_directory = os.path.dirname(os.path.abspath(__file__))

ALL_CONTRASTS = os.path.join(_package_directory, '..', 'ibc_data',
                             'all_contrasts.tsv')


def get_labels(contrasts='all'):
    """
    Returns the list of labels for each passed contrast name

    Parameters
    ----------
    contrasts: list of str, default 'all'
               Each element of the list is a contrast name in the document.
               The default argument will select all present contrasts

    Returns
    -------
    contrast_dict: dict
                   Dictionary containing the contrasts provided by the user
                   as keys, and their corresponding labels as values
    """
    df = pd.read_csv(ALL_CONTRASTS, sep='\t')
    if contrasts == 'all':
        contrasts = df['contrast'].values

    contrast_dict = {}

    con_slice = df[df['contrast'].isin(contrasts)]
    not_found = np.setdiff1d(contrasts, con_slice['contrast'])

    if not_found.size != 0:
        warnings.warn("The following contrast names were not "
                      "found: {}".format(not_found))

    for index, con in con_slice.iterrows():

        labels = con.loc[con == 1.0].index
        con_name = "({}) {}".format(con.task, con.contrast)
        contrast_dict[con_name] = [label for label in labels]

    return contrast_dict


def add_labels(contrast, labels, output_file=ALL_CONTRASTS):
    """
    Adds all the passed labels to the selected contrast

    Paramenters
    -----------
    contrast: str
              Name of the contrast that will get the labels

    labels: list of str
            Labels that the user wants to add. The labels must exist as
            columns in the file

    output_file: str or path object
                 Path to csv file where the new label database is to be saved
                 with the changed
    """
    df = pd.read_csv(ALL_CONTRASTS, sep='\t')
    con_index = df[df['contrast'] == contrast].index
    for label in labels:
        if label in df.columns:
            df.at[con_index, label] = 1.0
        else:
            print("No label with the name {} could be found".format(label))
            df.at[con_index, label] = 1.0
            df.fillna(0.0, inplace=True)
            print("Added {}\n".format(label))

    df.to_csv(output_file, sep='\t', index=False)


def _flatten_contrast(contrast):
    """Helper function to change a labels_dict entry into a flattened list"""

    trans = str.maketrans("", "", "()")
    flat_contrast = contrast[0].translate(trans).split(" ")
    flat_contrast.extend(contrast[1])

    return flat_contrast


def sparse_labels(output_dir=os.path.dirname(ALL_CONTRASTS), save=True):
    """
    Transform the all_contrasts.csv file into a more readable, sparse file.
    The new file will contain the name of each task, each contrast and only
    the names of the labels that are related to them in each row.

    Parameters
    ----------
    output_dir: str, default ibc_data dir path
                Path for saving the new file. Defaults to the same directory
                where all_contrasts.csv is located

    save: bool, default True

    Returns
    -------
    sparse_df: pd.DataFrame
                   New dataframe with only the task name, contrast name and
                   names of labels in each row
    """

    labels_dict = get_labels()
    sparse_list = list(map(_flatten_contrast, labels_dict.items()))

    col_names = ['Task', 'Contrast']
    col_names.extend(["Label{}".format(i + 1 for i in range(10))])

    sparse_df = pd.DataFrame(sparse_list, columns=col_names)

    return sparse_df


