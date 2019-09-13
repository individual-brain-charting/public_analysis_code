import json
import sys
import pandas as pd

import os

_package_directory = os.path.dirname(os.path.abspath(__file__))

ALL_CONTRASTS = os.path.join(_package_directory, '..', 'ibc_data',
                             'all_contrasts.tsv')
CONTRAST_COL = 'contrast'


def get_labels(contrasts, contrast_col=CONTRAST_COL):
    """
    Returns the list of labels for each passed contrast name

    Parameters
    ----------

    contrasts: list of str
               each element of the list is a contrast name in the document

    contrast_col: str
                  string used to locate the column of the labels file that
                  stores contrast names

    Returns
    -------

    contrast_dict: dict
                   dictionary containing the contrasts provided by the user
                   as keys, and their corresponding labels as values
    """
    df = pd.read_csv(ALL_CONTRASTS, sep='\t')

    contrast_dict = {}

    for contrast in contrasts:
        con = df[df[contrast_col] == contrast]

        if len(con.index) == 0:
            err = f"There is no contrast with the name {contrast}"
            raise ValueError(err)
        else:
            labels = df.columns[con.isin([1.0]).any()]
            contrast_dict[contrast] = [label for label in labels]

    return contrast_dict


def add_labels(contrast, labels, contrast_col=CONTRAST_COL,
               output_file=ALL_CONTRASTS):
    """
    Adds all the passed labels to the selected contrast

    Paramenters
    -----------

    contrast: str
              name of the contrast that will get the labels

    labels: list of str
            labels that the user wants to add. The labels must exist as
            columns in the file

    contrast_col: str
              string used to locate the column of the labels file that
              stores contrast names

    output_file: str or path object
                 path to csv file where the new label database is to be saved
                 with the changed
    """
    df = pd.read_csv(ALL_CONTRASTS, sep='\t')
    con_index = df[df[contrast_col] == contrast].index
    for label in labels:
        if label in df.columns:
            df.at[con_index, label] = 1.0
        else:
            print(f"There is no label with the name {label}")

    df.to_csv(output_file, sep='\t', index=False)
