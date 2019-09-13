import json
import sys
import pandas as pd

import os

_package_directory = os.path.dirname(os.path.abspath(__file__))

ALL_CONTRASTS = os.path.join(_package_directory, '..', 'ibc_data',
                             'all_contrasts.tsv')
CONTRAST_COL = 'contrast'


def get_labels(contrasts, contrast_col=CONTRAST_COL):
    """Returns the list of labels for each passed contrast name"""
    df = pd.read_csv(ALL_CONTRASTS, sep='\t')

    for contrast in contrasts:
        con = df[df[contrast_col] == contrast]

        if len(con.index) == 0:
            err = f"There is no contrast with the name {contrast}"
            raise ValueError(err)
        else:
            labels = df.columns[con.isin([1.0]).any()]
            print(f"The labels for {contrast} are: "
                  f"{[label for label in labels]}")


def add_labels(contrast, labels, contrast_col=CONTRAST_COL,
               output_file=ALL_CONTRASTS):
    """Adds all the passed labels to the selected contrast"""
    df = pd.read_csv(ALL_CONTRASTS, sep='\t')
    con_index = df[df[contrast_col] == contrast].index
    for label in labels:
        if label in df.columns:
            df.at[con_index, label] = 1.0
        else:
            print(f"There is no label with the name {label}")

    df.to_csv(output_file, sep='\t', index=False)
