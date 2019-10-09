"""
This small module is used to go back and forth
"""
import os
import pandas as pd
import numpy as np
import ibc_public


def make_compact_table(cs, output_file=None):
    """ Create a compact list of labels per contrast

    Parameters
    ----------
    cs: Pandas Dataframe,
        describing a sparse table of contrasts
        needs the 'contrast', 'task', 'tags', 'pretty name', 'negative label',
                  'positive label' column names,
                   plus other interpreted as cognitive labels
    
    output_file: string,
                 tsv file path where this should be written to

    Returns
    -------
    output_df: Pandas Dataframe
               represents the sem structure.
               has keys ['task', 'contrast', 'pretty name',
                         'negative label', 'positive label', 'tags']

    """
    # define the tags to be associated with each contrast
    tags = []
    
    for index, con in cs.iterrows():
        tag = con.loc[con == 1.0].index
        tags.append(list(tag))

    # generate the output as a dictionary
    output = {'contrast': cs.contrast.values,
              'task':cs.task.values,
              'tags': tags,
              'pretty name': cs['pretty name'].values,
              'negative label': cs['negative label'].values,
              'positive label': cs['positive label'].values,
    }
    # convert it to a DataFrame
    columns = ['task', 'contrast', 'pretty name', 'negative label',
               'positive label', 'tags']
    output_df = pd.DataFrame(output, columns=columns)

    # possibly write it to a file
    if output_file is not None:
        output_df.to_csv(
            output_file, sep='\t', columns=columns, index=False)    

    return output_df


def _clean_tag(tag):
    """ """
    bad_chars = [']', '[', "'"]
    tag = tag.replace("'", "")
    tag = tag.replace('"', '')
    tag = tag.replace(',', '')
    for bc in bad_chars:
        tag = tag.strip(bc)
    tag_ = [] 
    for t in tag.split(' '):
        tag_.append(t)
    return tag_


def expand_table(compact_table, output_file=None, tag_columns=None):
    """ Converse operation of make_compact_table

    Parameters
    ----------
    cs: Pandas Dataframe,
        describing a sparse table of contrasts
        needs the 'contrast', 'task', 'tags', 'pretty name', 'negative label',
                  'positive label' 'tags' column names,
                   where tags yeilds a list of cognitive labels
    
    output_file: string,
                 tsv file path where this should be written to

    Returns
    -------
    output_df: Pandas Dataframe
               represents the sem structure.
               has keys ('task', 'contrast', 'pretty name',
                         'negative label', 'positive label'),
               plus one column per tag
    """
    # define the tags to be associated with each contrast
    if tag_columns is None:
        tag_list = list(compact_table.tags.values)
        all_tags = []
        for tag in tag_list:
            for t in _clean_tag(tag):
                if len(t):
                    all_tags.append(t)
        tag_columns = np.unique(all_tags)
        
    # create dictionary
    output = {'contrast': compact_table.contrast.values,
              'task':compact_table.task.values,
              'pretty name': compact_table['pretty name'].values,
              'negative label': compact_table['negative label'].values,
              'positive label': compact_table['positive label'].values,
    }
    # initialize values with -1
    for tc in tag_columns:
        output[tc] = -np.ones(shape=len(compact_table))

    # create output dataframe
    columns = ['task', 'contrast', 'pretty name', 'negative label',
               'positive label'] + list(tag_columns)
    output_df = pd.DataFrame(output, columns=columns)

    # fill with ones where it makes sense
    for index, con in compact_table.iterrows():
        tags = _clean_tag(con['tags'])
        for tag in tags:
            output_df.at[index, output_df.columns == tag] = 1

    # replace -1 with empty values
    output_df = output_df.replace(-1, '')
    # replace 1.0 with 1
    output_df = output_df.replace(1, '1')

    # if ajn output path is provided, write it there
    if output_file is not None:
        output_df.to_csv(
            output_file, sep='\t', columns=columns, index=False)    
        
    return output_df


# get the pass of the all_contrasts file
_package_directory = os.path.dirname(
    os.path.abspath(ibc_public.__file__))
all_contrasts = os.path.join(
    _package_directory, '../ibc_data', 'all_contrasts.tsv')

# get the corresponding directory
write_dir = os.path.dirname(all_contrasts)

# read the sparse table as a dataframe
sparse_contrasts = os.path.join(
    _package_directory, '../ibc_data', 'all_contrasts_sparse.tsv')
cs = pd.read_csv(all_contrasts, sep='\t')

# take the tag names (useful to preserve ordering)
ref_columns = cs.columns[5:]

rewrite = True
"""
# create a compact table
if rewrite:
    compact_table = make_compact_table(
        cs, os.path.join(write_dir, 'all_contrasts.tsv'))
else:
    compact_table = make_compact_table(cs)
"""
compact_contrasts = os.path.join(write_dir, 'all_contrasts.tsv')
compact_table = pd.read_csv(compact_contrasts, sep='\t')

# create a sparse table
if rewrite:
    sparse_table = expand_table(
        compact_table, os.path.join(write_dir, 'all_contrasts_sparse.tsv'))
else:
    sparse_table = expand_table(compact_table)
