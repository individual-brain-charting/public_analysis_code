"""
Interactive display of volume FastSRM results in the browser

Authors: Ana Luisa Pinho

Created: April 2021
Last update: May 2021

Compatibility: Python 3.7
"""

import os
import numpy as np

import ibc_public.utils_data

from nilearn.input_data import NiftiMasker
from nilearn import plotting


# Mask of the grey matter of the IBC participants
_package_directory = os.path.dirname(
    os.path.abspath(ibc_public.utils_data.__file__))
mask_gm = os.path.join(_package_directory,
                       '../ibc_data', 'gm_mask_1_5mm.nii.gz')

# Masker
masker = NiftiMasker(mask_img=mask_gm).fit()

# Data paths
this_dir = os.path.dirname(os.path.abspath(__file__))
clips_path = os.path.join(this_dir, 'data_paper3_results', 'volume_encoding',
                          'pearson_correlations_clips.npy')
raiders_path = os.path.join(this_dir, 'data_paper3_results', 'volume_encoding',
                            'pearson_correlations_raiders.npy')

# Load the 2D-array map
clips_array = np.load(clips_path)
raiders_array = np.load(raiders_path)

# Generate niimg files
clips_map = masker.inverse_transform(clips_array)
raiders_map = masker.inverse_transform(raiders_array)

# Save NIfTI files
clips_outpath = os.path.join(this_dir, 'data_paper3_results',
                             'volume_encoding',
                             'pearson_correlations_clips_map.nii.gz')
raiders_outpath = os.path.join(this_dir, 'data_paper3_results',
                               'volume_encoding',
                               'pearson_correlations_raiders_map.nii.gz')
clips_map.to_filename(clips_outpath)
raiders_map.to_filename(raiders_outpath)

# Create visualization object
clips_view = plotting.view_img(clips_map)
raiders_view = plotting.view_img(raiders_map)

# Open the interactive panel in the browser
clips_view.open_in_browser()
raiders_view.open_in_browser()
