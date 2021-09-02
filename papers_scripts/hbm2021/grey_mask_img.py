"""
Produce image with grey mask

author: Ana Luisa Pinho
date: July 2019
"""

import os
import numpy as np

import ibc_public.utils_data
from nilearn import plotting

import matplotlib

# Get grey matter mask
_package_directory = os.path.dirname(
    os.path.abspath(ibc_public.utils_data.__file__))
mask = os.path.join(
    _package_directory, '../ibc_data', 'gm_mask_1_5mm.nii.gz')

# Output path
output_path = '../../../admin/papers/descriptive_paper/'
output_name = 'eps_figs/grey_mask_1_5mm.eps'
output = os.path.join(output_path, output_name)

cmap = matplotlib.colors.ListedColormap('w', name='from_list', N=256)

plotting.plot_roi(mask, cmap=cmap, bg_img=None,
                  cut_coords=(-1, 24, 6),
                  black_bg = 'True',
                  output_file=output)
