"""
Interactive display of surface FastSRM results and Glasser atlas in the browser

Authors: Ana Luisa Pinho

Created: May 2021

Compatibility: Python 3.7
"""

import os

import numpy as np

from nilearn.surface import load_surf_data
from nilearn.datasets import fetch_surf_fsaverage
from nilearn.plotting import plot_surf_stat_map, view_surf


# Some paths
this_dir = os.path.dirname(os.path.abspath(__file__))
atlas_path = os.path.join(this_dir, 'data_paper3_results', 'glasser_atlas')
results_path = os.path.join(
    this_dir, 'data_paper3_results', 'second_level_surface_encoding')

# Loads the high-resolution fsaverage mesh (163842 nodes)
fsaverage = fetch_surf_fsaverage(mesh='fsaverage')

# Load data
surf_lh = load_surf_data(os.path.join(atlas_path, 'lh.HCPMMP1.annot'))
surf_rh = load_surf_data(os.path.join(atlas_path, 'rh.HCPMMP1.annot'))
zvals_lh_array = np.load(os.path.join(results_path, 'zvals_lh.npy'))
zvals_rh_array = np.load(os.path.join(results_path, 'zvals_rh.npy'))
zvals_lh = load_surf_data(zvals_lh_array)
zvals_rh = load_surf_data(zvals_rh_array)

# Plot atlas
output_file_glh = os.path.join(atlas_path, 'fig_lh.png')
output_file_grh = os.path.join(atlas_path, 'fig_rh.png')
plot_surf_stat_map(fsaverage.infl_left, surf_lh, hemi='left', view='lateral',
                   colorbar=False, bg_map=fsaverage.sulc_left,
                   output_file='fig_lh.png')

plot_surf_stat_map(fsaverage.infl_right, surf_rh, hemi='right', view='lateral',
                   colorbar=False, bg_map=fsaverage.sulc_left,
                   output_file='fig_rh.png')

# Interactive display of atlas and results (w/o threshold) in the browser
glasser_lh = view_surf(fsaverage.infl_left, surf_lh)
glasser_rh = view_surf(fsaverage.infl_right, surf_rh)

map_lh = view_surf(fsaverage.infl_left, zvals_lh)
map_rh = view_surf(fsaverage.infl_right, zvals_rh)

glasser_lh.open_in_browser()
glasser_rh.open_in_browser()
map_lh.open_in_browser()
map_rh.open_in_browser()
