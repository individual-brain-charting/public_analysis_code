"""
This script fetches the HCP Language elementary contrasts from
the NeuroVault collection id=4337, generate the group-level main contrast and
computes ROIs from it.

Author: Ana Luisa Pinho

Last update: June 2020

Compatibility: Python 3.5

"""


import os
import glob

import numpy as np
import pandas as pd

from nilearn.datasets import fetch_neurovault, neurovault
from nistats.reporting import plot_design_matrix
from nilearn import plotting
from nistats.second_level_model import SecondLevelModel
from nistats.reporting import make_glm_report
from nilearn.image import smooth_img, mean_img, load_img, new_img_like

from scipy import ndimage


def fetch_nv_collection(contrast):
    nv_data = fetch_neurovault(
    max_images=788,
    collection_id=4337,
    mode='overwrite',
    data_dir='/storage/store/data/HCP900/hcplang',
    cognitive_paradigm_cogatlas=neurovault.Contains(
        'language processing fMRI task paradigm'),
    contrast_definition=neurovault.Contains(contrast),
    map_type='Z map',
    task='LANGUAGE')

    print([meta['id'] for meta in nv_data['images_meta']])


def compute_group_z_map(second_level_input, n_sub, output_pathway):
    # Model the effect of conditions (sample 1 vs sample 2).
    condition_effect = np.hstack(([1] * n_sub, [- 1] * n_sub))

    # Model the subject effect:
    # each subject is observed in sample 1 and sample 2.
    subject_effect = np.vstack((np.eye(n_sub), np.eye(n_sub)))
    subjects = ['S%02d' % i for i in range(1, n_sub + 1)]

    # We then assemble those in a design matrix and...
    design_matrix = pd.DataFrame(
        np.hstack((condition_effect[:, np.newaxis], subject_effect)),
        columns=['Story vs. Math'] + subjects)

    # ... plot the design_matrix.
    plot_design_matrix(design_matrix, output_file=
                       os.path.join(output_pathway,
                                    'design_matrix_story_math.png'))

    # Specify the analysis model and fit it
    second_level_model = SecondLevelModel().fit(second_level_input,
                                                design_matrix=design_matrix)

    # Estimate the contrast
    z_map = second_level_model.compute_contrast('Story vs. Math',
                                                output_type='z_score')

    # Report of the GLM
    report = make_glm_report(second_level_model,
                             contrasts='Story vs. Math',
                             title='Group-Level HCP900 Story vs.Math Report',
                             cluster_threshold=5,
                             height_control='fdr',
                             min_distance=8.,
                             plot_type='glass',
    )

    report.save_as_html(os.path.join(output_pathway, 'report.html'))

    # Save contrast nifti-file
    z_map.to_filename(os.path.join(output_pathway,
                                   'group_hcplang900_story_math.nii.gz'))

    # Plot contrast
    threshold = 3.1  # correponds to  p < .001, uncorrected
    display = plotting.plot_glass_brain(z_map, threshold=threshold,
                                        colorbar=True,
                                        plot_abs=False,
                                        title='Story vs. Math (unc p<0.001)',
                                        output_file=os.path.join(
                                        output_pathway,
                                        'group_hcplang900_story_math'))

    return z_map


def compute_mean_epi(fmri_filenames, output_pathway):
    fmri_img = smooth_img(fmri_filenames, fwhm=5)
    mean_epi = mean_img(fmri_img)

    # Save
    mean_epi.to_filename(os.path.join(output_pathway, 'mean_epi.nii.gz'))

    # Plot
    plotting.plot_epi(mean_epi, title='Smoothed mean EPI',
                      output_file=os.path.join(output_pathway, 'mean_epi.png'))

    return mean_epi


# ############## Fetch data from NeuroVault collection id=4337 ################

# contrast_list = ['STORY', 'MATH']
# for c in contrast_list:
#     fetch_nv_collection(c)

# ########################### Inputs ##########################################

# List images
data_parent_dir = '/storage/store/data/HCP900/hcplang/neurovault'
story_path = os.path.join(data_parent_dir, 'collection_4337_story')
math_path = os.path.join(data_parent_dir, 'collection_4337_math')

story_contrasts = glob.glob(os.path.join(story_path, "*.nii.gz"))
math_contrasts = glob.glob(os.path.join(math_path, "*.nii.gz"))

# Define some extra-inputs
all_contrasts = story_contrasts + math_contrasts
n_subjects = len(story_contrasts)

# Threshold to compute the roi masks
threshold = 16

# ########################### Outputs #########################################

output_path = '/storage/store/work/agrilopi/hcplang900/'
rois_dir = os.path.join(output_path, 'rois_zbigger_%s' % threshold)

if not os.path.exists(output_path):
    os.mkdir(output_path)

if not os.path.exists(rois_dir):
    os.mkdir(rois_dir)

# ################### Compute group-level z-map ###############################

# z_map = compute_group_z_map(all_contrasts, n_subjects, output_path)


# ####################### Compute the mean EPI ################################

# mean_epi = compute_mean_epi(all_contrasts, output_path)


# ######################## Compute ROI mask ###################################

# ########## Load images ##########

# Comment the following lines if we are also running the previous section
z_map = load_img(os.path.join(output_path,
                              'group_hcplang900_story_math.nii.gz'))

mean_epi = load_img(os.path.join(output_path, 'mean_epi.nii.gz'))

# #################################

# Thresholding
thresholded_z_values = z_map.get_data()
thresholded_z_values[thresholded_z_values < threshold] = 0

# Binarization
bin_z_values = (thresholded_z_values != 0)

# Dilation
dil_bin_z_values = ndimage.binary_dilation(bin_z_values)
dil_bin_z_values = dil_bin_z_values.astype(int)
dil_bin_z_map = new_img_like(mean_epi, dil_bin_z_values)
plotting. plot_roi(dil_bin_z_map, mean_epi, title='Dilated mask',
                   output_file=os.path.join(output_path, 'dilated_mask.png'))

# Creating separate masks for every ROIs
labels, n_labels = ndimage.label(dil_bin_z_values)
print(n_labels)

for label_id in np.arange(n_labels + 1)[1:]:
    print(label_id)
    roi_data = np.array([])
    roi_data = (labels == label_id).astype(np.int)
    roi_img = new_img_like(dil_bin_z_map, roi_data)
    roi_img.to_filename(os.path.join(rois_dir,
                                     'roi_%s' % label_id + '.nii.gz'))
    plotting. plot_roi(roi_img, mean_epi, title='ROI %s' % label_id,
                   output_file=os.path.join(rois_dir,
                                            'roi_%s' % label_id + '.png'))
