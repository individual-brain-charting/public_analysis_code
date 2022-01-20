"""
This script computes the mean activity in ROIS
for a given set of contrasts from the language network.

Authors: Ana Luisa Pinho, Bertrand Thirion

Last update: June 2020

Compatibility: Python 3.5

"""

import os
import glob
import re
import json

from joblib import Memory

import numpy as np

from math import *

import ibc_public
from ibc_public.utils_data import (data_parser, SMOOTH_DERIVATIVES,
                                   SUBJECTS, CONTRASTS, LABELS)

from nilearn.input_data import NiftiMasker, NiftiLabelsMasker
from nilearn.image import smooth_img, math_img, new_img_like
from nilearn import plotting

import matplotlib
matplotlib.use('Agg')  # to work in Drago
from matplotlib import pyplot as plt


def flatten(li):
    return sum(([x] if not isinstance(x, list) else flatten(x)
                for x in li), [])


def make_rois_img(rois_aggregate):
    rois_img = rois_aggregate[0]
    for i in range(1, len(rois_aggregate)):
        rois_img = math_img('i1 + %d * i2' % (i + 1),
                            i1=rois_img, i2=rois_aggregate[i])
    return rois_img


def adapt_roi(rois_group, subject, parser, masker, contrasts_set):
    func = parser[parser.subject == subject][parser.acquisition == 'ffx']\
           [parser.contrast.isin(contrasts_set)].path.values
    X = masker.transform(func)
    indiv_rois = []
    for roi in rois_group:
        print(roi)
        smooth_roi = smooth_img(math_img('img * 1.0', img=roi), fwhm=20.)
        y = masker.transform(roi).astype(np.int) * 1.0
        n_voxels = int(np.sum(y > 0))
        y_ = y.dot(np.linalg.pinv(X)).dot(X)
        # masker.inverse_transform(y_)\
        #     .to_filename('/tmp/%s' % os.path.basename(roi))
        z = np.ravel(y_ * masker.transform(smooth_roi))
        threshold = z[np.argsort(-z)[n_voxels // 2]]
        z = z > threshold
        indiv_roi = masker.inverse_transform(z)
        indiv_rois.append(indiv_roi)
    rois_img = make_rois_img(indiv_rois)
    return rois_img


def subject_specific_rois(rois_set):
    """
    Generate subject-specific ROIs
    """
    _package_directory = os.path.dirname(
        os.path.abspath(ibc_public.utils_data.__file__))
    mask_gm = os.path.join(
        _package_directory, '../ibc_data', 'gm_mask_1_5mm.nii.gz')
    masker = NiftiMasker(mask_img=mask_gm).fit()

    rois_images = []
    for subject in parser.subject.unique():
        rois_img = mem.cache(adapt_roi)(rois_set, subject, parser,
                                        masker, not_chosen_contrasts)
        rois_images.append(rois_img)
        # # Check ROIs for every subject
        # plotting.plot_roi(rois_img)
        # plotting.show()
    return rois_images


def roi_average(roi_paths, rois_parent_folder, tasks, df,
                selected_contrasts = None, subject_specific_rois = False,
                roi_masks = None):
    """
    Function to compute the average and std of z-scores for a set of voxels
    inside of subject-specific, functional Regions-of-Interest (ROIs)
    in a set of contrast z-maps.
    """
    # For every ROI, ...
    roi_names = []
    all_rois_contrast_avgs = []
    all_contrast_names = []
    for r, roi_path in enumerate(roi_paths, start=1):
        # ROI name
        roi_name = re.match('.*' + rois_parent_folder + '/(.*).nii.gz',
                            roi_path).groups()[0]
        roi_names.append(roi_name)
        # For every task, ...
        roi_contrast_avgs = []
        contrast_names = []
        for t, task in enumerate(tasks):
            # Select the entries in the data frame only concerned to
            # the ffx z-maps
            task_df = df[df.task == task][df.acquisition == 'ffx']
            contrasts = task_df.contrast.unique()
            if selected_contrasts is not None:
                contrasts = np.intersect1d(selected_contrasts, contrasts)
            # For every contrast, ...
            for contrast in contrasts:
                flatten_list = []
                for s, subject in enumerate(SUBJECTS):
                    img_paths = []
                    # Create the subject-specific ROI mask
                    if subject_specific_rois:
                        if roi_masks is None:
                            raise ValueError('roi_masks not defined!')
                        else:
                            roi = math_img('img == %d' %r, img=roi_masks[s])
                            print('Extracting subject-specific ' + \
                                  '"%s" as ROI for %s from the contrast "%s".'
                                  %(roi_name, subject, contrast))
                    else:
                        roi = roi_path
                        print('Extracting general ' + \
                              '"%s" as ROI for %s from the contrast "%s".'
                              %(roi_name, subject, contrast))
                    masker = NiftiLabelsMasker(labels_img=roi)
                    masker.fit()
                    # Paths of the contrast z-maps of every participants
                    img_path = task_df[task_df.contrast == contrast]\
                                      [task_df.subject == subject]\
                                      .path.values[-1]
                    img_paths.append(img_path)
                    print(img_paths)
                    # For each participant, extract data from z-map
                    # according to the subject-specific ROI mask and
                    # average straight off the values of
                    # the corresponding voxels
                    mask_data = masker.transform(img_paths)
                    flatten_list.extend(mask_data.tolist()[0])
                roi_contrast_avgs.append(flatten_list)
                # Labels of the contrasts
                contrast_names.append(LABELS[contrast][1].values[0] +
                                      ' vs. ' +
                                      LABELS[contrast][0].values[0])
        # Append for all ROIs
        all_rois_contrast_avgs.append(roi_contrast_avgs)
        all_contrast_names.append(contrast_names)
    return all_rois_contrast_avgs, all_contrast_names, roi_names


def plot_horizontal_barplots(rois_contrast_averages, c_labels, roi_tags,
                             color_map, write_dir):
    """
    Plot ROIs in horizontal barplots.
    Save each plot in a pdf file.
    """
    # Set figure
    for roi_avgs, labels, cbar, tag in zip(rois_contrast_averages, c_labels,
                                           color_map, roi_tags):
        plt.figure(figsize=(8, 4))
        # Convert list of lists with results into a numpy array
        roi_avgs = np.array(roi_avgs)
        # Define subplot of bar charts and its position in the fig
        # plt.axes([left, bottom, width, height])
        ax = plt.axes([.4, .175, len(labels)*.07, .7])
        # ax.bar(x, height, width=0.8,
        #        bottom=None, *, align='center',
        #        data=None, **kwargs)
        ax.barh(np.arange(len(labels)), roi_avgs.mean(1), xerr=roi_avgs.std(1),
                error_kw=dict(capsize=2, captick=3), color=cbar)
        # X-axis
        plt.xlim((-3, 5))
        plt.xlabel('z-scores')
        # Y-axis
        ax.set_yticks(np.arange(len(labels)))
        y_labels = [i for i in labels]
        ax.set_yticklabels(y_labels)
        # Plot titles
        plt.title(tag, fontsize=14, fontweight='bold')
        # Save figure
        plt.savefig(os.path.join(write_dir, 'roi_' + tag + '.pdf'), dpi=300)


def plot_barh_panel(rois_contrast_averages, c_labels, roi_tags,
                    color_map, pt_number, write_dir, fname):
    """
    Plot a subset of 3 ROIs in horizontal barplots,
    sharing the same y-axis labels.
    Save the resulting plot in a png file.
    """
    # Set figure
    fig = plt.figure(figsize=(18.5, 6.))
    for n, roi_avgs in enumerate(rois_contrast_averages):
        # Convert list of lists with results into a numpy array
        roi_avgs = np.array(roi_avgs)
        # Define subplot of bar charts and its position in the fig
        # plt.axes([left, bottom, width, height])
        ax = plt.axes([.41 + n*.2025, .175, len(c_labels[n])*.02175, .675])
        # ax.bar(x, height, width=0.8,
        #        bottom=None, *, align='center',
        #        data=None, **kwargs)
        # Compute 95% confidence interval
        ci_95 = []
        ci_95 = 1.96 * roi_avgs.std(1)/sqrt(pt_number)
        ax.barh(np.arange(len(c_labels[n])), roi_avgs.mean(1),
                xerr=ci_95, error_kw=dict(capsize=2, captick=3),
                color=color_map[n])
        # X-axis
        ax.xaxis.set_ticks(np.arange(-4., 10., 2.))
        plt.xticks(fontsize=26)
        # Y-axis
        ax.set_yticks(np.arange(len(c_labels[n])))
        if n == 0:
            y_labels = [y for y in c_labels[n]]
            ax.set_yticklabels(y_labels, fontsize=26)
        else:
            ax.set_yticklabels('')
        # Plot titles
        if full_names[roi_tags[n]][:22] == 'Inferior Frontal Gyrus':
            if roi_tags[n] in ['IFGorb', 'IFGtri']:
                plot_title_1 = full_names[roi_tags[n]][:22]
                plot_title_2 = full_names[roi_tags[n]][22:]
                plt.suptitle(plot_title_1, x=.6, fontsize=26,
                             fontweight='bold')
                plt.title(plot_title_2, fontsize=26, fontweight='bold',
                          style='italic')
            else:
                plot_title = full_names[roi_tags[n]][:9] + '\n' + \
                             full_names[roi_tags[n]][9:]
                plt.title(plot_title, fontsize=26, fontweight='bold')
        elif full_names[roi_tags[n]] in ['anterior Superior Temporal Sulcus',
                                         'posterior Superior Temporal Sulcus']:
            plot_title_1 = full_names[roi_tags[n]][:9]
            plot_title_2 = full_names[roi_tags[n]][9:]
            plt.title(plot_title_1, fontsize=26, fontweight='bold',
                      style='italic')
            if roi_tags[n] in ['aSTS', 'pSTS']:
                plt.suptitle(plot_title_2, x=.801, fontsize=26,
                             fontweight='bold')
            else:
                plt.suptitle(plot_title_2, x=.595, fontsize=26,
                             fontweight='bold')
        else:
            if full_names[roi_tags[n]] == 'Temporoparietal Junction':
                plot_title = full_names[roi_tags[n]][:15] + '\n' + \
                             full_names[roi_tags[n]][15:]
            elif full_names[roi_tags[n]] == 'Ventromedial Prefrontal Cortex':
                plot_title = full_names[roi_tags[n]][:13] + '\n' + \
                             full_names[roi_tags[n]][13:]
            else:
                plot_title = full_names[roi_tags[n]]
            plt.title(plot_title, fontsize=26, fontweight='bold')
    # Save fig
    fig.text(.5, .01, 'z-scores', ha='center', fontsize=26, fontweight='bold')
    plt.savefig(os.path.join(write_dir, fname), dpi=600)


def plot_barh_panel_2rois(rois_contrast_averages, c_labels, roi_tags,
                    color_map, pt_number, write_dir, fname):
    """
    Plot a subset of 2 ROIs in horizontal barplots,
    sharing the same y-axis labels.
    Save the resulting plot in a png file.
    """
    # Set figure
    fig = plt.figure(figsize=(18.5, 6.))
    for n, roi_avgs in enumerate(rois_contrast_averages):
        # Convert list of lists with results into a numpy array
        roi_avgs = np.array(roi_avgs)
        # Define subplot of bar charts and its position in the fig
        # plt.axes([left, bottom, width, height])
        ax = plt.axes([.41 + n*.3, .175, len(c_labels[n])*.02175, .675])
        # ax.bar(x, height, width=0.8,
        #        bottom=None, *, align='center',
        #        data=None, **kwargs)
        # Compute 95% confidence interval
        ci_95 = []
        ci_95 = 1.96 * roi_avgs.std(1)/sqrt(pt_number)
        ax.barh(np.arange(len(c_labels[n])), roi_avgs.mean(1),
                xerr=ci_95, error_kw=dict(capsize=2, captick=3),
                color=color_map[n])
        # X-axis
        ax.xaxis.set_ticks(np.arange(-4., 10., 2.))
        plt.xticks(fontsize=26)
        # Y-axis
        ax.set_yticks(np.arange(len(c_labels[n])))
        if n == 0:
            y_labels = [y for y in c_labels[n]]
            ax.set_yticklabels(y_labels, fontsize=26)
        else:
            ax.set_yticklabels('')
        # Plot titles
        if full_names[roi_tags[n]][:22] == 'Inferior Frontal Gyrus':
            plot_title_1 = full_names[roi_tags[n]][:22]
            plot_title_2 = full_names[roi_tags[n]][22:]
            plt.suptitle(plot_title_1, x=.8, fontsize=26,
                         fontweight='bold')
            plt.title(plot_title_2, fontsize=26, fontweight='bold',
                      style='italic')
        elif full_names[roi_tags[n]] in ['anterior Superior Temporal Sulcus',
                                         'posterior Superior Temporal Sulcus']:
            plot_title_1 = full_names[roi_tags[n]][:9]
            plot_title_2 = full_names[roi_tags[n]][9:]
            plt.title(plot_title_1, fontsize=26, fontweight='bold',
                      style='italic')
            if roi_tags[n] in ['aSTS', 'pSTS']:
                plt.suptitle(plot_title_2, x=.801, fontsize=26,
                             fontweight='bold')
            else:
                plt.suptitle(plot_title_2, x=.595, fontsize=26,
                             fontweight='bold')
        elif full_names[roi_tags[n]] in \
             ['anterior Superior Temporal Sulcus and Temporal Pole']:
            plot_title_1 = full_names[roi_tags[n]][:33]
            plot_title_2 = full_names[roi_tags[n]][33:]
            plt.suptitle(plot_title_1, x=.5, fontsize=26,
                         fontweight='bold')
            plt.title(plot_title_2, fontsize=26, fontweight='bold')
        else:
            if full_names[roi_tags[n]] == 'Temporoparietal Junction':
                plot_title = full_names[roi_tags[n]][:15] + '\n' + \
                             full_names[roi_tags[n]][15:]
            elif full_names[roi_tags[n]] == 'Ventromedial Prefrontal Cortex':
                plot_title = full_names[roi_tags[n]][:13] + '\n' + \
                             full_names[roi_tags[n]][13:]
            else:
                plot_title = full_names[roi_tags[n]]
            plt.title(plot_title, fontsize=26, fontweight='bold')
    # Save fig
    fig.text(.5, .01, 'z-scores', ha='center', fontsize=26, fontweight='bold')
    plt.savefig(os.path.join(write_dir, fname), dpi=600)


def plot_panel(rois_contrast_averages, c_labels, roi_tags, write_dir):
    """
    Plot ROIs in a panel of vertical barplots.
    Save it in one pdf file.
    """
    # Set figure
    fig = plt.figure(figsize=(17, 17))
    for r, (roi_avgs, labels) in enumerate(zip(rois_contrast_averages,
                                               c_labels)):
        # How to plot in the next column and row
        if r % 3 == 0:
            q = 0
            if r == 0:
                row = 0
            else:
                row += 1
        else:
            q += 1
        # Convert list of lists with results into a numpy array
        roi_avgs = np.array(roi_avgs)
        # Define subplot of bar charts and its position in the fig
        # plt.axes([left, bottom, width, height])
        x_ratio = .075 + len(labels)*.03
        y_ratio = .25
        ax = plt.axes([.075 + x_ratio*q, .05 + y_ratio*row,
                       len(labels)*.03, .16])
        # ax.bar(x, height, width=0.8,
        #        bottom=None, *, align='center',
        #        data=None, **kwargs)
        ax.bar(np.arange(len(labels)), roi_avgs.mean(1), yerr=roi_avgs.std(1),
               error_kw=dict(capsize=2, captick=3))
        # X-axis
        ax.set_xticks(np.arange(len(labels)))
        x_labels = ['c%s' % i for i in np.arange(len(labels))]
        ax.set_xticklabels(x_labels)
        # Y-axis
        plt.ylim((-3, 5))
        plt.ylabel('z-scores')
        # Plot titles
        plt.title(roi_tags[r], fontsize=16, fontweight='bold')
    # LEGEND
    fig.text(.5, .96, 'Legend:', size=14)
    for t, text in enumerate(x_labels):
        fig.text(.5, .93 - t * .0175, text, size=14)
        fig.text(.55, .93 - t * .0175, labels[t], size=14)
    # Build a rectangular frame around the legend
    # Axes of the figure
    figax = plt.axes((0., 0., 1., 1.))
    figax.axis('off')
    left, width = .48, .32
    bottom, height = .79, .1925
    right = left + width
    top = bottom + height
    p = plt.Rectangle((left, bottom), width, height, fill=False)
    figax.add_patch(p)
    # Save figure
    plt.savefig(os.path.join(write_dir, 'rois_panel.pdf'), dpi=300)


def glass_brain_rois(rois_set, color_list, output_dir, fname, view='z',
                     plot_title = None):
    """
    Plot a set of Regions-of-Interest (ROIs) in one single glass brain.
    Saves the resulting figure in a pre-specified directory.

    view = 'x': sagittal
    view = 'y': coronal
    view = 'z': axial
    view = 'l': sagittal left hemisphere only
    view = 'r': sagittal right hemisphere only
    """
    if plot_title is None:
        display = plotting.plot_glass_brain(None, display_mode=view,
                                            black_bg=False, alpha=1.)
    else:
        display = plotting.plot_glass_brain(None, display_mode=view,
                                            black_bg=False, alpha=1.,
                                            title=plot_title)
    for roi, color in zip(rois_set, color_list):
        display.add_overlay(roi, cmap=plotting.cm.alpha_cmap(color,
                                                             alpha_min=1.))
    if view in ['x', 'l', 'r']:
        view_tag = 'sagittal'
    elif view == 'y':
        view_tag == 'coronal'
    elif view == 'z':
        view_tag = 'axial'
    # Save figure
    display.savefig(os.path.join(output_dir,
                                 fname + '_' + view_tag + '.png'), dpi=600)


def panel_ss_rois(rois_ss_imgs):
    """
    Generate panel with glass-brain plots of subject-specific ROIs
    """
    fig = plt.figure(figsize=(12, 10))
    c = 1/4 * ((len(SUBJECTS) - 1) // 4)
    for sj, rois_img in enumerate(rois_ss_imgs):
        rs = rois_img.get_data()
        n_rois = np.unique(rs)
        roi_maps = []
        for label in n_rois[1:]:
            bin_subject_roi = (rs == label)
            bin_subject_roi = bin_subject_roi.astype(int)
            bin_subject_roi_map = new_img_like(rois_img, bin_subject_roi)
            roi_maps.append(bin_subject_roi_map)
        axes = plt.axes([.25 * np.mod(sj, 4), c - 1/4 * (sj // 4), .25, .25])
        display = plotting.plot_glass_brain(None, display_mode='l', axes=axes,
                                            black_bg=False, alpha=1.,
                                            title=SUBJECTS[sj])
        for roi, color in zip(roi_maps, colors):
            display.add_overlay(roi,
                                cmap=plotting.cm.alpha_cmap(color,
                                                            alpha_min=1.))

    fig.savefig(os.path.join(cache,'panel_rois_subject_specific.png'), dpi=600)


def store_averages(ids):
    with open(os.path.join(cache, 'rois' + ids + 'zmap_averages.json'),
              'w') as fw_avg:
        json.dump(rois_zmap_averages, fw_avg)

    with open(os.path.join(cache, 'rois' + ids + 'contrast_labels.json'),
              'w') as fw_label:
        json.dump(contrast_labels, fw_label)

    with open(os.path.join(cache, 'rois' + ids + 'terms.json'),
              'w') as fw_term:
        json.dump(roi_terms, fw_term)


def open_json_files(ido):
    with open(os.path.join(cache, 'rois' + ido + 'zmap_averages.json'),
              'r') as fr_avg:
        zmap_avgs = json.load(fr_avg)

    with open(os.path.join(cache, 'rois' + ido + 'contrast_labels.json'),
              'r') as fr_label:
        c_labs = json.load(fr_label)

    with open(os.path.join(cache, 'rois' + ido + 'terms.json'),
              'r') as fr_term:
        r_terms = json.load(fr_term)

    return zmap_avgs, c_labs, r_terms


# ######################### Caching ###########################################

main_parent_dir = '/neurospin/tmp/bthirion/'
alt_parent_dir = '/storage/store/work/agrilopi/'

# study_folder = 'ibc_roi_analysis_pallier'
study_folder = 'ibc_roi_analysis_hcplang900_II_z16'

if os.path.exists(main_parent_dir):
    cache = main_parent_dir + study_folder
else:
    cache = alt_parent_dir +  study_folder

mem = Memory(cachedir=cache, verbose=0)

# ########################### TASKS ###########################################

# Access to the ffx z-maps of all participant
task_list = ['ArchiEmotional', 'ArchiSocial', 'ArchiSpatial',
             'ArchiStandard', 'HcpEmotion', 'HcpGambling',
             'HcpLanguage', 'HcpMotor', 'HcpRelational',
             'HcpSocial', 'HcpWm',
             'RSVPLanguage']

# BIDS conversion of task names
# Load dictionary file
with open(os.path.join('bids_postprocessed.json'), 'r') as f:
    task_dic = json.load(f)

TASKS = [task_dic[tkey] for tkey in task_list]
TASKS = flatten(TASKS)

parser = data_parser(derivatives=SMOOTH_DERIVATIVES,
                     subject_list=SUBJECTS,
                     conditions=CONTRASTS,
                     task_list=TASKS)

lang_contrasts = ['sentence-word', 'sentence-jabberwocky', 'word-pseudo',
                  'word-consonant_string', 'pseudo-consonant_string',
                  'reading-listening', 'reading-checkerboard',
                  'computation-sentences']

not_chosen_contrasts = [c for c in
                        parser[parser.acquisition == 'ffx'].contrast.unique()
                        if c not in lang_contrasts]

# ################################# ROIs ######################################

# ### Select a subset of rois

# ## For Pallier's ROIs
rois_folder = 'rois_pallier'
rois_subset_idx = [0, 1, 4, 5, 6, 8]
rois_upper_subset_idx = [0, 1, 5]
rois_lower_subset_idx = [4, 6, 8]

# ### For HCPlang900 ROIs
# rois_folder = 'rois_hcplang900_z16'

# ## With FG and without vmPFC
# rois_subset_idx = np.arange(6)
# rois_upper_subset_idx = [1, 2, 3]
# rois_lower_subset_idx = [0, 4, 5]

# ## With vmPFC and without FG
# rois_subset_idx = [2, 3, 4, 6]
# rois_upper_subset_idx = [6, 2]
# rois_lower_subset_idx = [4, 3]
# ##
# rois_subset_idx = np.arange(4)
# rois_upper_subset_idx = [3, 0]
# rois_lower_subset_idx = [2, 1]

# Load the ROIs...
rois_path = os.path.abspath(rois_folder)
rois_list = glob.glob(os.path.join(rois_path, '*.nii.gz'))
rois_list.sort()
rois_subset = [rois_list[x] for x in rois_subset_idx]
rois_upper_subset = [rois_list[x] for x in rois_upper_subset_idx]
rois_lower_subset = [rois_list[x] for x in rois_lower_subset_idx]

# Estimate subject-specific ROIs
# CHANGE HERE!!!
rois_imgs = subject_specific_rois(rois_subset)

# ####################### COLORMAPS FOR PLOTTING ##############################

# Create color list from matplotlib colormap
# gist_rainbow_12_colors = [0,  23,  46,  69,  92, 137, 149, 162, 185, 215,
#                           229, 255]
cmap = plt.get_cmap('gist_rainbow')
colors = cmap([0, 46, 92, 149, 185, 229])
colors_upper = cmap([0, 46, 149])
colors_lower = cmap([92, 185, 229])
colors_4rois = cmap([27, 149, 165, 255])
colors_upper_2rois = cmap([255, 27])
colors_lower_2rois = cmap([165, 149])

# ######################## PLOT GLASS-BRAIN ROIs ##############################

# Generate glass brain with generic ROIs
# glass_brain_rois(rois_subset, colors_4rois, cache,
#                  'rois_hcp_glass_brain', view='l')

# Generate glass brain with subject-specific ROIs
panel_ss_rois(rois_imgs)

# ####################### EXTRACT ROI Z-AVERAGES ##############################

# # Compute roi average across participants for every contrast ...
# # ...and store them

# # CHANGE HERE!!!
# No dual regression
# rois_zmap_averages, contrast_labels, roi_terms = roi_average(
#     rois_lower_subset, rois_folder, task_list, parser,
#     selected_contrasts = lang_contrasts)

# CHANGE HERE!!!
# With dual regression
rois_zmap_averages, contrast_labels, roi_terms = roi_average(
    rois_upper_subset, rois_folder, task_list, parser,
    selected_contrasts = lang_contrasts, subject_specific_rois = True,
    roi_masks = rois_imgs)

# Set the identifier to store the files
# CHANGE HERE!!!
id_save = '_upper_subset_ss_'
# id_save = '_lower_subset_ss_'
# id_save = '_upper_subset_'
# id_save = '_lower_subset_'

# Save roi averages in json files
store_averages(id_save)

# ########################## PLOT Z-AVERAGES ##################################

# Set the identifier to open the files
# CHANGE HERE!!!
id_open = '_upper_subset_ss_'
# id_open = '_lower_subset_ss_'
# id_open = '_upper_subset_'
# id_open = '_lower_subset_'

# Load json files
rois_zmap_averages, contrast_labels, roi_terms = open_json_files(id_open)

# ### Complete name of the regions
# CHANGE HERE!!!
# ## Pallier ROIs
# full_names = {'IFGorb': 'Inferior Frontal Gyrus pars orbitalis',
#               'IFGtri': 'Inferior Frontal Gyrus pars triangularis',
#               'Precentral_Pallier_2011': 'Precentral Gyrus',
#               'Putamen': 'Putamen', 'TP': 'Temporal Pole',
#               'TPJ': 'Temporoparietal Junction',
#               'aSTS': 'anterior Superior Temporal Sulcus',
#               'dmPFC_Pallier_2011': 'dorsal medial Prefrontal Cortex',
#               'pSTS': 'posterior Superior Temporal Sulcus'}

# OR HERE!!!
# ## HCPlang900 ROIs
full_names = {'left_FG': 'Fusiform Gyrus',
              'left_FP': 'Frontal Pole',
              'left_IFG': 'Inferior Frontal Gyrus pars orbitalis and pars' + \
                          ' triangularis',
              'left_TPJ': 'Temporoparietal Junction',
              'left_aSTS_TP': 'anterior Superior Temporal Sulcus and' + \
                              ' Temporal Pole',
              'left_pSTS': 'posterior Superior Temporal Sulcus',
              'vmPFC': 'Ventromedial Prefrontal Cortex'}

# Re-order contrasts for plotting
new_order = [2, 1, 0, 3, 6, 7, 4, 5]
reord_contrast_labels = []
reord_rois_zmap_averages = []
for rl in np.arange(len(roi_terms)):
    contrast_order = [contrast_labels[rl][co] for co in new_order]
    zmap_avg_order = [rois_zmap_averages[rl][ro] for ro in new_order]
    reord_contrast_labels.append(contrast_order)
    reord_rois_zmap_averages.append(zmap_avg_order)

# #### Which type of plot? ####

# Set the identifier to do the plotting
# CHANGE HERE!!!
id_plot = '_upper_subset_ss'
# id_plot = '_lower_subset_ss'
# id_plot = '_upper_subset'
# id_plot= '_lower_subset'

# # Plot horizontal barplots of z-averages
# plot_horizontal_barplots(rois_zmap_averages, contrast_labels, roi_terms,
#                          colors, cache)

# Plot of z-averages grouped horizontal barplots
# CHANGE HERE!!!
# plot_barh_panel(reord_rois_zmap_averages, reord_contrast_labels, roi_terms,
#                 colors_upper, len(SUBJECTS), cache, 'rois' + id_plot + '.eps')
plot_barh_panel_2rois(reord_rois_zmap_averages, reord_contrast_labels,
                      roi_terms, colors_upper_2rois, len(SUBJECTS), cache,
                      'rois' + id_plot + '_4rois.eps')

# # Plot z-averages panel of vertical bar plots with all ROIs
# plot_panel(rois_zmap_averages, contrast_labels, roi_terms, cache)
