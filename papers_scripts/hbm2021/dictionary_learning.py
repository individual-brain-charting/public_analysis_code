"""
Script that learns dictionary of functional loadings on the acquired data

Author: Bertrand Thirion

Last update: June 2020

Compatibility: Python 3.5

"""

import os
import json
import matplotlib.pyplot as plt
from nilearn.input_data import NiftiMasker
from nilearn import plotting
import nibabel as nib
import numpy as np
from joblib import Memory
# from utils_group_analysis import sorted_contrasts
from ibc_public.utils_data import (
    data_parser, SMOOTH_DERIVATIVES, DERIVATIVES, SUBJECTS, CONTRASTS,
    make_surf_db, all_contrasts)
import ibc_public
from utils_dictionary import make_dictionary, dictionary2labels, _make_labels


def flatten(li):
    return sum(([x] if not isinstance(x, list) else flatten(x)
                for x in li), [])

if 0:
    cache = '/neurospin/tmp/bthirion'
else:
    cache = '/storage/workspace/bthirion/'

write_dir = '/storage/store/work/agrilopi/individual_topographies'
if not os.path.exists(write_dir):
    os.mkdir(write_dir)

mem = Memory(cachedir=write_dir, verbose=0)
subject_list = SUBJECTS
n_subjects = len(subject_list)

# Access to the data
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

do_surface = True

if do_surface:
    db = make_surf_db(derivatives=DERIVATIVES, conditions=CONTRASTS,
                      subject_list=subject_list, task_list=TASKS)
    # db = db[db.subject != 'sub-15']
else:
    # Mask of the grey matter across subjects
    db = data_parser(derivatives=SMOOTH_DERIVATIVES, subject_list=subject_list,
                     conditions=CONTRASTS, task_list=task_list)
    _package_directory = os.path.dirname(
        os.path.abspath(ibc_public.utils_data.__file__))
    mask_gm = os.path.join(
        _package_directory, '../ibc_data', 'gm_mask_1_5mm.nii.gz')
    n_voxels = nib.load(mask_gm).get_data().sum()

    # reference image for display
    mean_anat = os.path.join(DERIVATIVES, 'group', 'anat', 'T1avg.nii.gz')

# Gather a set of images organized along subjects and contrasts dimensions
# Extract a subset of images from the database
df = db[db.task.isin(task_list)]
df = df.sort_values(by=['subject', 'task', 'contrast'])

contrasts = df.contrast.unique()
n_contrasts = len(contrasts)


def make_surf_data(df, subject_list):
    paths = []
    subject_list = df.subject.unique()
    for contrast in contrasts:
        for subject in subject_list:
            mask = (df.contrast == contrast).values *\
                   (df.subject == subject).values
            if len(df[mask]) == 0:
                print(subject, contrast)
            paths.append(df[mask][df.side == 'lh'].path.values[-1])
            paths.append(df[mask][df.side == 'rh'].path.values[-1])
    Xr = np.array([nib.load(texture).darrays[0].data
                  for texture in list(paths)])
    n_voxels = Xr.shape[1]
    Xr = Xr.reshape(n_contrasts, int(2 * n_subjects * n_voxels))
    return Xr, n_voxels


if do_surface:
    Xr, n_voxels = make_surf_data(df, subject_list)
    # mem.cache(make_surf_data)(df, subject_list)
else:
    paths = []
    for contrast in contrasts:
        for subject in subject_list:
            mask = (df.contrast == contrast).values *\
                   (df.subject == subject).values
            if len(df[mask]) == 0:
                print(subject, contrast)
            paths.append(df[mask].path.values[-1])
    # image masking
    masker = NiftiMasker(
        mask_img=mask_gm, memory=write_dir, smoothing_fwhm=None).fit()
    Xr = masker.transform(paths).reshape(
        n_contrasts, int(n_subjects * n_voxels))

# learn a dictionary of elements
n_components = 20
alpha = 4.0  # for l1
# alpha = .3  # for multi-task MultiTaskLasso
# alpha = 1.  # for multui-task enet
Xr[np.isnan(Xr)] = 0
dictionary, components_ = make_dictionary(
    Xr, n_components=20, alpha=alpha, write_dir=write_dir, contrasts=contrasts,
    method='sparse', l1_ratio=.25)
plt.show(block=False)

#  dictionary, components_ = cluster(Xr, n_components=20)
if do_surface:
    from utils_surface_plots import clean_surface_map
    components = np.reshape(components_,
                            (n_subjects, 2 * n_voxels, n_components))
    for i in range(n_subjects):
        components[i][:n_voxels] = clean_surface_map(
            components[i][:n_voxels], 'left', 300)
        components[i][n_voxels:] = clean_surface_map(
            components[i][n_voxels:], 'right', 300)
    mean_components = np.median(components, 0)
else:
    components = np.reshape(components_,
                            (n_subjects, n_voxels, n_components))
    mean_components = np.median(components, 0)


def map_surface_label(components, name, output_dir=write_dir, facecolor='k'):
    from utils_surface_plots import make_atlas_surface
    """
    labels = np.zeros(components.shape[0]).astype(np.int)
    mask = components.max(1) > 0
    labels[mask] = np.argmax(components, 1)[mask] + 1
    make_atlas_surface(labels[:n_voxels], 'left', name, output_dir)
    make_atlas_surface(labels[n_voxels:], 'right', name, output_dir)
    """
    mask = components.max(1) > 0
    labels = np.zeros(components.shape[0]).astype(np.int)
    labels[mask] = np.argmax(components, 1)[mask] + 1
    make_atlas_surface(
        labels[:n_voxels], 'left', name, output_dir)
    make_atlas_surface(
        labels[n_voxels:], 'right', name, output_dir)


if do_surface:
    # path = os.path.join(write_dir, 'labels.png')
    # dictionary2labels(dictionary, task_list, path, facecolor=[.5, .5, .5],
    #                   contrasts=contrasts)
    map_surface_label(mean_components, name='surf_mean')
    for i in range(n_subjects):
        map_surface_label(components[i], name='surf_%s' % subject_list[i])
    # make a figure out of labels + text
    for side in ['left', 'right']:
        plt.figure(figsize=(9, 7), facecolor='k', edgecolor='k')
        for i in range(n_subjects):
            ax = plt.axes([.25 * (i // 4), .25 * np.mod(i, 4), .25, .25])
            path = os.path.join(
                write_dir, 'surf_%s_lateral_%s.png' % (subject_list[i], side))
            img = plt.imread(path)
            img[img.sum(2) == 4, :3] = 0
            ax.imshow(img[60:-60, 90:-90])
            plt.axis('off')

        # include the group image
        ax = plt.axes([.75, .75, .25, .25])
        path = os.path.join(write_dir, 'surf_mean_lateral_%s.png' % side)
        img = plt.imread(path)
        img[img.sum(2) == 4, :3] = 0.2
        ax.imshow(img[60:-60, 90:-90])
        plt.axis('off')
        plt.plot([0, 0], [0, 360], linewidth=2, color='r')
        plt.plot([460, 460], [0, 360], linewidth=2, color='r')
        plt.plot([0, 460], [360, 360], linewidth=2, color='r')
        plt.plot([0, 460], [0, 0], linewidth=2, color='r')

        # include the labels
        """
        ax = plt.axes([.75, .25, .25, .5])
        path = os.path.join(write_dir, 'labels.png')
        img = plt.imread(path)
        ax.imshow(img)
        plt.axis('off')
        """
        plt.subplots_adjust(top=.99, right=.99, left=.00, bottom=.00,
                            hspace=.00, wspace=.00)
        fig_name = os.path.join(write_dir, 'surf_group_lateral_%s.pdf' % side)
        plt.savefig(fig_name, facecolor='k', edgecolor='k',
                    transparent=True, frameon=False, pad_inches=0., dpi=300)


LABELS = _make_labels(all_contrasts, task_list)
labels_bottom = [LABELS[name][0] for name in contrasts]
labels_top = [LABELS[name][1] for name in contrasts]


def plot_dictionary(components, idx, write_dir):
    """ Plot a dictionary element acorss subjects
    """
    mean_val = components[:, :, idx].mean(0)
    mean_img = masker.inverse_transform(mean_val)
    fig = plt.figure(figsize=(14, 5), facecolor='k')
    # colorbar_ax = fig.add_axes([.98, 0.05, .02, .9], axis_bgcolor='k')
    # vmax = np.max(np.abs(mean_val))
    # _draw_colorbar(colorbar_ax, vmax=vmax, offset = vmax/2)
    cut_coords = plotting.find_xyz_cut_coords(mean_img)
    for i, subject in enumerate(subject_list):
        anat = db[db.contrast == 't1'][db.subject == subject].path.values[-1]
        img = masker.inverse_transform(components[i, :, idx])
        axes = plt.axes([.00 + .13 * np.mod(i, 3), .245 * (i / 3), .13, .245])
        plotting.plot_stat_map(img, bg_img=anat, axes=axes, display_mode='x',
                               cut_coords=cut_coords[0:1], dim=0,
                               threshold=3.0,
                               black_bg=True, vmax=8,  # title=subject,
                               colorbar=False)

    axes = plt.axes([.4, .3, .57, .4])
    colors = plt.cm.hsv(np.linspace(0, 255 / n_contrasts, 255))
    plt.bar(range(n_contrasts), dictionary[idx], color=colors)
    plt.xticks(np.linspace(1., n_contrasts + .8, num=n_contrasts + 1),
               labels_bottom, rotation=75, ha='right', fontsize=9, color='w')
    for nc in range(n_contrasts):
        plt.text(nc, dictionary[idx].max() + .001, labels_top[nc], rotation=75,
                 ha='left', va='bottom', color='w', fontsize=9)
    #
    plt.axis('tight')
    plt.subplots_adjust(bottom=.3, top=.7)
    fig.savefig(os.path.join(write_dir, 'snapshot_%02d.png' % idx),
                facecolor='k')
    plt.close(fig)


if not do_surface:
    for i in range(n_components):
        fig = plt.figure(figsize=(11, 5), facecolor='k')
        axes = plt.axes([.01, .01, .5, .48])
        cmp_ = masker.inverse_transform(mean_components[:, i])
        plotting.plot_stat_map(
            cmp_, axes=axes, colorbar=False, display_mode='x', cut_coords=3,
            black_bg=True, bg_img=mean_anat, dim=0)
        axes = plt.axes([.01, .51, .5, .48])
        plotting.plot_stat_map(
            cmp_, axes=axes, colorbar=False, display_mode='z', cut_coords=3,
            black_bg=True, bg_img=mean_anat, dim=0)
        axes = plt.axes([.53, .3, .44, .42])
        colors = plt.cm.hsv(np.linspace(0, 255 / n_contrasts, 255))
        plt.bar(range(n_contrasts), dictionary[i], color=colors)
        plt.xticks(np.linspace(1., n_contrasts + .8, num=n_contrasts + 1),
                   labels_bottom, rotation=75, ha='right', fontsize=10,
                   color='w')
        for nc in range(n_contrasts):
            plt.text(nc, dictionary[i].max() + .001, labels_top[nc],
                     rotation=75, ha='left', va='bottom', color='w',
                     fontsize=10)
        #
        plt.axis('tight')
        plt.subplots_adjust(bottom=.3, top=.7)
        fig.savefig(os.path.join(write_dir, 'mean_component_%02d.png' % i),
                    facecolor='k')
        plt.close()

    for i in range(n_subjects):
        plotting.plot_stat_map(
            masker.inverse_transform(components[i, :, 0]), colorbar=False,
            output_file=os.path.join(write_dir, '_component_%02d.png' % i),
            display_mode='x', cut_coords=5, threshold=.00)

    for i, subject in enumerate(subject_list):
        bg_img = db[db.contrast == 't1'][db.subject == subject].path.values[-1]
        plotting.plot_prob_atlas(
            masker.inverse_transform(components[i, :, :].T),
            view_type='filled_contours',
            output_file=os.path.join(write_dir, 'dictionary_%s.png' % subject),
            linewidths=.5, dim=0, display_mode='x',
            bg_img=bg_img,
            cut_coords=[-50], black_bg=True)

    for idx in range(n_components):
        plot_dictionary(components, idx, write_dir)

    for i in range(n_components):
        cmp_ = masker.inverse_transform(mean_components[:, i])
        nib.save(cmp_, os.path.join(
            write_dir, 'mean_component_%02d.nii.gz' % i))
"""
for i in [0, 1, 8]:
    for j in range(n_subjects):
        cmp_ = masker.inverse_transform(components[j, :, i])
        nib.save(cmp_, os.path.join(
            write_dir, 'component_%02d_sub-%02d.nii.gz' % (i, j)))
"""
