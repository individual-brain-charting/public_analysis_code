"""
Generate more snapshots of rbain activity for the sake of demonstration 

Author: Bertrand Thirion, 2017
"""
import glob
import os
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.input_data import NiftiMasker
from joblib import Memory, Parallel, delayed
from nilearn import plotting
from nilearn.image import math_img
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from nistats.thresholding import map_threshold
from ibc_public.utils_data import (
    data_parser, BETTER_NAMES, DERIVATIVES, SMOOTH_DERIVATIVES, SUBJECTS,
    CONDITIONS, CONTRASTS)
import matplotlib.pyplot as plt
                             
cache = '/neurospin/tmp/bthirion'
mem = Memory(cachedir=cache, verbose=0)
task_list = ['ArchiEmotional', 'ArchiSocial', 'ArchiSpatial',
             'ArchiStandard', 'HcpEmotion', 'HcpGambling',
             'HcpLanguage', 'HcpMotor', 'HcpRelational',
             'HcpSocial', 'HcpWm',
             'RSVPLanguage']


def break_string(input_string, length=40):
    """Insert '\n' in the input string after length characters"""
    import numpy as np
    print(len(input_string))
    parts = input_string.split(' ')
    size_parts = np.array([len(part) for part in parts])
    cum_size = np.cumsum(size_parts)
    if cum_size.max() <= length:
        return input_string
    idx = np.maximum(1, np.where(cum_size > length)[0].min())
    parts.insert(idx, '\n ')
    output_string = ''
    for part in parts:
        output_string += (part + ' ')
    return output_string[:-1]

def plot_contrasts(df, task_contrast, masker, write_dir, cut=0,
                   display_mode='x', name=''):
    """
    Parameters
    ----------
    df: pandas dataframe,
        holding information on the database indexed by task, contrast, subject
    task_contrasts: list of tuples,
               Pairs of (task, contrasts) to be displayed
    masker: nilearn.NiftiMasker instance,
            used to mask out images
    write_dir: string,
               where to write the result
    """
    from nilearn.plotting import cm
    fig = plt.figure(figsize=(16, 4), facecolor='k')
    plt.axis('off')
    n_maps = len(task_contrast)
    cmap = plt.get_cmap(plt.cm.gist_rainbow)
    color_list = cmap(np.linspace(0, 1, n_maps + 1))
    break_length = 165. / n_maps
    grid = 5 * np.ones((10, 10))
    grid[0] = 1
    for i in range(n_maps):
        delta = 1./ n_maps
        pos =[delta * i, 0.01, delta, .1]
        ax = fig.add_axes(pos, facecolor='k')
        ax.axis('off')
        inset = fig.add_axes([delta * i, 0.01, .01, .05])
        inset.imshow(grid, cmap=cm.alpha_cmap(color_list[i]))
        inset.axis('off')
        x_text = .08
        y_text = .95
        ax.text(x_text, y_text,
                break_string(BETTER_NAMES[task_contrast[i][1]], break_length),
                va='top', ha='left', fontsize=11, color='w',
                transform=ax.transAxes)

    for i, subject in enumerate(SUBJECTS):
        # anat = df[df.contrast == 't1_bet'][df.subject == subject].path.values[-1]
        anat = df[df.contrast == 'highres_gm'][df.subject == subject].path.values[-1]
        print(anat)
        axes = plt.axes([.01 + .167 * np.mod(i, 6) , .12 + .44 * (i / 6), .165, .44])
        th_imgs = []
        for task, contrast in task_contrast:
            imgs = df[df.task == task][df.contrast == contrast]\
                   [df.subject == subject][df.acquisition == 'ffx'].path.values
            if len(imgs > 0):
                img = imgs[-1]
                threshold = np.percentile(masker.transform(img), 99)
                th_img, _ = map_threshold(
                    img, threshold=threshold, height_control='height',
                    cluster_threshold=5)
                th_imgs.append(th_img)
        plotting.plot_prob_atlas(
            th_imgs, bg_img=anat, axes=axes,
            display_mode=display_mode,
            cut_coords=[cut], black_bg=True, annotate=False, dim=0, # title=subject,
            colorbar=False, view_type = 'filled_contours', linewidths=2.)
        axes.axis('off')
    fig.savefig(os.path.join(write_dir, 'snapshot_%s.pdf' % name),
                facecolor='k', dpi=300)
    # plt.close(fig)

db = data_parser(derivatives=SMOOTH_DERIVATIVES, conditions=CONTRASTS)
# db = db[db.task.isin(task_list)]
mask_gm = nib.load(os.path.join(DERIVATIVES, 'group', 'anat', 'gm_mask.nii.gz'))
masker = NiftiMasker(mask_img=mask_gm, memory=mem).fit()

write_dir = 'output'
if not os.path.exists(write_dir):
    os.mkdir(write_dir)
"""
task_contrast = [('ArchiSocial', 'false_belief-mechanistic_video'),
                 ('ArchiSocial', 'false_belief-mechanistic_audio'),
                 ('ArchiSocial', 'triangle_mental-random'),
                 ('hcp_social', 'mental-random')]
plot_contrasts(db, task_contrast, masker, write_dir, cut=-50, display_mode='x',
               name='social')

#plot_contrasts(db, task_contrast, masker, write_dir, cut=20, display_mode='z')
task_contrast = [('HcpWm', 'body-avg'),
                 ('HcpWm', 'face-avg'),
                 ('HcpWm', 'place-avg'),
                 ('HcpWm', 'tools-avg'),
                 ('HcpEmotion', 'shape'),
                 ('HcpEmotion', 'face-shape'),
                 ('RSVPLanguage', 'consonant_string')]
plot_contrasts(db, task_contrast, masker, write_dir, cut=-10, display_mode='z',
               name='visual')
task_contrast = [('HcpMotor', 'left_hand-avg'),
                 ('HcpMotor,', 'right_hand-avg'),
                 ('HcpMotor', 'left_foot-avg'),
                 ('HcpMotor', 'right_foot-avg'),
                 ('HcpMotor',	'tongue-avg')]
plot_contrasts(db, task_contrast, masker, write_dir, cut=-10, display_mode='y',
               name='motor')
task_contrast = [('RSVPLanguage', 'sentence-jabberwocky'),
                 ('RSVPLanguage', 'sentence-word'),
                 ('RSVPLanguage', 'word-consonant_string'),
                 ('RSVPLanguage', 'pseudo-consonant_string'),
                 ('ArchiSocial', 'mechanistic_video')]
plot_contrasts(db, task_contrast, masker, write_dir, cut=50, display_mode='x',
               name='standard')
"""
task_contrast = [('ArchiStandard', 'left-right_button_press'),
                 ('ArchiStandard', 'reading-listening'),
                 ('ArchiSocial', 'false_belief-mechanistic_audio'),
                 ('ArchiStandard', 'computation-sentences'),
                 ('ArchiStandard', 'horizontal-vertical')]
                 

plot_contrasts(db, task_contrast, masker, write_dir, cut=40, display_mode='x',
               name='standard')
plt.show()
