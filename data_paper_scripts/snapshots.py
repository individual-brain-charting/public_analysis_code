"""
This script generate snapshots of brain activity for the different tasks across subjects
"""
import os
import matplotlib.image as mpimg
from nilearn import plotting
import matplotlib.pyplot as plt
import pandas as pd
from nistats.second_level_model import SecondLevelModel
import numpy as np


db = data_parser(derivatives=SMOOTH_DERIVATIVES)
mask_gm = nib.load(os.path.join(DERIVATIVES, 'group', 'anat', 'gm_mask.nii.gz'))
glm = SecondLevelModel(mask=mask_gm)
BETTER_NAMES =  ''



write_dir = ''
sorted_contrasts = ''

for task in sorted_contrasts.keys():
    task_dir = os.path.join(write_dir, task)
    if not os.path.exists(task_dir):
        os.mkdir(task_dir)
    contrasts = sorted_contrasts[task]
    n_contrasts = len(contrasts)
    # First do the random effects glass brain figure
    for i, contrast in enumerate(contrasts):
        contrast_mask = (db.contrast.values == contrast)
        dmtx = pd.DataFrame(np.ones(np.sum(contrast_mask)))
        glm.fit(list(db.path[contrast_mask].values), design_matrix=dmtx)
        grp_stat = glm.compute_contrast([1], stat_type='t', output_type='z_score')
        plotting.plot_glass_brain(
                grp_stat, display_mode='z', title=BETTER_NAMES[contrast],
                threshold=3., vmax=8, plot_abs=False, black_bg=True,
                output_file='/tmp/rfx_%s.png' % contrast)
        plt.figure(figsize=(7, 2 * n_contrasts + 1), facecolor='k', edgecolor='k')
        delta = (4 * n_contrasts - 1.) / (4 * n_contrasts ** 2)
        for i, contrast in enumerate(contrasts):
            ax = plt.axes([0., 1 - (i + 1) * delta, 1., delta], axisbg='k')
            ax.imshow(mpimg.imread('/tmp/rfx_%s.png' % contrast))
            plt.axis('off')        
        ax =  plt.axes([0.02, 0.0, .8, 1./ (8 * n_contrasts)],
                       axisbg='k')
        _draw_colorbar(ax, vmax=8, offset=3., orientation='horizontal', fontsize=14)        
        ax =  plt.axes([0.84, .01, .15, 1./ (8 * n_contrasts)], axisbg='k')
        ax.text(0, 0, 'z-scale', color='w', fontsize=14)
        ax.axis('off')
        plt.savefig(os.path.join(task_dir, 'glass_brain_rfx_colorbar_%s.pdf' % task),
                    facecolor='k', edgecolor='k', transparent=True, frameon=False,
                    pad_inches=0.)
        plt.close()
