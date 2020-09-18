"""
This script generates figures that assess low-level data quality
for the second release of the IBC dataset:
* amount of motion
* tSNR of the data

Authors: Bertrand Thirion, Ana Luisa Pinho 2020

Compatibility: Python 3.5

"""
import os
import glob
import json
import warnings

from joblib import Parallel, delayed

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nilearn.input_data import NiftiMasker
from nilearn import plotting

# ############################### INPUTS ######################################

SOURCEDATA = '/neurospin/ibc/sourcedata'
DERIVATIVES = '/neurospin/ibc/derivatives'

# Extract data from dataframe only referring to a pre-specified subset of
# participants
sub_no = [1, 4, 5, 6, 7, 9, 11, 12, 13, 14, 15]

# ### All tasks ###
# Tasks terminology must conform with the one used in all_contrasts.tsv and
# main_contrasts.tsv
task_list = ['mtt_we', 'mtt_sn',
             'preference',
             'theory_of_mind', 'emotional_pain', 'pain_movie',
             'vstm', 'enumeration',
             'self', 'bang']
sufix = 'rel2'

# ### Single task ###
# task_list = ['mtt_we']
# sufix = '%s' % task_list[0]


cache = '/neurospin/tmp/agrilopi/tsnr_2'
mem = '/neurospin/tmp/agrilopi'

# #############################################################################

# Define subjects' paths
sub_path = [os.path.join(DERIVATIVES, 'sub-%02d' % s) for s in sub_no]
SUBJECTS = [os.path.basename(full_path) for full_path in sub_path]

# BIDS conversion of task names
# Load dictionary file
with open(os.path.join('bids_preprocessed.json'), 'r') as f:
    task_dic = json.load(f)

TASKS = [task_dic[tkey] for tkey in task_list]
TASKS = [item for sublist in TASKS for item in sublist]


def data_parser(derivatives=DERIVATIVES):
    """Generate a dataframe that contains all the data corresponding
    to the acquisitions of the 2nd release."""
    paths = []
    subjects = []
    sessions = []
    modalities = []
    contrasts = []
    tasks = []
    acquisitions = []

    for sbj in SUBJECTS:
        # T1 images
        t1_relat_path = 'sub-*/ses-00/anat/w%s_ses-00_T1w_nonan.nii.gz' % sbj
        t1_abs_path = os.path.join(derivatives, t1_relat_path)
        t1_imgs_ = glob.glob(os.path.join(t1_abs_path))
        if not t1_imgs_:
            msg = 'w%s_T1w_nonan.nii.gz file not found!' % (sbj)
            warnings.warn(msg)
        for img in t1_imgs_:
            session = img.split('/')[-3]
            subject = img.split('/')[-4]
            paths.append(img)
            sessions.append(session)
            subjects.append(subject)
            modalities.append('T1')
            contrasts.append('t1')
            tasks.append('')
            acquisitions.append('')

    for sbj in SUBJECTS:
        # gray-matter images
        c1_relat_path = 'sub-*/ses-00/anat/mwc1%s_ses-00_T1w.nii.gz' % sbj
        c1_abs_path = os.path.join(derivatives, c1_relat_path)
        c1_imgs_ = glob.glob(os.path.join(c1_abs_path))
        if not c1_imgs_:
            msg = 'mwc1%s_T1w.nii.gz file not found!' % (sbj)
            warnings.warn(msg)
        for img in c1_imgs_:
            session = img.split('/')[-3]
            subject = img.split('/')[-4]
            paths.append(img)
            sessions.append(session)
            subjects.append(subject)
            modalities.append('T1')
            contrasts.append('gm')
            tasks.append('')
            acquisitions.append('')

    for sbj in SUBJECTS:
        # white-matter images
        c2_relat_path = 'sub-*/ses-00/anat/mwc2%s_ses-00_T1w.nii.gz' % sbj
        c2_abs_path = os.path.join(derivatives, c2_relat_path)
        c2_imgs_ = glob.glob(os.path.join(c2_abs_path))
        if not c2_imgs_:
            msg = 'mwc2%s_T1w.nii.gz file not found!' % (sbj)
            warnings.warn(msg)
        for img in c2_imgs_:
            session  = img.split('/')[-3]
            subject = img.split('/')[-4]
            paths.append(img)
            sessions.append(session)
            subjects.append(subject)
            modalities.append('T1')
            contrasts.append('wm')
            tasks.append('')
            acquisitions.append('')

    for sbj in SUBJECTS:
        # preprocessed images
        for acq in ['ap', 'pa']:
            for task in TASKS:
                bold_name = 'wrdc%s_ses*_task-%s_dir-%s*_bold.nii.gz' \
                            % (sbj, task, acq)
                bold_path = os.path.join(derivatives, 'sub-*/ses-*/func',
                                         bold_name)
                bold = glob.glob(bold_path)
                if not bold:
                    # Add exception for 'bang' task, since 'ap' was
                    # never part of acq planning
                    if acq == 'ap' and task == 'Bang':
                        pass
                    else:
                        msg = 'wrdc*.nii.gz file for task ' + \
                              '%s %s in %s not found!' % (task, acq, sbj)
                        warnings.warn(msg)

                for img in bold:
                    basename = os.path.basename(img)
                    parts = basename.split('_')
                    task_ = None
                    for part in parts:
                        if part[4:7] == 'sub':
                            subject = part[4:10]
                        elif part[:3] == 'ses':
                            session = part
                        elif part[:5] == 'task-':
                            task_ = part[5:]
                        elif part[:4] == 'dir-':
                            acquisition = part[4:]
                    if task not in TASKS:
                        continue
                    paths.append(img)
                    sessions.append(session)
                    subjects.append(subject)
                    modalities.append('bold')
                    contrasts.append('preprocessed')
                    tasks.append(task_)
                    acquisitions.append(acquisition)

                rps_name = 'rp_dc%s_ses*_task-%s_dir-%s*_bold.txt' \
                           % (sbj, task, acq)
                rps_path = os.path.join(derivatives, 'sub-*/ses-*/func',
                                        rps_name)
                rps = glob.glob(rps_path)
                if not rps:
                    # Add exception for 'bang' task, since 'ap' was
                    # never part of acq planning
                    if acq == 'ap' and task == 'Bang':
                        pass
                    else:
                        msg = 'rp-dc*.txt file for task' + \
                              '%s %s in %s not found!' % (task, acq, sbj)
                        warnings.warn(msg)

                for rp_file in rps:
                    basename = os.path.basename(rp_file)
                    parts = basename.split('_')
                    task_ = None
                    for part in parts:
                        if part[:3] == 'sub':
                            subject = part
                        elif part[:3] == 'ses':
                            session = part
                        elif part[:5] == 'task-':
                            task_ = part[5:]
                        elif part[:4] == 'dir-':
                            acquisition = part[4:]
                    if task not in TASKS:
                        continue
                    paths.append(rp_file)
                    sessions.append(session)
                    subjects.append(subject)
                    modalities.append('bold')
                    contrasts.append('motion')
                    tasks.append(task_)
                    acquisitions.append(acquisition)

    # create a dictionary with all the information
    db_dict = dict(
        path=paths,
        subject=subjects,
        modality=modalities,
        contrast=contrasts,
        session=sessions,
        task=tasks,
        acquisition=acquisitions,
    )
    # create a DataFrame out of the dictionary and write it to disk
    db = pd.DataFrame().from_dict(db_dict)
    return db

def compute_tsnr_(img, masker):
    x = masker.transform(img)
    return(x.mean(0) / np.maximum(1.e-8, x.std(0)))

def compute_tsnr(imgs, masker, n_jobs=6):
    """Compute the SNR of the provided image"""
    import numpy as np
    tsnr = Parallel(n_jobs=n_jobs)(delayed(compute_tsnr_)(img, masker)
                              for img in imgs)
    return(np.array(tsnr))


def average_brain_mask(derivatives=DERIVATIVES):
    """Compute an average brain masks across all the brain masks available"""
    from nilearn.masking import intersect_masks
    masks = glob.glob(os.path.join(derivatives, 'sub-*/ses-00/mask.nii.gz'))
    return(intersect_masks(masks, .25))


def motion_histogram(db):
    """compute motion histograms from realignment files"""
    rps = list(db[db.contrast == 'motion'].path)
    n_bins = 100
    bins = np.linspace(-2, 2, n_bins + 1)
    H = np.zeros((6, n_bins))
    xlist = np.empty((6, 0))
    for rp in rps:
        X = np.loadtxt(rp).T
        X[3:] *= (180. / np.pi)
        H += np.array([np.histogram(x, bins)[0] for x in X])
        xlist = np.hstack((xlist, X))

    # Process values do get convidence intervals
    xlist.sort(1)
    left, right = int(.0005 * xlist.shape[1]), int(.9995 * xlist.shape[1])
    print('.999 confindence interval')
    print(xlist[:, left])
    print(xlist[:, right])
    left, right = int(.005 * xlist.shape[1]), int(.995 * xlist.shape[1])
    print('.99 confindence interval')
    print(xlist[:, left])
    print(xlist[:, right])

    # Plot the histograms
    H = (H.T / H.sum(1))
    mbins = .5 * (bins[1:] + bins[:-1])
    plt.figure(figsize=(6, 4))
    plt.plot(mbins, H, linewidth=1)
    plt.fill(mbins, H, alpha=.3)
    plt.legend(['translation x', 'translation y', 'translation z',
                'rotation x', 'rotation y', 'rotation z'], fontsize=10)
    plt.xlabel('mm/degrees')
    plt.ylabel('normalized histogram')
    plt.title('Histogram of motion parameters')
    # plot the support
    for i, color  in enumerate(['b', 'g', 'r', 'c', 'm', 'y']):
        plt.plot([xlist[i, left], xlist[i, right]],
                 [-0.001 - .003 * i, -.001 - .003 * i], linewidth=3,
                 color=color)
    # dummy line for
    plt.plot([xlist[i, left], xlist[i, right]], [-0.018, -.018], color='w')
    plt.axis('tight')
    plt.subplots_adjust(bottom=.12, left=.14)
    plt.savefig(os.path.join(cache, 'rp_%s.png' % sufix), dpi=600)
    # plt.show()

if __name__ == '__main__':
    db = data_parser()
    mask = average_brain_mask()
    mask.to_filename('/tmp/mask.nii.gz')
    masker = NiftiMasker(mask_img=mask, memory=mem).fit()
    imgs = list(db[db.contrast == 'preprocessed'].path)
    tsnr = compute_tsnr(imgs, masker, n_jobs=1)
    tsnr_map = masker.inverse_transform(tsnr.mean(0))
    tsnr_map.to_filename(os.path.join(cache, 'average_tsnr_%s.nii.gz' % sufix))
    # Load pre-computed .nii.gz files
    # tsnr_map = os.path.join(cache, 'average_tsnr_%s.nii.gz' % sufix)
    tsnr_plot = plotting.plot_epi(tsnr_map, vmax=70, colorbar=True)
    tsnr_plot.savefig(os.path.join(cache, 'average_tsnr_%s.png' % sufix), dpi=600)
    motion_histogram(db)
