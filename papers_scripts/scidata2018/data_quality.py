"""
This script generates figures that assess low-level data quality:
* amount of motion
* tSNR of the data

Authors: Bertrand Thirion, 2017
"""
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from joblib import Memory, Parallel, delayed
from nilearn import plotting

SOURCEDATA = '/neurospin/ibc/sourcedata'
DERIVATIVES = '/neurospin/ibc/derivatives'
SUBJECTS = [os.path.basename(full_path) for full_path in
            sorted(glob.glob(os.path.join(DERIVATIVES, 'sub-*')))]
TASKS = ['ArchiSocial', 'ArchiSpatial', 'ArchiStandard', 'ArchiEmotional',
         'HcpRelational', 'HcpWm', 'HcpSocial', 'HcpEmotion', 'HcpMotor',
         'HcpGambling', 'HcpLanguage', 'NspLanguage']
mem = '/neurospin/tmp/bthirion'

def data_parser(derivatives=DERIVATIVES):
    """Generate a dataframe that contains all the data corresponding
    to the archi, hcp and rsvp_language acquisitions"""
    paths = []
    subjects = []
    sessions = []
    modalities = []
    contrasts = []
    tasks = []
    acquisitions = []

    # T1 images
    imgs_ = sorted(glob.glob(os.path.join(
        derivatives, 'sub-*/ses-*/anat/wsub*_T1w_nonan.nii.gz')))
    for img in imgs_:
        session = img.split('/')[-3]
        subject = img.split('/')[-4]
        paths.append(img)
        sessions.append(session)
        subjects.append(subject)
        modalities.append('T1')
        contrasts.append('t1')
        tasks.append('')
        acquisitions.append('')
        
    # gray-matter images
    imgs_ = sorted(glob.glob(os.path.join(
        derivatives, 'sub-*/ses-*/anat/mwcc1sub*_T1w.nii.gz')))
    for img in imgs_:
        session = img.split('/')[-3]
        subject = img.split('/')[-4]
        paths.append(img)
        sessions.append(session)
        subjects.append(subject)
        modalities.append('T1')
        contrasts.append('gm')
        tasks.append('')
        acquisitions.append('')
        
    # white-matter image
    imgs_ = sorted(glob.glob(os.path.join(
        derivatives, 'sub-*/ses-*/anat/mwc2sub*_T1w.nii.gz')))
    for img in imgs_:
        session = img.split('/')[-3]
        subject = img.split('/')[-4]
        paths.append(img)
        sessions.append(session)
        subjects.append(subject)
        modalities.append('T1')
        contrasts.append('wm')
        tasks.append('')
        acquisitions.append('')
    
    # preprocessed images
    bold = sorted(glob.glob(os.path.join(
        derivatives, 'sub-*/ses-*/func/wrdcsub*_ses*_task*_bold.nii.gz')))

    for img in bold:
        basename = os.path.basename(img)
        parts = basename.split('_')
        task = None
        for part in parts:
            if part[:3] == 'sub':
                subject = part
            elif part[:3] == 'ses':
                session = part
            elif part[:5] == 'task-':
                task = part[5:]
            elif part[:4] == 'dir-':
                acquisition = part[4:]

        if task in ['NspLanguage%02d' % i for i in range(6)]:
            task = 'NspLanguage'
        if task not in TASKS:
            continue
        paths.append(img)
        sessions.append(session)
        subjects.append(subject)
        modalities.append('bold')
        contrasts.append('preprocessed')
        tasks.append(task)
        acquisitions.append(acquisition)

    rps = sorted(glob.glob(os.path.join(
        derivatives, 'sub-*/ses-*/func/rp_dcsub*_ses*_task*_bold.txt')))

    for rp_file in rps:
        basename = os.path.basename(rp_file)
        parts = basename.split('_')
        task = None
        for part in parts:
            if part[:3] == 'sub':
                subject = part
            elif part[:3] == 'ses':
                session = part
            elif part[:5] == 'task-':
                task = part[5:]
            elif part[:4] == 'dir-':
                acquisition = part[4:]

        if task in ['NspLanguage%02d' % i for i in range(6)]:
            task = 'NspLanguage'
        if task not in TASKS:
            continue
        paths.append(rp_file)
        sessions.append(session)
        subjects.append(subject)
        modalities.append('bold')
        contrasts.append('motion')
        tasks.append(task)
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

    # create a FataFrame out of the dictionary and write it to disk
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
    """Compute an avergae brain masks across all the brain masks available"""
    from nilearn.masking import intersect_masks
    masks = glob.glob(os.path.join(derivatives, 'sub-*/ses-*/mask.nii.gz'))
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
    plt.figure(figsize=(5, 4))
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
                 [-0.001 - .003 * i, -.001 - .003 * i], linewidth=3, color=color)
    # dummy line for
    plt.plot([xlist[i, left], xlist[i, right]], [-0.018, -.018], color='w')
    plt.axis('tight')
    plt.subplots_adjust(bottom=.12, left=.14)
    plt.savefig(os.path.join('output', 'rp.pdf'))
    plt.show()

    
if __name__ == '__main__':
    db = data_parser()
    """
    mask = average_brain_mask()
    mask.to_filename('/tmp/mask.nii.gz')
    from nilearn.input_data import NiftiMasker
    masker = NiftiMasker(mask_img=mask, memory=mem).fit()
    imgs = list(db[db.contrast == 'preprocessed'].path)
    tsnr = compute_tsnr(imgs, masker, n_jobs=6)
    tsnr_map = masker.inverse_transform(tsnr.mean(0))
    tsnr_map.to_filename(os.path.join('output', 'average_tsnr.nii.gz'))
    plotting.plot_epi(tsnr_map, vmax=60, colorbar=True,
                      output_file=os.path.join('output', 'average_tsnr.pdf'))
    """
    motion_histogram(db)
