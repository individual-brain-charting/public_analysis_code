"""
Common utilities to all scripts

Author: Bertrand Thirion, Ana Luisa Pinho 2016-2020

Compatibility: Python 3.5

"""

import os
import glob
import warnings
from collections import defaultdict
import pandas as pd
import shutil
import numpy as np
# from ibc_public.utils_annotations import expand_table
from tqdm import tqdm

main_parent_dir = '/neurospin/ibc'
alt_parent_dir = '/storage/store2/data/ibc'

if os.path.exists(main_parent_dir):
    ibc = main_parent_dir
else:
    ibc = alt_parent_dir

DERIVATIVES = os.path.join(ibc, 'derivatives')
SMOOTH_DERIVATIVES = os.path.join(ibc, 'smooth_derivatives')
THREE_MM = os.path.join(ibc, '3mm')

SUBJECTS = ['sub-%02d' % i for i in
            [1, 2, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15]]
_package_directory = os.path.dirname(os.path.abspath(__file__))
# Useful for the very simple examples
CONDITIONS = pd.read_csv(os.path.join(
    _package_directory, '..', 'ibc_data', 'ibc_conditions.tsv'), sep='\t')
CONTRASTS = pd.read_csv(os.path.join(
    _package_directory, '..', 'ibc_data', 'main_contrasts.tsv'), sep='\t')
ALL_CONTRASTS = os.path.join(
    _package_directory, '..', 'ibc_data', 'all_contrasts.tsv')


# Note that LABELS and BETTER NAMES ARE RELATIVE TO CONTRASTS
LABELS = {}
BETTER_NAMES = {}
all_contrasts = pd.read_csv(ALL_CONTRASTS, sep='\t')
for i in range(len(all_contrasts)):
    task = all_contrasts.task[i]
    contrast = all_contrasts.contrast[i]
    target = all_contrasts[
        (all_contrasts.task == task) &
        (all_contrasts.contrast == contrast)
    ]
    if len(target['pretty name'].values):
        BETTER_NAMES[contrast] = target['pretty name'].values[0]
    else:
        BETTER_NAMES[contrast] = contrast
    pos, neg = contrast, ''
    if len(target['negative label'].values):
        neg = target['negative label'].values[0]
    if len(target['positive label'].values):
        pos = target['positive label'].values[0]
    LABELS[contrast] = [neg, pos]


def get_subject_session(protocols):
    """
    Utility to get all (subject, session) for a given protocol or set
    of protocols

    Parameters
    ----------
    protocols: string or list,
               name(s) of the protocols the user wants to retrieve

    Returns
    -------
    subject_session: list of tuples
                     Each element correspondes to a (subject, session) pair
                     for the requested protocols
    """
    import pandas as pd
    df = pd.read_csv(os.path.join(
        _package_directory, '../ibc_data', 'sessions.csv'), index_col=0)
    subject_session = []

    # corerce to list
    if isinstance(protocols, str):
        protocols_ = [protocols]
    else:
        protocols_ = protocols

    for protocol in protocols_:
        for session in df.columns:
            if (df[session] == protocol).any():
                subjects = df[session][df[session] == protocol].keys()
                for subject in subjects:
                    subject_session.append((subject,  session))
    return subject_session


def data_parser(derivatives=DERIVATIVES, conditions=CONDITIONS,
                subject_list=SUBJECTS, task_list=False, verbose=0,
                acquisition_dir='all'):
    """Generate a dataframe that contains all the data corresponding
    to the archi, hcp and rsvp_language acquisitions

    Parameters
    ----------
    derivatives: string, optional
        path toward a valid BIDS derivatives directory

    conditions: pandas DataFrame, optional,
        dataframe describing the conditions under considerations

    subject_list: list, optional,
        list of subjects to be included in the analysis

    task_list: list_optional,
        list of tasks to be returned

    verbose: Bool, optional,
             verbosity mode

    acquisition_dir={'all', 'ap', 'pa', 'ffx'}, default='all'
        which acquisition to select

    Returns
    -------
    db: pandas DataFrame,
        "database" yielding informations (path,
        subject, modality, contrast, session, task, acquisition)
        on the images under consideration
    """
    paths = []
    subjects = []
    sessions = []
    modalities = []
    contrasts = []
    tasks = []
    acquisitions = []

    # T1 images
    for subject in subject_list:
        t1_path = 'sub-*/ses-*/anat/w%s_ses-00_T1w.nii.gz' % subject
        t1_abs_path = os.path.join(DERIVATIVES, t1_path)
        t1_imgs_ = glob.glob(os.path.join(t1_abs_path))
        for img in t1_imgs_:
            session = img.split('/')[-3]
            subject_id = img.split('/')[-4]
            paths.append(img)
            sessions.append(session)
            subjects.append(subject_id)
            modalities.append('T1')
            contrasts.append('t1')
            tasks.append('')
            acquisitions.append('')

    for subject in subject_list:
        t1bet_path = 'sub-*/ses-*/anat/w%s_ses-00_T1w_bet.nii.gz' % subject
        t1bet_abs_path = os.path.join(DERIVATIVES, t1bet_path)
        t1bet_imgs_ = glob.glob(os.path.join(t1bet_abs_path))
        for img in t1bet_imgs_:
            session = img.split('/')[-3]
            subject_id = img.split('/')[-4]
            paths.append(img)
            sessions.append(session)
            subjects.append(subject_id)
            modalities.append('T1')
            contrasts.append('t1_bet')
            tasks.append('')
            acquisitions.append('')

    for subject in subject_list:
        ht1_path = 'sub-*/ses-*/anat/w%s*_acq-highres_T1w_bet.nii.gz' % subject
        ht1_abs_path = os.path.join(DERIVATIVES, ht1_path)
        ht1_imgs_ = glob.glob(os.path.join(ht1_abs_path))
        for img in ht1_imgs_:
            session = img.split('/')[-3]
            subject_id = img.split('/')[-4]
            paths.append(img)
            sessions.append(session)
            subjects.append(subject_id)
            modalities.append('T1')
            contrasts.append('highres_t1_bet')
            tasks.append('')
            acquisitions.append('')

    # gray-matter images
    for subject in subject_list:
        mwc1_path = 'sub-*/ses-*/anat/mwc1%s_ses-00_T1w.nii.gz' % subject
        mwc1_abs_path = os.path.join(DERIVATIVES, mwc1_path)
        mwc1_imgs_ = glob.glob(os.path.join(mwc1_abs_path))
        for img in mwc1_imgs_:
            session = img.split('/')[-3]
            subject_id = img.split('/')[-4]
            paths.append(img)
            sessions.append(session)
            subjects.append(subject_id)
            modalities.append('T1')
            contrasts.append('gm')
            tasks.append('')
            acquisitions.append('')

    for subject in subject_list:
        hmwc1_path = 'sub-*/ses-*/anat/mwc1%s*_acq-highres_T1w.nii.gz' % subject
        hmwc1_abs_path = os.path.join(DERIVATIVES, hmwc1_path)
        hmwc1_imgs_ = glob.glob(os.path.join(hmwc1_abs_path))
        for img in hmwc1_imgs_:
            session = img.split('/')[-3]
            subject_id = img.split('/')[-4]
            paths.append(img)
            sessions.append(session)
            subjects.append(subject_id)
            modalities.append('T1')
            contrasts.append('highres_gm')
            tasks.append('')
            acquisitions.append('')

    # white-matter image
    for subject in subject_list:
        mwc2_path = 'sub-*/ses-*/anat/mwc2%s_ses-00_T1w.nii.gz' % subject
        mwc2_abs_path = os.path.join(DERIVATIVES, mwc2_path)
        mwc2_imgs_ = glob.glob(os.path.join(mwc2_abs_path))
        for img in mwc2_imgs_:
            session = img.split('/')[-3]
            subject_id = img.split('/')[-4]
            paths.append(img)
            sessions.append(session)
            subjects.append(subject_id)
            modalities.append('T1')
            contrasts.append('wm')
            tasks.append('')
            acquisitions.append('')

    # preprocessed bold images and corresponding motion parameters
    if derivatives in [DERIVATIVES, THREE_MM]:
        for sbj in subject_list:
            for acq in ['ap', 'pa']:
                for task in task_list:
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
                        if task not in task_list:
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
                    rps_path = os.path.join(DERIVATIVES, 'sub-*/ses-*/func',
                                            rps_name)
                    rps = glob.glob(rps_path)
                    if not rps:
                        # Add exception for 'bang' task, since 'ap' was
                        # never part of acq planning
                        if acq == 'ap' and task == 'Bang':
                            pass
                        else:
                            msg = 'rp-dc*.txt file for task ' + \
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
                        if task not in task_list:
                            continue
                        paths.append(rp_file)
                        sessions.append(session)
                        subjects.append(subject)
                        modalities.append('bold')
                        contrasts.append('motion')
                        tasks.append(task_)
                        acquisitions.append(acquisition)

    # fixed-effects activation images (postprocessed)
    con_df = conditions
    contrast_name = con_df.contrast

    acq_card = '*' # if acquisition == 'all'
    if acquisition_dir in ['ffx', 'ap', 'pa']:
        acq_card = 'dir-%s' % acquisition_dir

    for subject in subject_list:
            for i in range(len(con_df)):
                contrast = contrast_name[i]
                task = con_df.task[i]
                task_name = task

                if (task_list is not None) and (task not in task_list):
                    if verbose:
                        print('%s found as task, not in task_list' % task)
                    continue
                
                wildcard = os.path.join(
                    derivatives, subject, '*',
                    'res_task-%s_space-MNI152*_%s' % (task, acq_card),
                    'stat_maps', '%s.nii.gz' % contrast)
                imgs_ = glob.glob(wildcard)
                if len(imgs_) == 0:
                    print('Missing %s' % wildcard)
                imgs_.sort()
                # some renaming
                contrast_id = contrast
                for img in imgs_:
                    acq = 'unknown'
                    if 'dir-' in img:
                        acq = img.split('dir-')[1][:2]
                        if acq == 'ff':
                            acq = 'ffx'
                    session = img.split('/')[-4]
                    paths.append(img)
                    sessions.append(session)
                    subjects.append(img.split('/')[-5])
                    modalities.append('bold')
                    contrasts.append(contrast_id)
                    tasks.append(task_name)
                    acquisitions.append(acq)

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


def average_anat(db):
    """ return the average anatomical image """
    from nilearn.image import mean_img
    anat_imgs = db[db.contrast == 't1'].path.values
    mean_anat = mean_img(anat_imgs)
    return mean_anat


def gm_mask(db, ref_affine, ref_shape, threshold=.25):
    """ Utility to create a gm mask by averaging glm images from the db """
    from nilearn.image import mean_img
    import nibabel as nib
    gm_imgs = db[db.contrast == 'gm'].path.values
    mean_gm = mean_img(
        gm_imgs, target_affine=ref_affine, target_shape=ref_shape)
    gm_mask = nib.Nifti1Image((mean_gm.get_data() > .25).astype('uint8'),
                              ref_affine)
    return mean_gm, gm_mask


def resample_images(paths, ref_affine, ref_shape):
    """ Utility to resample images provided as paths and return an image"""
    from nilearn.image import resample_img
    import nibabel as nib
    imgs = []
    for path in paths:
        fmri_image = resample_img(
            path, target_affine=ref_affine, target_shape=ref_shape)
        imgs.append(fmri_image)
    img = nib.concat_images(imgs)
    return img


def summarize_db(df, plot=True):
    # create a summary table of the acquired data, organized by contrast
    tasks = pd.unique(df.task[df.task != ''])
    ld = []
    for task in tasks:
        subjects_ = df.subject[df.task == task]
        ld.append(dict([(subject_, (subjects_ == subject_).sum())
                        for subject_ in pd.unique(subjects_)]))

    summary = pd.DataFrame(ld, index=tasks)
    summary[summary.isnull()] = 0

    if plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(5, 5))
        vals = summary.values
        plt.imshow(vals, interpolation='nearest')
        plt.yticks(range(vals.shape[0]), summary.index)
        plt.xticks(range(vals.shape[1]), summary.columns, rotation=60)
        plt.subplots_adjust(left=.23, bottom=.13, right=.97)
        plt.savefig("/tmp/summary.png")
    return summary


def _unique_frequencies(frequencies):
    frequencies_ = np.array(frequencies)
    unique_terms = np.unique(frequencies_.T[0])
    frequencies = dict([
        (term,
         np.mean(frequencies_[frequencies_[:, 0] == term, 1].astype(np.float)))
        for term in unique_terms])
    return frequencies


def _highest_frequency(frequencies):
    values = np.array([x for x in frequencies.values()])
    keys = np.array([x for x in frequencies.keys()])
    return keys[np.argmax(values)]


def horizontal_fingerprint(coef, roi_name, labels_bottom, labels_top,
                           output_file, wc=True, nonneg=False, dpi=300):
    """Create a fingerprint figure, optionally with a word cloud"""
    import matplotlib.pyplot as plt
    import wordcloud as wc
    n_contrasts = coef.size
    plt.figure(figsize=(12, 5.25), facecolor='k')
    plt.axes([.03, .32, .6, .34])
    colors = plt.cm.hsv(np.linspace(0, 255 / n_contrasts, 255))
    plt.bar(range(n_contrasts), coef, color=colors, ecolor='k')
    plt.xticks(np.linspace(1., n_contrasts + .8, num=n_contrasts + 1),
               labels_bottom, rotation=75, ha='right', fontsize=10, color='w')

    ymax = .03 + coef.max() + .01 * (coef.max() - coef.min())
    ymin = coef.min() - .01
    for nc in range(n_contrasts):
        plt.text(nc, ymax, labels_top[nc], rotation=75,
                 ha='left', va='bottom', color='w')
    #
    plt.text(n_contrasts + 10, 1.8 * ymax - .7 * ymin, roi_name,
             color=[1, .5, 0], fontsize=18, weight='bold',
             bbox=dict(facecolor='k', alpha=0.5))
    plt.axis('tight')
    plt.subplots_adjust(bottom=.3, top=.7)
    frequencies = [(labels_top[i], np.exp(coef[i]))
                   for i in range(n_contrasts)]
    if not nonneg:
            frequencies += [(labels_bottom[i], np.exp(-coef[i]))
                            for i in range(n_contrasts)]
    frequencies = _unique_frequencies(frequencies)
    print(_highest_frequency(frequencies))
    wordcloud = wc.WordCloud().generate_from_frequencies(frequencies)
    plt.axes([.66, .1, .33, .7])
    plt.imshow(wordcloud)
    plt.axis('tight')
    plt.axis("off")
    plt.savefig(output_file, facecolor='k', dpi=dpi)


def copy_db(df, write_dir, filename='result_db.csv'):
    """Create a copy of all the files to create a portable database."""
    # Create output folder if it doesn't already exist
    if not os.path.exists(write_dir):
        os.mkdir(write_dir)

    df1 = df.copy()

    # Copy all files listed in df1 to output location
    paths = []
    for i in tqdm(df.index):
        filename_, extension = os.path.splitext(df.iloc[i].path)
        extension_ = os.path.splitext(filename_)[1]
        extension = extension_ +  extension

        # Derive filename depending on whether output is surface or volume
        ## Volume data
        fname = '%s_%s_%s_%s_%s_%s%s' % (
            df.iloc[i].modality, df.iloc[i].subject, df.iloc[i].session,
            df.iloc[i].task, df.iloc[i].contrast, df.iloc[i].mesh, extension
        )
        ## Surface data
        if extension == '.gii':
            fname = '%s_%s_%s_%s_%s_%s_%s%s' % (
            df.iloc[i].modality, df.iloc[i].subject, df.iloc[i].session,
                df.iloc[i].task, df.iloc[i].contrast, df.iloc[i].side,
                df.iloc[i].mesh, extension
            )

        new_path = os.path.join(write_dir, fname)
        shutil.copy(df.iloc[i].path, new_path)
        paths.append(fname)

    # Update df1 paths with new paths
    df1.path = paths

    # TODO: add mask !

    # Copy df to output location
    if filename is not None:
        df1.to_csv(os.path.join(write_dir, filename))

    return df1


def make_db(
    derivatives=DERIVATIVES,
    conditions=CONDITIONS,
    subject_list=SUBJECTS,
    task_list=False,
    space="fsaverage5",
    extension=".gii",
    acquisition="ffx",
):
    """ Create a database for surface data (gifti files)

    Parameters
    ----------
    derivatives: string, optional,
        Directory where the study will be found
    conditions: pandas DataFrame, optional,
        Dataframe describing the conditions under consideration
    subject_list: list, optional,
        list of subjects to be included in the analysis
    task_list: list_optional,
        list of tasks to be returned
    space: string, optional,
        Should be one of ["fsaverage5", "fsaverage7", "individual", "MNI152"],
        Default: "fsaverage5"
    extension: string, optional
        Extension of the files to be found.
        Default: ".gii"
    acquisition: string, optional
        Acquisiton to be picked. One of ["ffx", "ap", "pa", "all"].
        Default: "ffx"

    Returns
    -------
    db: pandas DataFrame,
        "database" yielding informations (path,
        subject, modality, contrast, session, task, acquisition, hemisphere)
        on the gifti files under consideration
    """
    imgs = []
    sides = []
    subjects = []
    sessions = []
    contrasts = []
    tasks = []
    modalities = []
    spaces = []
    acquisitions = []

    # Check that given mesh value is valid
    available_spaces = [
        "fsaverage5",
        "fsaverage7",
        "individual",
        "MNI152",
    ]
    volumetric_spaces = [
        "MNI152",
    ]
    if space not in available_spaces:
        raise ValueError(
            'Space value (%s) unknown ; should be one of %s'
            % (space, available_spaces)
        )

    # fixed-effects activation images
    con_df = conditions
    contrast_name = con_df.contrast
    missing_images_per_subject = defaultdict(list)

    # Without the following print, other prints won't show up ;
    # this is probably an issue with tqdm
    print("Build CSV file rows")

    for subject in tqdm(subject_list, desc="Search subject maps"):
        for i in range(len(con_df)):
            contrast = contrast_name[i]
            task = con_df.task[i]
            if (task_list is not False) and (task not in task_list):
                continue

            # Set directory depending on space type
            if acquisition == 'all':
                dir_ = 'res_task-%s_space-%s*' % (task, space)
            elif acquisition == 'ffx':
                dir_ = 'res_task-%s_space-%s*_dir-ffx' % (task, space)
            elif acquisition in ['ap', 'pa']:
                dir_ = 'res_task-%s_space-%s*_dir-%s' % (
                    task, space, acquisition
                )

            if space in volumetric_spaces:
                selected_imgs_filename = os.path.join(
                    derivatives, subject, '*', dir_, 'stat_maps',
                    '%s%s' % (contrast, extension)
                )
                if acquisition in ['ap', 'pa']:
                    selected_imgs_filename = os.path.join(
                        derivatives, subject, '*', dir_, 'z_score_maps',
                        '%s%s' % (contrast, extension)
                    )
                imgs_ = glob.glob(selected_imgs_filename)
                imgs_.sort()

                # Store missing images
                if len(imgs_) == 0:
                    missing_images_per_subject[subject].append(
                        [subject, contrast, task]
                    )

                for img in imgs_:
                    session = img.split('/')[-4]
                    imgs.append(img)
                    sessions.append(session)
                    subjects.append(img.split('/')[-5])
                    contrasts.append(contrast)
                    tasks.append(task)
                    modalities.append('bold')
                    spaces.append(space)
                    acquisitions.append(acquisition)

            else:
                for side in ['lh', 'rh']:
                    selected_imgs_filename = os.path.join(
                        derivatives, subject, '*', dir_, 'stat_maps',
                        '%s_%s%s' % (contrast, side, extension)
                    )
                    if acquisition in ['ap', 'pa']:
                        selected_imgs_filename = os.path.join(
                            derivatives, subject, '*', dir_, 'z_score_maps',
                            '%s*%s%s' % (contrast, side, extension)
                        )
                    imgs_ = glob.glob(selected_imgs_filename)
                    imgs_.sort()

                    # Display warning when no image is found
                    if len(imgs_) == 0:
                        missing_images_per_subject[subject].append(
                            [subject, contrast, task, side]
                        )

                    for img in imgs_:
                        session = img.split('/')[-4]
                        imgs.append(img)
                        sessions.append(session)
                        subjects.append(img.split('/')[-5])
                        contrasts.append(contrast)
                        tasks.append(task)
                        sides.append(side)
                        modalities.append('bold')
                        spaces.append(space)
                        acquisitions.append(acquisition)

    total_missing_maps = sum([
        len(missing_images_per_subject[subject])
        for subject in missing_images_per_subject.keys()
    ])
    print(f"{len(imgs)} images found, {total_missing_maps} were missing")

    for subject in missing_images_per_subject.keys():
        print(
            f"Missing images for subject {subject}:\t"
            f"{len(missing_images_per_subject[subject])}"
        )

    if space in volumetric_spaces:
        # create a dictionary with all the information
        db_dict = dict(
            path=imgs,
            subject=subjects,
            contrast=contrasts,
            session=sessions,
            task=tasks,
            modality=modalities,
            space=spaces,
            acquisition=acquisitions
        )
    else:
        # create a dictionary with all the information
        db_dict = dict(
            path=imgs,
            subject=subjects,
            contrast=contrasts,
            session=sessions,
            task=tasks,
            side=sides,
            modality=modalities,
            mesh=spaces,
            acquisition=acquisitions
        )

    # create a FataFrame out of the dictionary and write it to disk
    db = pd.DataFrame().from_dict(db_dict)

    return db


def make_surf_db(
    derivatives=DERIVATIVES,
    conditions=CONDITIONS,
    subject_list=SUBJECTS,
    task_list=False,
    mesh="fsaverage5",
    extension=".gii",
    acquisition="ffx"
):
    """Create a CSV file listing available surface maps."""
    return make_db(
        derivatives=derivatives,
        conditions=conditions,
        subject_list=subject_list,
        task_list=task_list,
        space=mesh,
        extension=extension,
        acquisition=acquisition
    )


def make_vol_db(
    derivatives=DERIVATIVES,
    conditions=CONDITIONS,
    subject_list=SUBJECTS,
    task_list=False,
    space="MNI152",
    extension=".nii.gz",
    acquisition="ffx"
):
    """Create a CSV file listing available volumetric maps."""
    return make_db(
        derivatives=derivatives,
        conditions=conditions,
        subject_list=subject_list,
        task_list=task_list,
        space=space,
        extension=extension,
        acquisition=acquisition
    )
