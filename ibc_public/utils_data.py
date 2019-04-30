"""
Common utilities to all scripts

Author: Bertrand Thirion, 2016-2018
"""

import glob
import os
import pandas as pd
import shutil

main_parent_dir = '/neurospin/ibc'
alt_parent_dir = '/storage/store/data/ibc'

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
    _package_directory, '..', 'ibc_data', 'conditions.tsv'), sep='\t')
CONTRASTS = pd.read_csv(os.path.join(
    _package_directory, '..', 'ibc_data', 'main_contrasts.tsv'), sep='\t')
ALL_CONTRASTS = pd.read_csv(os.path.join(
    _package_directory, '..', 'ibc_data', 'all_contrasts.tsv'), sep='\t')

# Note that LABELS and BETTER NAMES ARE RELATIVE TO CONTRASTS
LABELS = {}
BETTER_NAMES = {}
for i in range(len(CONTRASTS)):
    BETTER_NAMES[CONTRASTS.contrast[i]] = CONTRASTS['pretty name'][i]
    LABELS[CONTRASTS.contrast[i]] = [CONTRASTS['left label'][i],
                                     CONTRASTS['right label'][i]]


def get_subject_session(protocol):
    """ utility to get all (subject, session) for a given protocol"""
    import pandas as pd
    df = pd.read_csv(os.path.join(
        _package_directory, '../ibc_data', 'sessions.csv'), index_col=0)
    subject_session = []
    for session in df.columns:
        if (df[session] == protocol).any():
            subjects = df[session][df[session] == protocol].keys()
            for subject in subjects:
                subject_session.append((subject,  session))
    return subject_session


def data_parser(derivatives=DERIVATIVES, conditions=CONDITIONS,
                subject_list=SUBJECTS, task_list=False):
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
    imgs_ = sorted(glob.glob(os.path.join(
        DERIVATIVES, 'sub-*/ses-*/anat/wsub*_T1w.nii.gz')))
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

    imgs_ = sorted(glob.glob(os.path.join(
        DERIVATIVES, 'sub-*/ses-*/anat/wsub*_T1w_bet.nii.gz')))
    for img in imgs_:
        session = img.split('/')[-3]
        subject = img.split('/')[-4]
        paths.append(img)
        sessions.append(session)
        subjects.append(subject)
        modalities.append('T1')
        contrasts.append('t1_bet')
        tasks.append('')
        acquisitions.append('')

    imgs_ = sorted(glob.glob(os.path.join(
        DERIVATIVES, 'sub-*/ses-*/anat/wsub*_acq-highres_T1w_bet.nii.gz')))
    for img in imgs_:
        session = img.split('/')[-3]
        subject = img.split('/')[-4]
        paths.append(img)
        sessions.append(session)
        subjects.append(subject)
        modalities.append('T1')
        contrasts.append('highres_t1_bet')
        tasks.append('')
        acquisitions.append('')

    # gm images
    imgs_ = sorted(glob.glob(os.path.join(
        derivatives, '../derivatives', 'sub-*/ses-*/anat/mwc1su*_T1w.nii.gz')))
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

    imgs_ = sorted(glob.glob(os.path.join(
        derivatives, '../derivatives', 'sub-*/ses-*/anat/mwc1su*_T1w.nii.gz')))
    for img in imgs_:
        session = img.split('/')[-3]
        subject = img.split('/')[-4]
        paths.append(img)
        sessions.append(session)
        subjects.append(subject)
        modalities.append('T1')
        contrasts.append('highres_gm')
        tasks.append('')
        acquisitions.append('')

    # wm image
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

    # fixed-effects activation images
    con_df = conditions
    contrast_name = con_df.contrast

    for acq in ['ap', 'pa', 'ffx']:
        for subject in subject_list:
            for i in range(len(con_df)):
                contrast = contrast_name[i]
                task = con_df.task[i]
                if task_list and (task not in task_list):
                    continue
                task_name = task
                if task == 'rsvp_language':
                    task = 'language'
                    task_name = 'rsvp_language'
                if task == 'mtt_ns':
                    task = 'IslandNS'
                    task_name = 'mtt_ns'
                if task == 'mtt_we':
                    task = 'IslandWE'
                    task_name = 'mtt_we'

                wildcard = os.path.join(
                    derivatives, '%s/*/res_stats_%s*_%s/stat_maps/%s.nii.gz' %
                    (subject, task, acq, contrast))
                imgs_ = glob.glob(wildcard)
                if len(imgs_) == 0:
                    continue
                imgs_.sort()
                # some renaming
                contrast_id = contrast
                if (contrast_id == 'probe') and\
                   (task_name == 'rsvp_language'):
                    contrast_id = 'language_probe'

                for img in imgs_:
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
    import numpy as np
    frequencies_ = np.array(frequencies)
    unique_terms = np.unique(frequencies_.T[0])
    frequencies = [
        (term,
         np.mean(frequencies_[frequencies_[:, 0] == term, 1].astype(np.float)))
        for term in unique_terms]
    return frequencies


def horizontal_fingerprint(coef, seed, roi_name, labels_bottom, labels_top,
                           output_file, wc=True, nonneg=False):
    """Create a fingerprint figure, optionally with a word cloud"""
    import matplotlib.pyplot as plt
    import wordcloud as wc
    import numpy as np
    n_contrasts = coef.size
    plt.figure(figsize=(12, 5), facecolor='k')
    plt.axes([.03, .32, .6, .34])
    colors = plt.cm.hsv(np.linspace(0, 255 / n_contrasts, 255))
    plt.bar(range(n_contrasts), coef, color=colors, ecolor='k')
    plt.xticks(np.linspace(1., n_contrasts + .8, num=n_contrasts + 1),
               labels_bottom, rotation=75, ha='right', fontsize=12, color='w')

    ymax = coef.max() + .01 * (coef.max() - coef.min())
    ymin = coef.min() - .01
    for nc in range(n_contrasts):
        plt.text(nc, ymax, labels_top[nc], rotation=75,
                 ha='left', va='bottom', color='w')
    #
    x, y, z = seed
    plt.text(n_contrasts / 3, 1.7 * ymax - .7 * ymin,
             ' %s (%d, %d, %d)' % (roi_name, x, y, z),
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
    wordcloud = wc.WordCloud().generate_from_frequencies(frequencies)
    plt.axes([.65, .1, .33, .7])
    plt.imshow(wordcloud)
    plt.axis('tight')
    plt.axis("off")
    plt.savefig(output_file, facecolor='k')


def copy_db(df, write_dir, filename='result_db.csv'):
    # Create a copy of all the files to create a portable database
    if not os.path.exists(write_dir):
        os.mkdir(write_dir)
    df1 = df.copy()
    paths = []
    for i in df.index:
        if df.iloc[i].task is not '':
            fname = '%s_%s_%s_%s_%s.nii.gz' % (
                df.iloc[i].modality, df.iloc[i].subject, df.iloc[i].session,
                df.iloc[i].task, df.iloc[i].contrast)
        else:
            fname = '%s_%s_%s_%s.nii' % (
                df.iloc[i].modality, df.iloc[i].subject, df.iloc[i].session,
                df.iloc[i].contrast)
        print(fname)
        new_path = os.path.join(write_dir, fname)
        shutil.copy(df.iloc[i].path, new_path)
        paths.append(fname)
    df1.path = paths
    # TODO: add mask !
    if filename is not None:
        df1.to_csv(os.path.join(write_dir, filename))
    return df1


def make_surf_db(main_dir=DERIVATIVES, conditions=False, subject_list=SUBJECTS):
    """ Create a database for surface data (gifti files)

    main_dir: string,
              directory where the studd will be found

    conditions: Bool, optional,
                Whether to map conditions or contrasts
    """
    imgs = []
    sides = []
    subjects = []
    sessions = []
    contrasts = []
    tasks = []

    # fixed-effects activation images
    if conditions is True:
        con_df = CONDITIONS
        contrast_name = con_df.condition
    elif conditions == 'all':
        ALL = pd.read_csv('all_contrasts.tsv', sep='\t')
        con_df = ALL
        contrast_name = con_df.index
    else:
        con_df = CONTRASTS
        contrast_name = con_df.contrast

    for subject in subject_list:
        for i in range(len(con_df)):
            contrast = contrast_name[i]
            task = con_df.task[i]
            task_name = task
            if task == 'rsvp_language':
                task = 'language'
                task_name = 'rsvp_language'
            if task == 'mtt_ns':
                task = 'IslandNS'
                task_name = 'mtt_ns'
            if task == 'mtt_we':
                task = 'IslandWE'
                task_name = 'mtt_we'
            # some renaming
            if ((contrast == 'probe') & (task_name == 'rsvp_language')):
                    contrast = 'language_probe'
            for side in ['lh', 'rh']:
                wc = os.path.join(
                    main_dir, subject, '*/res_surf_%s_ffx/stat_surf/%s_%s.gii'
                    % (task, contrast, side))
                imgs_ = glob.glob(wc)
                imgs_.sort()
                for img in imgs_:
                    session = img.split('/')[5]
                    imgs.append(img)
                    sessions.append(session)
                    subjects.append(img.split('/')[4])
                    contrasts.append(contrast)
                    tasks.append(task_name)
                    sides.append(side)
            if task == 'language_':
                pass # stop

    # create a dictionary with all the information
    db_dict = dict(
        path=imgs,
        subject=subjects,
        contrast=contrasts,
        session=sessions,
        task=tasks,
        side=sides
    )

    # create a FataFrame out of the dictionary and write it to disk
    db = pd.DataFrame().from_dict(db_dict)
    return db
