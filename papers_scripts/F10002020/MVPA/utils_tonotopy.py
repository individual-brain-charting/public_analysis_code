"""
Util functions for the decoding of the tonotopy task of the IBC

Author: Juan Jesus Torre
Email: juanjesustorre@gmail.com
"""

import glob
import librosa
import os
import re

import nibabel as nib
import numpy as np

from pandas import read_csv

from ibc_public.utils_data import get_subject_session

from nilearn.image import high_variance_confounds
from nistats.design_matrix import make_first_level_design_matrix


def make_spectrogram(wavfile, n_fft=2048, hop_length=512, n_mels=128):
    """
    Create the mel spectrogram corresponding to a sound file

    Parameters
    ----------

    wavfile: str or path object
             Absolute path to the file
    
    n_fft: int
           Fourier transform parameter. Determines the number of rows in the
           STFT matrix
    
    hop_length: int
                Fourier transform parameter. Determines the number of columns
                in the SFTF matrix
    
    n_mels: int
            Mel spectrogram parameter. Determines the number of logarithmically
            spaced frequency bins for the final spectrogram
    
    Returns
    -------

    spectrogram: np.array
                 Array representing the spectrogram of the input sound
    """
    # Load the file
    signal, sampling_rate = librosa.load(wavfile)

    # Create mel spectrogram
    spec = librosa.feature.melspectrogram(signal, sr=sampling_rate, 
                                          n_fft=n_fft, 
                                          hop_length=hop_length,
                                          n_mels=n_mels)

    spec = librosa.power_to_db(spec, ref=np.max)

    # # Reduce the dimensionality in the time axis
    # assert spec.shape[1] == 44

    # step = 4

    # arrays = []
    # for i in range(spec.shape[1] // step):
    #     window = spec[:, i * step: (i + 1) * step]
    #     col = np.mean(window, axis=1)
    #     col = col[:, np.newaxis]
    #     arrays.append(col)
    
    # spectrogram = np.hstack(arrays)

    spectrogram = np.mean(spec, axis=1)
    spectrogram = spectrogram[:, np.newaxis]

    spectrogram = spectrogram.flatten()

    return spectrogram


def parse_z(path, conditions):
    """
    Get images, sub, group and labels from a directory of images

    Parameters
    ----------

    path: str
          path to the dir with the images. No files besides the images can be
          in the directory. Images must be .nii or .nii.gz images

    conditions: list of str
                labels to search for in the target folders

    Returns
    -------

    filenames: list
               list of paths the z-maps of the directory
    """

    # Get all (subject, session) pairs
    task_list = ['audio1', 'audio2']
    
    filenames = []

    for task in task_list:
        session_list = sorted(get_subject_session(task))
            
        for sub, ses in session_list:
            session_path = os.path.join(path, sub, ses)
            if not os.path.exists(session_path):
                print("Folder not found for {}, {}. Skipping...".format(sub, ses))
                continue

            print("Code reached for {}, {}".format(sub, ses))
            # ses_glob = glob.glob(os.path.join(session_path, "res_stats_audio_*_*/stat_maps"))
            ses_glob = glob.glob(os.path.join(session_path, "res_stats_audio_*_*/z_score_maps"))
            for run_glob in ses_glob:
                try:
                    file_list = [os.path.join(run_glob, file) for
                                 file in os.listdir(run_glob) if
                                 any(x + '-others.nii.gz' in file for x in conditions)]
                except IndexError:
                    print("Found empty folder for {}, {}. Skipping...".format(sub, ses))
                    continue

                filenames.extend(file_list)

    return filenames


def parse_dir(path: str):
    """
    Get images, sub, group and labels from a directory of images

    Parameters
    ----------

    path: str
          path to the dir with the images. No files besides the images can be
          in the directory. Images must be .nii or .nii.gz images

    Returns
    -------

    filenames = list
                list of paths to all the files of the directory

    subjects = np.array
               numpy array with the subject name corresponding to each image

    runs = np.array
           numpy array with the numbers corresponding to the runs of each
           image

    labels = np.array
             array with the labels corresponding to each sample
    """

    # Parse the components
    parsed = [tuple(x[4:].strip('.nii.gz').split('_')) for x in all_images]
    subjects, _, run, labels, _ = (list(i) for i in zip(*parsed))

    filenames = [os.path.join(path, image) for image in all_images]
    subjects = np.array(subjects, dtype=int)
    runs = np.array(run, dtype=int)
    labels = np.array(labels)

    return filenames, subjects, runs, labels


def sort_by_chunk(string: str, chunk_n: int = -2, sep: str = '_') -> str:
    """Get a chunk of an str to use as key for sorting"""

    return string.split(sep)[chunk_n]


def make_dmtxs(events, fmri, confounds=None,
               t_r=2, mumford=True, task='audio'):
    """
    Generates the design matrices to fit a GLMs approach to a particular
    fmri session. Every design matrix contains one regressor for the trial
    of interest, and another regressor that sums every other trial

    Parameters
    ----------

    events: tsv file
            contains information about the onset, duration and condition
            of the images

    fmri: 4D nifti file
          neuroimaging data

    confounds: txt file, default=None
               file with information about the confounds

    t_r: int, default=2
         repetition time of the acquisition in seconds

    mumford: bool, default True
             variable grouping criteria for each design matrix.

             If True, each trial will keep its labeling name and every other
             trial will be grouped in a 'nuisance' regressor.

             If False, each trial will keep an unique name, all the other
             trials of its same category will be grouped in another regressor,
             and each other category will be modeled separately

    Returns
    -------

    design_matrix_list: list of pandas.DataFrame objects
                        one design matrix per trial, with said trial as the
                        regressor of interest and all other trials as nuisance
                        regressors

    trial_names: list of str
                 Original names of the trials. Used to generate spectrograms
    """

    n_scans = nib.load(fmri).shape[3]
    
    # define the time stamps for different images
    frame_times = np.linspace(0, (n_scans - 1) * t_r, n_scans)
    
    if task == 'audio':
        mask = np.array([1, 0, 1, 1, 0, 1, 1, 0, 1, 1])
        n_cycles = 28
        cycle_duration = 20
        t_r = 2
        cycle = np.arange(0, cycle_duration, t_r)[mask > 0]
        frame_times = np.tile(cycle, n_cycles) +\
            np.repeat(np.arange(n_cycles) * cycle_duration, mask.sum())
        frame_times = frame_times[:-2]  # for some reason...

    paradigm = read_csv(events, sep='\t')
    split_trials = [condition.split('_')[0] for condition in paradigm['trial_type']]

    if confounds:
        motion = ['tx', 'ty', 'tz', 'rx', 'ry', 'rz']
        conf = np.loadtxt(confounds)
    else:
        conf = None

    design_matrix_list = []
    trial_names = []

    trial_n = len(paradigm.index)
    for i in range(trial_n):
        paradigm_copy = paradigm.copy()

        trial_type = paradigm_copy['trial_type']
        this_trial = trial_type.iloc[i]

        if mumford:
            paradigm_copy['trial_type'] = np.where(trial_type.index == i,
                                                   trial_type,
                                                   'nuisance')
        else:
            paradigm_copy['trial_type'] = np.where(trial_type.index == i,
                                                   "{}_00".format(trial_type[i].split("_")[0]),
                                                   split_trials)

        dmtx = make_first_level_design_matrix(frame_times,
                                              events=paradigm_copy,
                                              hrf_model='spm',
                                              add_regs=conf,
                                              add_reg_names=motion)

        design_matrix_list.append(dmtx)
        trial_names.append(this_trial)

    return design_matrix_list, trial_names


def make_dmtx(events, fmri, mask_img, confounds=None, 
              t_r=2, compcorr=False, task='audio',
              normalize=True):
    """
    Generates the design matrix to fit the single GLM approach to a
    particular fmri session

    Parameters
    ----------

    events: tsv file
            contains information about the onset, duration and condition
            of the images

    fmri: 4D nifti file
          neuroimaging data

    mask_img: nifti-like object
              mask image to compute high variance confounds

    confounds: txt file, default=None
               file with information about the confounds

    t_r: int, default=2
         repetition time of the acquisition in seconds
    
    compcorr: bool, default=False
              whether to estimate high variance confounds or not
    
    normalize: bool, default=True
               If True, normalize the stim (i.e., give them 
               arbitrary numbers from 0 to n)

    Returns
    -------

    design_matrix: pandas.DataFrame object
                   design matrix with one trial per column
    """
    n_scans = nib.load(fmri).shape[3]

    # define the time stamps for different images
    frame_times = np.linspace(0, (n_scans - 1) * t_r, n_scans)
    if task == 'audio':
        mask = np.array([1, 0, 1, 1, 0, 1, 1, 0, 1, 1])
        n_cycles = 28
        cycle_duration = 20
        t_r = 2
        cycle = np.arange(0, cycle_duration, t_r)[mask > 0]
        frame_times = np.tile(cycle, n_cycles) +\
            np.repeat(np.arange(n_cycles) * cycle_duration, mask.sum())
        frame_times = frame_times[:-2]  # for some reason...

    paradigm = read_csv(events, sep='\t')

    if normalize:
        paradigm['trial_type'] = [condition.split('_')[0] for condition
                                  in paradigm['trial_type']]
    
    if confounds:
        motion = ['tx', 'ty', 'tz', 'rx', 'ry', 'rz']
        conf = np.loadtxt(confounds)
        if compcorr:
            hv_conf = high_variance_confounds(fmri, mask_img=mask_img)
            conf = np.hstack((hv_conf, conf))
            motion = ['conf_%d' % i for i in range(5)] + motion
    else:
        conf = None
        
    if normalize:        
        trial_type = paradigm["trial_type"].values
        for condition in set(trial_type):
            n_conds = (trial_type == condition).sum()
            trial_type[trial_type == condition] = ['%s_%02d' % (condition, i)
                                                    for i in range(n_conds)]

        paradigm["trial_type"] = trial_type

    dmtx = make_first_level_design_matrix(frame_times, events=paradigm,
                                          hrf_model='spm',
                                          add_regs=conf,
                                          add_reg_names=motion)
    return dmtx
