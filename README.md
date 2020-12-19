# Public analysis code for the IBC project

This Python package gives the pipeline used to process the MRI data obtained in
the Individual Brain Charting Project. More info on the data can be found at
[IBC public protocols](http://github.com/hbp-brain-charting/public_protocols)
and
[IBC webpage](http://project.inria.fr/IBC/)
.

The raw data are available here:
[OpenfMRI](http://openfmri.org/dataset/ds000244/)

The result sof typical analysis are give here:
[NeuroVault](http://neurovault.org/collections/2138)

## Install
Under the main working directory of this repository in your computer, run the following command in a command prompt:

```
pip install -e .
```

## Details

These script make it possible to preprocess the data
* run topup distortion correction
* run motion correction
* run coregistration of the fMRI scans to the individual T1 image
* run spatial normalization of the data
* run a general linear model to obtain brain activity maps for the main contrasts of the experiment.

## Core scripts

The core scripts are in the `processing` folder

- `pipeline.py` lunches the full analysis on fMRI data (pre-processing + GLM)
- `glm_only.py` launches GLM analyses on the data
- `surface_based_analyses` launches surface extraction and registration with Freesurfer; it also projects fMRI data to the surface
- `surface_glm_analysis.py` runs glm analyses on the surface
- `dmri_preprocessing` (WIP) is for diffusion daat. It relies on dipy.
- `anatomical mapping` (WIP) yields T1w, T2w and MWF surrogates from anatomical acquisitions.
- `script_retino.py` yields some post-processing for retinotopic acquisitions (derivation of retinotopic representations from fMRI maps)

## Dependences

Dependences are :
* FSL (topup)
* SPM12 for preprocessing
* Freesurfer for surface-based analysis
* Nipype to call SPM12 functions
* Pypreprocess to generate preprocessing reports
* Nilearn for various functions
* Nistats to run general Linear models.

The scripts have been used with the following versions of software and environment:

* Python 3.5
* Ubuntu 16.04
* Nipype v0.14.0
* Pypreprocess v0.0.1.dev
* FSL v5.0.9
* SPM12 rev 7219
* Nilearn v0.4.0
* Nistats v0.0.1.a

## Future work

- More high-level analyses scripts
- Scripts for additional datasets not yet available
- scripts for surface-based analysis

## Contributions

Please feel free to report any issue and propose improvements on Github.

## Authors

Licensed under simplified BSD.

- Bertrand Thirion, 2015 - present
- Ana Luísa Pinho, 2015 - present
- Juan Jesús Torre, 2018 - 2020
