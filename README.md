# public_analysis_code
Public analysis code for the IBC project

This Python package gives the pipeline used to process the MRI data obtained in
the Individual brain Charting Project. More info on the data can be found at
[webpage](http://github.com/hbp-brain-charting/public_protocols)
[webpage](http://project.inria.fr/IBC/)

The raw data are available here:
[webpage] http://openfmri.org/dataset/ds000244/

The result sof typical analysis are give here:
[webpage] http://neurovault.org/collections/2138

## Details

These script make it possible to preprocess the data
* run topup distortion correction
* run motion correction
* run corectistratyion of the fMRI scans with the individual T1 image
* run spatial normalization of the data
* run a general linear model to obtain brain activity maps for the main contrasts of the experiment.

Dependences are :
* FSL (topup)
* SPM12 for preprocessing
* Freesurfer for sueface-based analysis
* nipype to call SPM12 functions
* pypreprocess to generate preprocessing reports
* Nilearn for various functions
* Nistats to run general Linear models.

The scripts have been used with the following versions of software and environment:
FSL v5.0.9
SPM12 rev 7219
Nipype v0.14.0
Pypreprocess v0.0.1.dev
Nilearn v0.4.0
Nistats v0.0.1.a
Python 2.7.2
Ubuntu 16.04

## Future work

- More high-level analyses scripts
- Scripts for additional datasets not yet available
- scripts for surface-based analysis

## Contributions

Please feel free to report any issue and propose improvements on github.

## Author

Licensed under simplified BSD.

Bertrand Thirion, 2015 - present
