# DWI preprocessing
Preproc is done using MRtrix's `dwidenoise` and FSL's `topup` and `eddy` in script `dmri_preprocessing_tractography.py`

## Steps
0. 4 runs of dwimages concatenated, similarly for bvecs and bvals
1. Denoising: used `dwidenoise`, only denoising dwimages
2. FSL Topup:
    * collate B0 volumes, these are specific volumes in the dwimages indexed 0, 60, 122, 183
    * make an acquisition parameter matrix reqd by the FSL topup func, second column in the matrix is the phase encoding direction, should be 1 for ap and -1 for pa [FSL topup doc](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/topup/TopupUsersGuide#A--datain) 
    * calculate distortion from the collated b0 images using FSL's `topup` and a set of parameters given in `b02b0.cnf` file
3. Masking (to be used with eddy correction), created using fslmaths
4. FSL Eddy correction using `eddy_cuda9.1`: the input index file has 1s and 3s in it which is correct **just update the comment in the script saying it is 1s and 2s**
5. No bias field correction done

# DWI tractography
Tractography is done using MRtrix in script `dmri_preprocessing_tractography.py`

## Steps
0. The output of eddy correction from above is converted to .mif format that is compatible with MRtrix
1. Generate response functions: using `dwi2response` on eddy corrected images
2. Calculate fiber orientation densities (estimates of amt of diffusion in 3 orthogonal directions): using `dwi2fod` with multi-shell multi-tissue constrained spherical deconvolution
3. Get grey-matter white-matter boundary: using `5tt2gmwmi`
4. Generate streamlines: using `tckgen`

## DWI connectome
* can sift weights generated from before tract2mni transformation be used to generate the connectome after transformation?
    - posted on MRTrix forum https://community.mrtrix.org/t/are-sift2-weights-still-interpretable-following-non-linear-transformation/6162
    - seems like they can be
* gen_connectome fails for sub-06
    - ~~TODO: check if tract2mni has worked as expected~~
    - very weird warps for sub-06
    - also for 07, 11, 15
    - ~~TODO: plot tracts on mni~~ except sub-06, others are fine
* TODO: look into intermediate files for mni transformation. use with different parameters?
* TODO: plot dwi connectome using `nilearn.plotting.plot_connectome`

## Resting state functional connectome
* ~~where are the confounds? Yasmin found some on EBRAINS release, use those?~~ found 'em
* two sessions of rs-fmri, use both?
* ap and pa which one to use? or how are they combined for distortion correction?
* Partial correlations exceed -1.0

## Similarity
* ~~TODO: use correlation coeff for measuring similarity~~ similar heatmaps
* functional connectivity obviously has high across-hemisphere correlations, which won't be the case for structural connectivity
    - ~~TODO: Analysis for each hemisphere separately?~~ results not that different
* try using correlation matrix instead of partial correlation matrix for functional connectivity
* try using non SIFT-weighted connectivity matrix for structural connectivity

## Others
* Some high negative / low positive correlations between different parcels. why?
* TODO: nilearn PR for region coordinates in Schaefer 2018 atlas fetcher
