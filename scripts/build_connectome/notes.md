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
