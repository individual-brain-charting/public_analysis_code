## DWI connectome
* can sift weights generated from before tract2mni transformation be used to generate the connectome after transformation?
    - posted on MRTrix forum https://community.mrtrix.org/t/are-sift2-weights-still-interpretable-following-non-linear-transformation/6162
    - seems like they can be
* gen_connectome fails for sub-06
    - TODO: check if tract2mni has worked as expected
* TODO: plot dwi connectome using `nilearn.plotting.plot_connectome`

## Resting state functional connectome
* where are the confounds? Yasmin found some on EBRAINS release, use those?
* two sessions of rs-fmri, use both?
* ap and pa which one to use? or how are they combined for distortion correction?
* Partial correlations exceed -1.0

## Others
* functional connectivity obviously shows across hemisphere connections, which won't be the case for structural connectivity
    - do analysis for each hemisphere separately?
* TODO: nilearn PR for region coordinates in Schaefer 2018 atlas fetcher