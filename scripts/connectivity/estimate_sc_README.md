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

# DWI connectome

## Steps
1. The streamlines obtained from tractography were first warped into MNI152 space using ANTs' image registration `antsRegistration` and MRtrix's `tcktransform` in script `estimate_sc.py`.
2. In addition, the script `estimate_sc.py` also transforms the given atlas to the native individual space. This way we can calculate two kinds of structural connectivity matrices: one in the MNI space and the other in the native individual space.
3. Finally, the two connectomes are calculated using MRtrix's `tck2connectome` function in the same script `estimate_sc.py`.
