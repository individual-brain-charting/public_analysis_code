## Get it working:

* Install some of the dependencies as follows:
     
        pip3 install nilearn nibabel matplotlib numpy joblib dicom progressbar Cython

* Install `qmri` from bioproj (requires CEA login) as follows:

        git clone -b ibc_changes https://bioproj.extra.cea.fr/git/qmri
        cd qmri
        python3 setup.py build_ext
        python3 setup.py install

* Install `pypreprocess` by following the instructions given [here](https://github.com/neurospin/pypreprocess)

* After installing all the dependencies, to run, simply do:
    
        python3 qmri_run_estimation.py

## qMRI preprocessing and estimation pipeline:

* `qmri_run_estimation.py` runs 4 variations of preprocessing and qmri-estimation pipelines on all 12 IBC subjects for which the data is available:
    
    * T2 estimation on qmri images in subject-space
    * T2 estimation on qmri images in MNI-space
    * T1 estimation on qmri images in subject-space
    * T1 estimation on qmri images in MNI-space

* `qmri_run_estimation.py` imports from `ibc_public/util_relaxo.py` containing preprocessing steps for qmri images that happen before estimation, run the estimation itself (explained in next point), and plot the final estimated maps:

    * all preprocessing is done using `pypreprocess`

    * function `t1_pipeline`:

        1. checks whether sourcedata is defaced (by looking for "Deface: True" field in .json sidecars) - if not, defaces it using `pydeface`
        2. copies and extracts t1 images from sourcedata in a tmp directory
        3. checks whether to return maps in MNI or subject-space:
            * if `do_normalise_before` is set to True, segments highest flip angle image and uses the segments to run spatial normalisation and transforms the raw images to MNI space and then computes mask in MNI space
            * if `do_normalise_before` is set to False, segments highest flip angle image and uses the segments to compute a mask in subject-space to remove the skull
        4. extracts relevant acquisition parameters from .json sidecar (by running `scripts/qmri_t1_map_b1_params.py`)
        5. runs qmri t1 estimation pipeline (`scripts/qmri_t1_map_b1.py`)
        6. thresholds voxel intensity at 99 percentile and plots the estimated maps.

    * function `t2_pipeline`:

        1. checks whether sourcedata is defaced (by looking for "Deface: True" field in .json sidecars) - if not, defaces it using `pydeface`
        2. copies and extracts t2 images from sourcedata in a tmp directory
        3. since t2 maps are low resolution, also copies highest flip-angle t1 image for better masking
        3. checks whether to return maps in MNI or subject-space:
            * if `do_normalise_before` is set to True, segments the t1 image and uses the segments to run spatial normalisation to transform the raw t1 and t2 images to MNI space and then use the high res t1 image to compute mask in MNI-space
            * if `do_normalise_before` is set to False, corregisters the t1 image to t2 image and uses the high res corregistered t1 image to compute a mask in subject-space
        4. runs qmri t2 estimation pipeline (`scripts/t2_map.py`)
        6. thresholds voxel intensity at 95 percentile and plots the estimated maps.

* Package [`qmri`](https://bioproj.extra.cea.fr/git/qmri), hosted on bioproj, is used for T1 and T2 map estimation.

    * `scripts/qmri_t1_map_b1.py` (for T1 estimation) and `scripts/qmri_t2_map.py` (for T2 estimation) are the scripts containing estimation pipelines using `qmri`.

## Author:

Himanshu Aggarwal 
himanshu.aggarwal@inria.fr
2021-22