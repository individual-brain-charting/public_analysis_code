""" this script launches the Quntitative T1 analysis tools on MRI
"""
import os
import shutil
import glob
from capsul.study_config import StudyConfig
from qmri.t1.pipeline import QT1Reconstruction
from qmri.toy_datasets import Enum
from qmri.t1.bafi_to_b1map import bafi_to_b1map
from qmri.t1.interpolation import resample_b1_afi_to_gre
from qmri.t1.despot1 import t1_from_despot1

TMP_DIR = '/tmp/qt1'

def make_ibc_dataset(subject):
    """
    The dataset is an Enum structure with some specific elements of
    interest:

        * **afiamplitude**: Nifti AFI amplitude image.
        * **afiamplitudedcm**: one Dicom AFI amplitude file.
        * **afiphase**: Nifti AFI phase image.
        * **afiphasedcm**: one Dicom AFI phase file.
        * **gre5**: Nifti GRE image with 5deg flip angle.
        * **gre5dcm**: one Dicom GRE with 5deg flip angle file.
        * **gre20**: Nifti GRE image with 20deg flip angle.
        * **gre20dcm**: one Dicom GRE with 20deg flip angle file.
        * **m0factor**: the AFI/GRE scale factor (if not one be carreful,
          you're probabely not using the approriate sequence).
        * **echonumbers**: a list with the echo orders (for instance
          [1, 2, 3, 4]).
    """
    scanner_path = '/neurospin/acquisition/database/TrioTim'
    if subject == 'sub001':
        data_dir = '/neurospin/tmp/ibc/main/sub001/sub001_02/anat'
        dcm_dir = os.path.join(scanner_path, '20150715/fl140183-4846_001')
        afiamplitude, afiphase, gre5, gre20 = 9, 11, 17, 21
        res_dir = '/neurospin/tmp/ibc/main/results/sub001_02'
    elif subject == 'sub002':
        data_dir = '/neurospin/tmp/ibc/main/sub002/sub002_01/anat'
        dcm_dir = os.path.join(scanner_path, '20150603/pg140140-4807_001')
        afiamplitude, afiphase, gre5, gre20 = 56, 58, 64, 68
        res_dir = '/neurospin/tmp/ibc/main/results/sub002_01'
    else:
        raise ValueError('Unknown subject')

    if not os.path.exists(res_dir):
        os.mkdir(res_dir)
    write_dir = os.path.join(res_dir, 'anat')

    dataset = Enum(
        afiamplitude = os.path.join(data_dir, 'b1-map_1_mag.nii'),
        afiamplitudedcm = glob.glob(os.path.join(dcm_dir, '*%06d*/*' % afiamplitude))[0], 
        afiphase = os.path.join(data_dir, 'b1-map_1_phase.nii'),
        afiphasedcm = glob.glob(os.path.join(dcm_dir, '*%06d*/*' % afiphase))[0], 
        gre5 = os.path.join(data_dir, 'afi_5d_mag.nii'),
        gre20 = os.path.join(data_dir, 'afi_20d_mag.nii'),
        gre5dcm = glob.glob(os.path.join(dcm_dir, '*%06d*/*' % gre5))[0],
        gre20dcm = glob.glob(os.path.join(dcm_dir, '*%06d*/*' % gre20))[0],
        m0factor = 1.9,
        echonumbers = [1, 2, 3, 4])
    return dataset, write_dir


def pilot_t1_pipeline(dataset, write_dir=TMP_DIR, method='analytic'):
    """
    Quantitative t1 map estimation
    ==============================

    Estimate a quantitive T1 map using the AFI (analytic) or VAFI
    (iterative) methods.
    The AFI method is fast but overestimate the real T1 values while the
    VAFI is slow but gives more realistic values.
    """
    # Study configuration
    study_config = StudyConfig(
        modules=["SmartCachingConfig"],
        use_smart_caching=True,
        output_directory=write_dir)

    # Processing definition
    pipeline = QT1Reconstruction()

    pipeline.afi_amplitude_image = dataset.afiamplitude
    pipeline.afi_amplitude_dicom = dataset.afiamplitudedcm
    pipeline.afi_phase_image = dataset.afiphase
    pipeline.afi_phase_dicom = dataset.afiphasedcm
    pipeline.gre_images = [dataset.gre5, dataset.gre20]
    pipeline.gre_dicoms = [dataset.gre5dcm, dataset.gre20dcm]
    pipeline.m0_factor = dataset.m0factor
    pipeline.echo_numbers = dataset.echonumbers

    pipeline.select_t1_reconstruction = method
    study_config.run(pipeline, executer_qc_nodes=True, verbose=1)
    print("\nOUTPUTS\n")
    for trait_name, trait_value in pipeline.get_outputs().items():
        print("{0}: {1}".format(trait_name, trait_value))


def pilot_t1_script(dataset, write_dir=TMP_DIR):
    """
    Quantitative t1 map estimation

        Estimate a quantitive T1 map using the AFI (analytic) or VAFI
        (iterative) methods.
        The AFI method is fast but overestimate the real T1 values while the
        VAFI is slow but gives more realistic values.
    """   
    # Processings
    b1_image, filled_b1_image = bafi_to_b1map(
        afi_amplitude_image=dataset.afiamplitude,
        afi_phase_image=dataset.afiphase,
        afi_amplitude_dicom=dataset.afiamplitudedcm,
        afi_phase_dicom=dataset.afiphasedcm,
        echo_numbers=dataset.echonumbers,
        output_directory=write_dir,
        b1_image_name="b1_map",
        filled_b1_image_prefix="filled")

    resampled_b1_image, resampled_afi_amplitude_image = resample_b1_afi_to_gre(
        b1_image=filled_b1_image,
        afi_amplitude_image=dataset.afiamplitude,
        gre_images=[dataset.gre5, dataset.gre20],
        output_directory=write_dir,
        resampled_image_prefix="resampled")

    (gre_fas, gre_trs, afi_fa, afi_tr, afi_tr_factor, t1_image,
     m0_image, fa_factor_image) = t1_from_despot1(
        b1_image=resampled_b1_image,
        gre_images=[dataset.gre5, dataset.gre20],
        gre_dicoms=[dataset.gre5dcm, dataset.gre20dcm],
        afi_amplitude_image=resampled_afi_amplitude_image,
        afi_amplitude_dicom=dataset.afiamplitudedcm,
        mask_image=None,
        m0_factor=dataset.m0factor,
        use_iterative_method=False,
        output_directory=write_dir,
        t1_image_name="t1_map",
        m0_image_name="m0_map",
        fa_factor_image_name="fa_factor_map")

    print("\nOUTPUTS\n")
    print("afi_fa: {0}".format(afi_fa))
    print("filled_b1_image: {0}".format(filled_b1_image))
    print("afi_tr: {0}".format(afi_tr))
    print("afi_tr_factor: {0}".format(afi_tr_factor))
    print("gre_fas: {0}".format(gre_fas))
    print("b1_image: {0}".format(b1_image))
    print("t1_image: {0}".format(t1_image))
    print("gre_trs: {0}".format(gre_trs))


def brain_mask(dataset, write_dir):
    """ Extract a brain mask using BET"""
    
    from nipype.interfaces import fsl
    btr = fsl.BET()
    btr.inputs.in_file = dataset.gre20
    btr.inputs.frac = 0.7
    btr.inputs.mask = True
    #btr.inputs.robust = True
    btr.inputs.out_file = os.path.join(write_dir, 'segmented_input.nii.gz')
    res = btr.run()
    print(res.outputs)


if __name__ == "__main__":
    for method in ['analytic', 'iterative']:
        for subject in ['sub001', 'sub002']:
            dataset, write_dir = make_ibc_dataset(subject)
            write_dir = os.path.join(write_dir, method)
            if not os.path.isdir(write_dir):
                os.makedirs(write_dir)

            brain_mask(dataset, write_dir)
            #pilot_t1_pipeline(dataset, write_dir, method=method)
            #pilot_t1_script(dataset, write_dir)
