"""
This script transforms given parcellation to the subject native diffusion space 
and creates structural connectivity matrix between ROIs from the transformed 
atlas using diffusion tractography.
MRtrix3 wiki on transforming atlases: 
https://community.mrtrix.org/t/use-of-volumetric-templates-parcellation-atlases/2253
Connectivity between two ROIs here is measured as per-bundle sum of SIFT2
weights normalised by ROI volume.
Discussion on using original SIFT2 weights (calculated before non-linear 
transformation of tracts): https://community.mrtrix.org/t/are-sift2-weights-still-interpretable-following-non-linear-transformation/6162
"""
import os
from nilearn import datasets
from nilearn import plotting
import pandas as pd
import matplotlib.pyplot as plt

def atlas2dwi(dwi_b0, mni_nifti, atlas_nifti, atlas_in_dwi_space):
    get_warp_cmd = (f"ANTS 3 -m PR'[{mni_nifti},{dwi_b0}, 1, 2]'" 
                    f" -o ANTS -r Gauss'[2,0]' -t SyN'[0.5]' -i 30x99x11"
                    f" --use-Histogram-Matching")

    apply_warp_cmd = (f"WarpImageMultiTransform 3 {atlas_nifti}"
                      f"{atlas_in_dwi_space} -R {dwi_b0}"
                      f" -i ANTSAffine.txt ANTSInverseWarp.nii --use-NN")

    print(get_warp_cmd)
    os.system(get_warp_cmd)

    print(apply_warp_cmd)
    os.system(apply_warp_cmd)

def tck2connectome(atlas, tck, connectivity_matrix, inverse_connectivity_matrix,
                   sift_weights=None):
    if sift_weights == None:
        cmd = (f"tck2connectome -force -symmetric -zero_diagonal "
               f"-scale_invnodevol {tck} {atlas} {connectivity_matrix} "
               f"-out_assignments {inverse_connectivity_matrix}")
    else:
        cmd = (f"tck2connectome -force -symmetric -zero_diagonal "
               f"-scale_invnodevol -tck_weights_in {sift_weights} "
               f"{tck} {atlas} {connectivity_matrix} -out_assignments "
               f"{inverse_connectivity_matrix}")

    print(cmd)
    os.system(cmd)

if __name__ == "__main__":

    DATA_ROOT = '/data/parietal/store2/data/ibc/derivatives/'
    sub_ses = {'sub-01': 'ses-12', 'sub-04': 'ses-08', 'sub-05': 'ses-08',
               'sub-06': 'ses-09', 'sub-07': 'ses-09', 'sub-08': 'ses-09',
               'sub-09': 'ses-09', 'sub-11': 'ses-09', 'sub-12': 'ses-09',
               'sub-13': 'ses-09', 'sub-14': 'ses-05', 'sub-15': 'ses-12'}
    for sub, ses in sub_ses.items():

        # setup tmp dir for saving figures
        tmp_dir = os.path.join(DATA_ROOT, sub, ses, 'dwi', 'atlas2dwi')
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        # get atlas
        atlas = datasets.fetch_atlas_schaefer_2018(data_dir=tmp_dir, 
                                                   resolution_mm=1,
                                                   n_rois=400)
        # give atlas a custom name
        atlas['name'] = 'schaefer400'
        # get the skull-stripped mni-t2 template
        mni_t2 = os.path.join(DATA_ROOT, sub, ses, 'dwi', 'tract2mni_tmp',
                              'mni_t2w.nii.gz')
        os.system(f'cp {mni_t2} {tmp_dir}')
        mni_nifti = os.path.join(DATA_ROOT, sub, ses, 'dwi', 'atlas2dwi',
                                 'mni_t2w.nii.gz')
        # transform the atlas to dwi space
        b0dwi = os.path.join(DATA_ROOT, sub, ses, 'dwi',
                             (f'{sub}_{ses}_desc-denoise-eddy-correct-b0_dwi'
                              f'.nii.gz'))
        atlas_in_dwi_space = os.path.join(tmp_dir,
                             (f'{atlas.name}_in_dwi_space.nii.gz'))
        atlas2dwi(b0dwi, mni_nifti, atlas.maps, atlas_in_dwi_space)
        # input files for tck2connectome without sift weights
        dwi_tck = os.path.join(DATA_ROOT, sub, ses, 'dwi',
                               f'tracks_{sub}_{ses}_t1.tck')
        connectivity_matrix = os.path.join(tmp_dir,
                                           (f'{atlas.name}_connectome_'
                                            f'{sub}_{ses}_t2.csv'))
        inverse_connectivity_matrix = os.path.join(tmp_dir,
                                                   (f'{atlas.name}_inverse-'
                                                    f'connectome_{sub}_{ses}'
                                                    f'_t2.csv'))
        tck2connectome(atlas_in_dwi_space, dwi_tck, connectivity_matrix,
                       inverse_connectivity_matrix, sift_weights=None)
        # input files for tck2connectome with sift weights
        connectivity_matrix_sift = os.path.join(tmp_dir,
                                                (f'{atlas.name}_connectome_'
                                                 f'sift-weighted_{sub}_{ses}'
                                                 f'_t2.csv'))
        inverse_connectivity_matrix_sift = os.path.join(tmp_dir,
                                                        (f'{atlas.name}_inverse'
                                                         f'-connectome_sift-'
                                                         f'weighted_{sub}_{ses}'
                                                         f'_t2.csv'))
        sift_weights = os.path.join(DATA_ROOT, sub, ses, 'dwi',
                                    f'sift-track_{sub}_{ses}.txt')
        tck2connectome(atlas_in_dwi_space, dwi_tck, connectivity_matrix_sift,
                       inverse_connectivity_matrix_sift, sift_weights)
        # save connectivity matrix without sift weights as csv
        mat = pd.read_csv(connectivity_matrix, names=atlas.labels)
        # plot heatmaps and save figs                     
        mat_fig = plt.figure(figsize = (50,50))
        plotting.plot_matrix(mat, labels=atlas.labels, figure=mat_fig)
        connectome_fig = os.path.join(tmp_dir,
                                     f'{atlas.name}_connectome_{sub}_{ses}_t2.png')
        mat_fig.savefig(connectome_fig, bbox_inches='tight')
        # save connectivity matrix with sift weights as csv
        mat_sift = pd.read_csv(connectivity_matrix_sift, names=atlas.labels)
        # plot heatmaps and save figs
        mat_sift_fig = plt.figure(figsize = (50,50))
        plotting.plot_matrix(mat_sift, labels=atlas.labels, figure=mat_sift_fig)
        connectome_sift_fig = os.path.join(tmp_dir,
                                           (f'{atlas.name}_connectome_sift-'
                                            f'weighted_{sub}_{ses}_t2.png'))
        mat_sift_fig.savefig(connectome_sift_fig, bbox_inches='tight')
        plt.close('all')
