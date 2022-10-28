import os
from nilearn import datasets
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def tck2connectome(atlas, tck, connectivity_matrix, inverse_connectivity_matrix, sift_weights=None):
    if sift_weights == None:
        cmd = f"tck2connectome -force -symmetric -zero_diagonal -scale_invnodevol {tck} {atlas} {connectivity_matrix} -out_assignments {inverse_connectivity_matrix}"
    else:
        cmd = f"tck2connectome -force -symmetric -zero_diagonal -scale_invnodevol -tck_weights_in {sift_weights} {tck} {atlas} {connectivity_matrix} -out_assignments {inverse_connectivity_matrix}"


if __name__ == "__main__":

    DATA_ROOT = '/data/parietal/store2/data/ibc/derivatives/'

    sub_ses = {'sub-01': 'ses-12', 'sub-04': 'ses-08', 'sub-05': 'ses-08',
               'sub-06': 'ses-09', 'sub-07': 'ses-09', 'sub-08': 'ses-09',
               'sub-09': 'ses-09', 'sub-11': 'ses-09', 'sub-12': 'ses-09',
               'sub-13': 'ses-09', 'sub-14': 'ses-05', 'sub-15': 'ses-12'}

    for sub, ses in sub_ses.items():

        tmp_dir = os.path.join(DATA_ROOT, sub, ses, 'dwi', 'tract2mni_tmp')

        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)

        schaefer400 = datasets.fetch_atlas_schaefer_2018(data_dir=tmp_dir, resolution_mm=1, n_rois=400)

        mni_tck = os.path.join(DATA_ROOT, sub, ses, 'dwi', f'mni-tracks_{sub}_{ses}_t1.tck')
        connectivity_matrix = os.path.join(DATA_ROOT, sub, ses, 'dwi', f'schaefer400_connectome_{sub}_{ses}.csv')
        inverse_connectivity_matrix = os.path.join(DATA_ROOT, sub, ses, 'dwi', f'schaefer400_inverse-connectome_{sub}_{ses}.csv')

        tck2connectome(schaefer400.maps, mni_tck, connectivity_matrix, inverse_connectivity_matrix, sift_weights=None)

        connectivity_matrix_sift = os.path.join(DATA_ROOT, sub, ses, 'dwi', f'schaefer400_connectome_sift-weighted_{sub}_{ses}.csv')
        inverse_connectivity_matrix_sift = os.path.join(DATA_ROOT, sub, ses, 'dwi', f'schaefer400_inverse-connectome_sift-weighted_{sub}_{ses}.csv')
        sift_weights = os.path.join(DATA_ROOT, sub, ses, 'dwi', f'sift-track_{sub}_{ses}.txt')

        tck2connectome(schaefer400.maps, mni_tck, connectivity_matrix, inverse_connectivity_matrix, sift_weights)
        plt.figure(figsize = (50,50))
        
        mat = pd.read_csv(connectivity_matrix, names=schaefer400.labels)
        sns.heatmap(mat, xticklabels=1).get_figure().savefig(os.path.join(DATA_ROOT, f'schaefer400_connectome_{sub}_{ses}.pdf'), bbox_inches='tight')

        mat_sift = pd.read_csv(connectivity_matrix_sift, names=schaefer400.labels)
        sns.heatmap(mat_sift, xticklabels=1).get_figure().savefig(os.path.join(DATA_ROOT, f'schaefer400_connectome_sift-weighted_{sub}_{ses}.pdf'), bbox_inches='tight')
