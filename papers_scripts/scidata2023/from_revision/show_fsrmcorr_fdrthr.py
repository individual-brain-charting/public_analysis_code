"""
Get a FDR-correction threshold for the t-values of the OLS results and plot
the median correlation values on the surface of fsaverage for the Raiders or
Clips task.
"""
# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from nilearn.datasets import fetch_surf_fsaverage
from nilearn.plotting import plot_stat_map, plot_surf_stat_map
from nilearn.glm import fdr_threshold
from scipy import stats
import ibc_public.utils_data
from ibc_fastsrm_utils import flatten, reshape_preprocdata, stacker
# %%
# ############################# FUNCTIONS ###########################
def zval_conversion(tval, dof):
    """ Convert t-values from permuted_ols to z-values.
    Parameters
    ----------
    tval : array-like
        t-values from the OLS results
    dof : int
        Degrees of freedom as number of subjects minus 1
    Returns
    -------
    zval : array-like
    """
    pval = stats.t.sf(tval, dof)
    one_minus_pval = stats.t.cdf(tval, dof)
    zval_sf = stats.norm.isf(pval)
    zval_cdf = stats.norm.ppf(one_minus_pval)
    zval = np.empty(pval.shape)
    use_cdf = zval_sf < 0
    use_sf = np.logical_not(use_cdf)
    zval[np.atleast_1d(use_cdf)] = zval_cdf[use_cdf]
    zval[np.atleast_1d(use_sf)] = zval_sf[use_sf]

    return zval

def mask_data(median_corr, stats_data, threshold):
    """ Mask the median correlation values with the z-values threshold.
    Parameters
    ----------
    median_corr : array-like
        Median of the individual correlation values
    stats_data : array-like
        z-values from the OLS results
    threshold : float
        Threshold for the FDR correction
    Returns
    -------
    masked_data_lh : array-like
        Masked median correlation values for the left hemisphere
    masked_data_rh : array-like
        Masked median correlation values for the right hemisphere
    """
    # Create a mask where the zvalues are greater than the threshold
    mask = np.where(stats_data > threshold, 1, 0)
    masked_data = np.multiply(median_corr, mask)
    # Split the masked data into left and right hemispheres
    masked_data_lh = np.split(masked_data, 2, axis=0)[0]
    masked_data_rh = np.split(masked_data, 2, axis=0)[1]

    return masked_data_lh, masked_data_rh

def plot_slices(masked_data_lh, masked_data_rh, thr=0.01, cmap='viridis',
                clbar=False, dpi=300):
    """ Plot the masked median correlation values on the surface of fsaverage.
    Parameters
    ----------
    masked_data_lh : array-like
        Masked median correlation values for the left hemisphere
    masked_data_rh : array-like
        Masked median correlation values for the right hemisphere
    thr : float
        Threshold for the masked data, decided so the plot is not cluttered
        with super small values
    Returns
    -------
    ofs_clone : list
        List of the output files for the plots
    """
    
    ofs = []
    for hemi in ['left', 'right']:
        for view in ['lateral', 'medial']:
            fig = os.path.join(mem, "mediancorr_correctedmask_surf_" +
                            f"{hemi[0]}h{view[0]}_{TASKS[0]}.svg")
            fig_png = os.path.join(mem, "mediancorr_correctedmask_surf_" +
                                    f"{hemi[0]}h{view[0]}_{TASKS[0]}.png")
            if hemi == 'left':
                mesh = fsaverage.infl_left
                bg_map = fsaverage.sulc_left
                stat = masked_data_lh
            else:
                mesh = fsaverage.infl_right
                bg_map = fsaverage.sulc_right
                stat = masked_data_rh
            plot_surf_stat_map(mesh, stat, bg_map=bg_map, hemi=hemi,
                               view=view, colorbar=clbar, threshold=thr,
                               vmax=0.6, vmin=0, output_file=fig, cmap=cmap)
            plot_surf_stat_map(mesh, stat, bg_map=bg_map, hemi=hemi,
                               view=view, colorbar=clbar, threshold=thr,
                               vmax=0.6, vmin=0, output_file=fig_png,
                               cmap=cmap, dpi=dpi,)
            ofs.append(fig_png)

    ofs_clone = []
    ofs_clone = ofs[:2]
    ofs_clone.append(ofs[-1])
    ofs_clone.append(ofs[2])
    #print(ofs_clone)

    return ofs_clone
# %%
# ############################# INPUTS ##############################

alt_parent_dir = '/storage/store/work/agrilopi'
fastsrm_dir = 'fastsrm'
main_dir = 'encoding_analysis'
data_dir = os.path.join(alt_parent_dir, fastsrm_dir, main_dir)
mem = '/storage/store3/work/aponcema/IBC_paper3/cache_two'

# %%
#TASKS = ['Clips'] 
TASKS = ['Raiders']
fsaverage = fetch_surf_fsaverage(mesh='fsaverage')
if TASKS == ['Clips']:
    suffix = 'clips'
else:
    suffix = 'raiders'

corr_results = np.load(os.path.join(data_dir, suffix,
                                    'surface_corr' + '_' +
                                    suffix + '.npy')) #(12, 2, 327684)
indiv_corr = np.load(os.path.join(data_dir, suffix,
                                  'surface_individual_corr' + '_' +
                                  suffix + '.npy')) #(12, 327684)
ols_results = np.load(os.path.join(data_dir, suffix,
                                    'surface_group_corr' + '_' +
                                    suffix + '.npz')) #t, logp_max_t, h0_max_t
# %%
# # ############################# MAIN ##############################

if __name__ == '__main__':
    
    # Degrees of freedom as number of subjects minus 1
    dof = indiv_corr.shape[0]-1
    # get the t-values from the OLS results
    tvals = np.reshape(ols_results['t'],(ols_results['t'].shape[1],)) #(327684,)
    # Compute z-values from t-values
    zvals = zval_conversion(tvals, dof) #(327684,)
    # Compute threshold for a given FDR correction
    threshold = fdr_threshold(zvals, alpha=0.05,)

    # Compute the median of the individual correlations
    median_corr = np.median(indiv_corr, axis=0)
    # Mask the median correlation values with the z-values threshold
    masked_data_lh, masked_data_rh = mask_data(median_corr, zvals, threshold)
    # Get side plots of the masked median correlation on surface
    side_figs = plot_slices(masked_data_lh, masked_data_rh, thr=0.1)

    # %%
    # Get the 4 views and plot them in a single figure
    plt.figure(figsize=(8, 1.6))
    for q, output in enumerate(side_figs):
        if q == 0:
            #ax = plt.axes([0.235 * q -.005, 0, .30, 1.])
            ax = plt.axes([0.225 * q -.005, 0, .23, 1.])
            ax.imshow(mpimg.imread(output)[70:-85, 40:-10])

        elif q == 1:
            ax = plt.axes([0.227, 0, .23, 1.])
            ax.imshow(mpimg.imread(output)[70:-85, 40:-10])
        elif q == 2:
            ax = plt.axes([0.225 * q + 0.012, 0, .23, 1.])
            ax.imshow(mpimg.imread(output)[70:-85, 40:-10])
        else:
            ax = plt.axes([0.70, 0, .23, 1.])
            ax.imshow(mpimg.imread(output)[70:-85, 40:-10])
        ax.axis('off')
    plt.suptitle(f'{TASKS[0]}', fontsize=14, y=1.05)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0)
    save_fig = True
    if save_fig:
        plt.savefig(os.path.join(mem, f'mediancorr_fsrm_{TASKS[0]}_fdr.svg'),
                    bbox_inches='tight')
        plt.savefig(os.path.join(mem, f'mediancorr_fsrm_{TASKS[0]}_fdr.png'),
                    dpi=300,bbox_inches='tight')
        plt.savefig(os.path.join(mem, f'mediancorr_fsrm_{TASKS[0]}_fdr.pdf'),
                    dpi=300,bbox_inches='tight')
    else:
        plt.show()
    # %%