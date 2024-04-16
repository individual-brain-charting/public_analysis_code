"""
Paired-t tests between Raiders and Clips across subjects

Author: Ana Luisa Pinho

Created: May 2020
Last update: August 2021

Compatibility: Python 3.9.1
"""

import os
import csv
import operator

import numpy as np
from scipy import stats

from nilearn.datasets import fetch_surf_fsaverage
from nilearn.surface import load_surf_data
from nilearn.plotting import plot_surf_stat_map, plot_surf_contours


# ############################## FUNCTIONS ####################################


def zval_conversion(tval, dof):
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


def threshold(z_vals, p_vals, alpha, height_control='fdr'):
    """Return the Benjamini-Hochberg FDR or Bonferroni threshold for
    the input correlations + corresponding p-values.
    """
    if alpha < 0 or alpha > 1:
        raise ValueError(
            'alpha should be between 0 and 1. {} was provided'.format(alpha))

    p_vals_ = np.sort(p_vals)
    idx = np.argsort(p_vals)

    z_vals_abs = np.abs(z_vals)
    z_vals_ = z_vals_abs[idx]

    n_samples = len(p_vals_)

    if height_control == 'fdr':
        pos = p_vals_ < alpha * np.linspace(1 / n_samples, 1, n_samples)
    elif height_control == 'bonferroni':
        pos = p_vals_ < alpha / n_samples
    else:
        raise ValueError('Height-control method not valid.')

    return (z_vals_[pos][-1] - 1.e-12) if pos.any() else np.infty


def plot_zmaps(meshes, smaps, hemispheres, backgrounds, fdr_threshold):
    # Path of figures
    zvals_fig_lhl = os.path.join(cache, 'surface_group_zvals_lh_lateral.png')
    zvals_fig_lhm = os.path.join(cache, 'surface_group_zvals_lh_medial.png')
    zvals_fig_rhl = os.path.join(cache, 'surface_group_zvals_rh_lateral.png')
    zvals_fig_rhm = os.path.join(cache, 'surface_group_zvals_rh_medial.png')

    # Plot results
    for mesh, smap, hemisphere, background in zip(meshes, smaps, hemispheres,
                                                  backgrounds):
        for bview in bviews:
            if hemisphere == 'left' and bview == 'lateral':
                output = zvals_fig_lhl
            elif hemisphere == 'left' and bview == 'medial':
                output = zvals_fig_lhm
            elif hemisphere == 'right' and bview == 'lateral':
                output = zvals_fig_rhl
            else:
                assert hemisphere == 'right' and bview == 'medial'
                output = zvals_fig_rhm
            plot_surf_stat_map(mesh, smap, hemi=hemisphere, view=bview,
                               colorbar=True, vmax=6., bg_map=background,
                               threshold=fdr_threshold,
                               output_file=output)


def glasser_proportions(zvalues_lh, zvalues_rh, fdr_threshold, header):
    # Threshold z-maps
    ztresh_lh = np.where(abs(zvalues_lh) >= fdr_threshold, zvals_lh, np.nan)
    ztresh_rh = np.where(abs(zvalues_rh) >= fdr_threshold, zvals_rh, np.nan)

    # Open dictionary with parcels, areas and regions id
    dict_parcels = {}
    with open('glasser_parcels2areas2regions.csv', mode='r') as infile:
        reader = csv.reader(infile)
        next(reader)  # skip the header
        dict_parcels = {row[0]: [row[1], row[2], row[3]] for row in reader}

    # Open dictionary with regions id and regions names
    dict_regions = {}
    with open('glasser_regions_names.csv', mode='r') as infile:
        reader = csv.reader(infile)
        next(reader)  # skip the header
        dict_regions = {row[0]: row[1] for row in reader}

    table = []
    for key, value in dict_parcels.items():
        # Get indices of vertices from the surface mesh for a given region
        parcel_lh = np.ravel(np.where(surf_lh == int(key)))
        parcel_rh = np.ravel(np.where(surf_rh == int(key)))

        # Take z-values belonging to the given region
        ztresh_region_lh = np.take(ztresh_lh, parcel_lh)
        ztresh_region_rh = np.take(ztresh_rh, parcel_rh)

        # Get number of significant vertices for the given region
        significant_lh = ztresh_region_lh[~np.isnan(ztresh_region_lh)].size
        significant_rh = ztresh_region_rh[~np.isnan(ztresh_region_rh)].size

        # Get total number of vertices for the given region
        n_ztresh_region_lh = ztresh_region_lh.size
        n_ztresh_region_rh = ztresh_region_rh.size

        # Compute proportion of significant vertices for the given region
        proportion_lh = (round(significant_lh/n_ztresh_region_lh * 100)) \
            if significant_lh != 0 else 0
        proportion_rh = (round(significant_rh/n_ztresh_region_rh * 100)) \
            if significant_rh != 0 else 0

        significant = significant_lh + significant_rh
        n_ztresh_region = n_ztresh_region_lh + n_ztresh_region_rh
        proportion = (round(significant / n_ztresh_region * 100)) \
            if significant != 0 else 0

        # # Max significant z-value for the given region
        # abs_zthresh_region_lh = np.abs(ztresh_region_lh)
        # abs_zthresh_region_rh = np.abs(ztresh_region_rh)

        # idx_zmax_lh = (np.nanargmax(abs_zthresh_region_lh)) \
        #     if ~np.all(np.isnan(abs_zthresh_region_lh)) else np.nan
        # idx_zmax_rh = (np.nanargmax(abs_zthresh_region_rh)) \
        #     if ~np.all(np.isnan(abs_zthresh_region_rh)) else np.nan

        # zmax_lh = (round(ztresh_region_lh[idx_zmax_lh])) \
        #     if idx_zmax_lh is not np.nan else np.nan
        # zmax_rh = (round(ztresh_region_rh[idx_zmax_rh])) \
        #     if idx_zmax_rh is not np.nan else np.nan

        # zmaxs = [zmax_lh, zmax_rh]
        # abs_zmaxs = [abs(zmax_lh), abs(zmax_rh)]
        # idx_zmax = np.nanargmax(abs_zmaxs) \
        #     if ~np.all(np.isnan(zmaxs)) else np.nan
        # zmax = zmaxs[idx_zmax] if idx_zmax is not np.nan else np.nan

        # Store results in a table
        table.append([dict_regions[value[2]], value[1], proportion_lh,
                      proportion_rh, proportion])

    # Sort by decreasing order of total area proportion within regions and
    # across regions
    dict_areas = {}
    dmax = {}
    for region in dict_regions.values():
        areas = []
        proportions_hem = []
        for entry in table:
            if entry[0] == region:
                areas.append(entry[1])
                proportions_hem.append(entry[4])
        idx_sorted = np.argsort(-np.array(proportions_hem))
        areas_sorted = [areas[j] for j in idx_sorted]
        dict_areas[region] = areas_sorted
        dmax[region] = np.max(proportions_hem)
    sorted_dmax = dict(sorted(dmax.items(), key=operator.itemgetter(1),
                              reverse=True))

    # Sort table
    table_sorted = []
    for region in sorted_dmax.keys():
        flag = 0
        for area in dict_areas[region]:
            for line in table:
                if line[0] == region and line[1] == area and line[4] >= 5.:
                    if flag:
                        line[0] = ''
                    else:
                        flag = 1
                    table_sorted.append(line[:-1])
        if flag == 0:
            print('This region is not part of the results: ', region)

    # Stack header of the table
    final_table = np.vstack((header, table_sorted))
    table_path = os.path.join(cache, 'glasser_zresults.csv')

    # Save table
    with open(table_path, 'w') as fp:
        a = csv.writer(fp)
        a.writerows(final_table)


# ############################### INPUTS ######################################

# Extract data from dataframe only referring to a pre-specified subset of
# participants
participants = [1, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15]

parent_dir = '/storage/store/work/agrilopi/fastsrm/encoding_analysis/'
main_dir = 'second_level/'
cache = parent_dir + main_dir

# Loads the high-resolution fsaverage mesh (163842 nodes)
fsaverage = fetch_surf_fsaverage(mesh='fsaverage')

# Glasser atlas
glasser_dir = parent_dir + 'glasser_atlas'
surf_lh = load_surf_data(os.path.join(glasser_dir, 'lh.HCPMMP1.annot'))
surf_rh = load_surf_data(os.path.join(glasser_dir, 'rh.HCPMMP1.annot'))

# ###################### Load contrasts #################################

clips_path = os.path.join(parent_dir, 'clips')
raiders_path = os.path.join(parent_dir, 'raiders')

# ################ Pearson Correlations ################

individual_clips_corr = np.load(os.path.join(
    clips_path, 'surface_individual_corr_clips.npy'))
individual_raiders_corr = np.load(os.path.join(
    raiders_path, 'surface_individual_corr_raiders.npy'))

# ################## Variables for plotting ##############################

meshes = [fsaverage.infl_left, fsaverage.infl_right]
hemispheres = ['left', 'right']
backgrounds = [fsaverage.sulc_left, fsaverage.sulc_right]
bviews = ['lateral', 'medial']

# ####################### Output files ###################################

HEADER = ['region', 'area', 'proportionlh', 'proportionrh']

# ################################ RUN ########################################

if __name__ == '__main__':

    # Compute paired-t-test to assess what are the regions that are
    # significantly different between Clips and Raiders
    tvals, pvals = stats.ttest_rel(individual_raiders_corr,
                                   individual_clips_corr,
                                   axis=0, nan_policy='raise',
                                   alternative='two-sided')

    # Compute z-values from t-values
    zvals = zval_conversion(tvals, individual_clips_corr.shape[0]-1)

    # Threshold z-values, ...
    # ... because pvalues are two-sided, alpha is corrected by a factor of 2
    fdr_thresh = threshold(zvals, pvals, 0.05/2, height_control='fdr')

    # Split results into the two hemispheres
    zvals_lh = np.split(zvals, 2, axis=0)[0]
    zvals_rh = np.split(zvals, 2, axis=0)[1]
    smaps = [zvals_lh, zvals_rh]

    # Save files
    np.save(os.path.join(cache, 'surface_group_zvals.npy'), zvals)
    np.save(os.path.join(cache, 'surface_group_zvals_lh.npy'), zvals_lh)
    np.save(os.path.join(cache, 'surface_group_zvals_rh.npy'), zvals_rh)

    # Plot z-maps
    plot_zmaps(meshes, smaps, hemispheres, backgrounds, fdr_thresh)

    # ### Get proportion of significant vertices per Glasser region ###
    glasser_proportions(zvals_lh, zvals_rh, fdr_thresh, HEADER)

    # Plot thresholded group-level results with Glasser Atlas contours

    # lh_lateral = plot_surf_stat_map(fsaverage.infl_left, zvals_lh,
    #                                 hemi='left', view='lateral',
    #                                 colorbar=True, vmax=6.,
    #                                 bg_map=fsaverage.sulc_left,
    #                                 threshold=fdr_thresh)

    # lh_medial = plot_surf_stat_map(fsaverage.infl_left, zvals_lh,
    #                                hemi='left', view='medial',
    #                                colorbar=True, vmax=6.,
    #                                bg_map=fsaverage.sulc_left,
    #                                threshold=fdr_thresh)

    # rh_lateral = plot_surf_stat_map(fsaverage.infl_right, zvals_rh,
    #                                 hemi='right', view='lateral',
    #                                 colorbar=True, vmax=6.,
    #                                 bg_map=fsaverage.sulc_right,
    #                                 threshold=fdr_thresh)

    # rh_medial = plot_surf_stat_map(fsaverage.infl_right, zvals_rh,
    #                                hemi='right', view='medial',
    #                                colorbar=True, vmax=6.,
    #                                bg_map=fsaverage.sulc_right,
    #                                threshold=fdr_thresh)

    # surf_lh = load_surf_data(os.path.join(glasser_dir, 'lh.HCPMMP1.annot'))
    # surf_rh = load_surf_data(os.path.join(glasser_dir, 'rh.HCPMMP1.annot'))

    # zvals_contour_lhl = os.path.join(glasser_dir, 'zvals_contour_lhl.png')
    # zvals_contour_lhm = os.path.join(glasser_dir, 'zvals_contour_lhm.png')
    # zvals_contour_rhl = os.path.join(glasser_dir, 'zvals_contour_rhl.png')
    # zvals_contour_rhm = os.path.join(glasser_dir, 'zvals_contour_rhm.png')

    # mydict = {}
    # with open('glasser_regions_labels.csv', mode='r') as infile:
    #     reader = csv.reader(infile)
    #     # skip header
    #     next(reader)
    #     mydict = {row[0]: row[1] for row in reader}

    # n = 9
    # regions_idx = np.arange(6*n + 1, 6*(n+1) + 1)
    # print(regions_idx)
    # glasser_labels = [mydict[str(x)] for x in regions_idx]
    # contour_colors = ['b', 'g', 'r', 'c', 'm', 'y']
    # plot_surf_contours(fsaverage.infl_left, surf_lh,
    #                    labels=glasser_labels,
    #                    levels=regions_idx,
    #                    figure=lh_lateral,
    #                    legend=True,
    #                    colors=contour_colors,
    #                    output_file=zvals_contour_lhl)

    # plot_surf_contours(fsaverage.infl_left, surf_lh,
    #                    labels=glasser_labels,
    #                    levels=regions_idx,
    #                    figure=lh_medial,
    #                    legend=True,
    #                    colors=contour_colors,
    #                    output_file=zvals_contour_lhm)

    # plot_surf_contours(fsaverage.infl_right, surf_rh,
    #                    labels=glasser_labels,
    #                    levels=regions_idx,
    #                    figure=rh_lateral,
    #                    legend=True,
    #                    colors=contour_colors,
    #                    output_file=zvals_contour_rhl)

    # plot_surf_contours(fsaverage.infl_right, surf_rh,
    #                    labels=glasser_labels,
    #                    levels=regions_idx,
    #                    figure=rh_medial,
    #                    legend=True,
    #                    colors=contour_colors,
    #                    output_file=zvals_contour_rhm)
