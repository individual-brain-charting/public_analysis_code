"""
This module contains pipelines and functions 
for generating qMRI t1 and t2-maps.

Author: Himanshu Aggarwal (himanshu.aggarwal@inria.fr), 2021-22
"""

import warnings
warnings.filterwarnings("ignore")

from os import (listdir, system, makedirs, sep)
from os.path import join, exists
import time
import shutil
from pypreprocess.nipype_preproc_spm_utils import (_do_subject_coregister,
                    _do_subject_normalize,_do_subject_segment, SubjectData)

from nilearn.masking import (compute_brain_mask, compute_background_mask,
                        compute_epi_mask,intersect_masks)
from nilearn.image import (math_img, resample_to_img, mean_img)
from nilearn import plotting as nilplot
from matplotlib.pyplot import (hist, savefig, title, close)
from matplotlib import cm
from nibabel import load
from numpy import (nanpercentile, nanmin)
from scipy import ndimage
from nilearn.image.image import new_img_like


def closing(image, iterations):
    image_data = image.get_fdata()
    image_affine = image.affine
    closing_test = ndimage.binary_closing(image_data, iterations=iterations).astype(int)

    closing_test = new_img_like(image, closing_test, image_affine)

    return closing_test

def to_T2space(t2_img, t1_img, output_dir):
    data = SubjectData()
    data.anat = t1_img
    data.func = [t2_img]
    data.output_dir = output_dir
    coreged = _do_subject_coregister(data,
                                    caching=False,
                                    hardlink_output=False,
                                    coreg_anat_to_func=True)
    return coreged

def to_MNI(image, data = SubjectData(), func=None):
    data.anat = image
    data.func = func
    data.output_dir = '.'

    normalized = _do_subject_normalize(data,
                                    caching=False,
                                    hardlink_output=False,
                                    func_write_voxel_sizes=[1, 1, 1],
                                    anat_write_voxel_sizes=[1, 1, 1])
    return normalized

def segment(image, normalize):
    data = SubjectData()
    data.anat = image
    segmented = _do_subject_segment(data, caching=False,
                    normalize=normalize, hardlink_output=False)
    return segmented

def plot_thresholded_qmap(img, coords, output_folder, brain=None, mask=None,
                        thresh=99, map="map", interactive=False):

    # flatten image data for calculating threshold
    img_arr = load(img).get_data().astype(float)
    img_arr_flat = img_arr.reshape((-1, img_arr.shape[-1])).flatten()

    # t1 map has negatives - keeping only positive values
    if nanmin(img_arr_flat) < 0:
        # too many zeros result in threshold at 0 - even at 99%
        # calculating threshold after removing -ves and 0s
        img_arr_flat = img_arr_flat[img_arr_flat > 0]

    # calculating threshold
    # if given threshold is specified as percentile
    if type(thresh) is str:
        threshold = nanpercentile(img_arr_flat, float(thresh))
        titl = 'Voxel distribution ranged between [0, {}] (at {} percentile)'.format(threshold, thresh)
    # if given threshold is specified as voxel value
    else:
        threshold = float(thresh)
        titl = 'Voxel distribution ranged between [0, {}]'.format(thresh)
    # removing voxels higher than threshold
    img_arr_flat_threshold = img_arr_flat[img_arr_flat <= threshold]
    print("{} - voxels ranging [0, {}] plotted".format(map, threshold, thresh))

    # plot distribution w/o thresholding in blue
    # hist(img_arr_flat)
    # show()

    # plot voxel distribution upto threshold
    hist(img_arr_flat_threshold)
    title(titl)
    fig_name = join(output_folder, '{}_vox_dist.pdf'.format(map))
    savefig(fig_name, bbox_inches='tight')
    close()

    # plot interactive thresholded t2 map image
    # this is buggy
    if interactive:
        html_view = nilplot.view_img(img,
                                    brain,
                                    cmap=cm.gray,
                                    symmetric_cmap=False,
                                    # vmin=0,
                                    vmax=threshold,
                                    threshold=0)
        fig_name = join(output_folder, '{}_interactive.html'.format(map))
        html_view.save_as_html(fig_name)
        
    # plot a simple thresholded image
    normal_view = nilplot.plot_img(img=img,
                                bg_img=brain,
                                cut_coords=coords,
                                cmap=cm.gray,
                                vmax=threshold,
                                vmin=0,
                                colorbar=True)
    fig_name = join(output_folder, '{}_plot.pdf'.format(map))
    normal_view.savefig(fig_name)
    normal_view.close()

# only one of do_normalise_before and do_normalise_after should be True
# both can be False
def t2_pipeline(do_coreg=True, do_normalise_before=False,
                do_segment=True, do_normalise_after=False, do_plot=True,
                keep_tmp=True, sub_name='sub-11', sess_num='ses-17',
                root_path='/neurospin/ibc'):

    DATA_LOC = join(root_path, 'sourcedata', sub_name, sess_num)
    SAVE_TO = join(root_path, 'derivatives', sub_name, sess_num)

    if do_normalise_before or do_normalise_after:
        space = "MNI152"
    else:
        space = "individual"

    if not exists(SAVE_TO):
        makedirs(SAVE_TO)

    start_time = time.time()

    # data files
    data_dir = DATA_LOC

    niftis = []
    jsons = []
    for fi in listdir(join(data_dir, 'anat')):
        if fi.split('_')[-1] == 'T2map.nii.gz':
            niftis.append(join(data_dir, 'anat', fi))
        elif fi.split('_')[-1] == 'T2map.json':
            jsons.append(join(data_dir, 'anat', fi))
        else:
            continue
    niftis.sort()
    jsons.sort()

    run_count = 0
    for nifti in niftis:
        # preprocessing directory setup
        time_elapsed = time.time() - start_time
        print('[INFO,  t={:.2f}s] copying necessary files...'.format(time_elapsed))
        cwd = SAVE_TO
        preproc_dir = join(cwd, 'tmp_t2', 'preproc')
        if not exists(preproc_dir):
            makedirs(preproc_dir)

        system('cp {} {}'.format(nifti, preproc_dir))
        nifti = join(preproc_dir, nifti.split(sep)[-1])
        system('gunzip -df {}'.format(nifti))
        nifti = join(preproc_dir, nifti.split(sep)[-1].split('.')[0] + '.nii')

        if do_coreg:
            # t1 image as an anatomical image
            t1_niftis = []
            for fi in listdir(join(data_dir, 'anat')):
                if fi.split('_')[-1] == 'T1map.nii.gz':
                    t1_niftis.append(join(data_dir, 'anat', fi))
            t1_niftis.sort()
            t1_nifti = t1_niftis[-1]
            system('cp {} {}'.format(t1_nifti, preproc_dir))
            t1_nifti = join(preproc_dir, t1_nifti.split(sep)[-1])
            system('gunzip -df {}'.format(t1_nifti))
            t1_nifti = join(preproc_dir, t1_nifti.split(sep)[-1].split('.')[0] + '.nii')

        # preprocessing step: spatial normalization to MNI space
        # of T1 maps
        if do_normalise_before:
            image = nifti
            if do_coreg:
                t1_img = t1_nifti
                time_elapsed = time.time() - start_time
                print('[INFO,  t={:.2f}s] segmenting the t1 image'.format(time_elapsed))
                out_info = segment(t1_img, False)
                time_elapsed = time.time() - start_time
                print('[INFO,  t={:.2f}s] transforming images to MNI space...'.format(time_elapsed))
                out_info = to_MNI(image=t1_img, data=out_info, func=image)
                normed_t1_img = out_info['anat']
                normed_nifti = out_info['func'][0]
            else:
                time_elapsed = time.time() - start_time
                print('[INFO,  t={:.2f}s] segmenting the t2 image'.format(time_elapsed))
                out_info = segment(image, False)
                time_elapsed = time.time() - start_time
                print('[INFO,  t={:.2f}s] transforming images to MNI space...'.format(time_elapsed))
                out_info = to_MNI(image, data=out_info)
                normed_nifti = out_info['anat']
            time_elapsed = time.time() - start_time
            print('[INFO,  t={:.2f}s] \t transformed {}'.format(time_elapsed, nifti.split(sep)[-1]))

        # preprocessing step: transform t1 image to t2 space
        if do_coreg:
            time_elapsed = time.time() - start_time
            print('[INFO,  t={:.2f}s] transforming t1 image to t2 space...'.format(time_elapsed))
            if do_normalise_before:
                t2_img = normed_nifti
                t1_img = normed_t1_img
            else:
                t2_img = nifti
                t1_img = t1_nifti
            mean_t2 = mean_img(t2_img)
            mean_t2_img = join(preproc_dir, 'mean_{}'.format(t2_img.split(sep)[-1]))
            mean_t2.to_filename(mean_t2_img)
            out_info = to_T2space(t2_img=mean_t2_img, t1_img=t1_img, output_dir=preproc_dir)
            print('[INFO,  t={:.2f}s] \t transformed {} to {} space'.format(time_elapsed,
                    t1_img.split(sep)[-1], mean_t2_img.split(sep)[-1]))


        # preprocessing step: segmenting largest flip angle image
        if do_segment:
            if do_normalise_before:
                time_elapsed = time.time() - start_time
                print('[INFO,  t={:.2f}s] creating a mask'.format(time_elapsed))
                if do_coreg:
                    image = normed_t1_img
                else:
                    image = normed_nifti
                mni = compute_brain_mask(image)
                closed_mni = closing(mni, 12)
                union = intersect_masks([mni, closed_mni], threshold=0)
            else:
                if do_coreg:
                    image = t1_img
                else:
                    image = nifti
                time_elapsed = time.time() - start_time
                print('[INFO,  t={:.2f}s] segmenting the image for creating a mask'.format(time_elapsed))
                out_info = segment(image, False)
                segments = [out_info['gm'], out_info['wm']]
                time_elapsed = time.time() - start_time
                print('[INFO,  t={:.2f}s] \t segmented {}'.format(time_elapsed, image.split(sep)[-1]))

                # preprocessing step: creating a mask
                time_elapsed = time.time() - start_time
                print('[INFO,  t={:.2f}s] creating a mask using segments'.format(time_elapsed))
                add = math_img("img1 + img2", img1=segments[0], img2=segments[1])
                if sub_name=='sub-08':
                    full = compute_epi_mask(add, exclude_zeros=True)
                else:
                    full = compute_epi_mask(add)
                insides = compute_background_mask(full, opening=12)
                union = intersect_masks([full, insides], threshold=0)

            mask_file = join(preproc_dir,'mask.nii')
            union.to_filename(mask_file)

            if do_coreg:
                resampled_mask = resample_to_img(mask_file, mean_t2_img, clip=True)
                rounded_resampled_mask = math_img('np.around(img1)', img1=resampled_mask)
                resampled_mask_img = join(preproc_dir, 'resampled_mask.nii')
                rounded_resampled_mask.to_filename(resampled_mask_img)
            else:
                resampled_mask_img = mask_file


        # estimation directory setup
        time_elapsed = time.time() - start_time
        print('[INFO,  t={:.2f}s] starting estimation...'.format(time_elapsed))
        recon_dir = join(cwd, 'tmp_t2', 'recon')
        if not exists(recon_dir):
            makedirs(recon_dir)

        if do_normalise_before:
            niftis_str = normed_nifti
        else:
            niftis_str = nifti

        if do_segment == False:
            mask_file = None


        # estimation: t2 estimation
        system(f"python3 ../scripts/qmri_t2_map.py\
            -v 1\
            -s {sub_name}\
            -o {recon_dir}\
            -n {niftis_str}\
            -m {resampled_mask_img}")

        recon_map = join(recon_dir, f'{sub_name}_T2map.nii.gz')

        # postprocessing: normalization of reconstructed t1 map
        if do_normalise_after:
            postproc_dir = join(cwd, 'tmp_t2', 'postproc')
            if not exists(postproc_dir):
                makedirs(postproc_dir)
            
            system('cp {} {}'.format(recon_map, postproc_dir))
            time_elapsed = time.time() - start_time
            print('[INFO,  t={:.2f}s] normalizing reconstructed map...'.format(time_elapsed))
            image = join(postproc_dir, f'{sub_name}_T2map.nii.gz')
            system('gunzip -df {}'.format(image))
            image = join(postproc_dir, f'{sub_name}_T2map.nii')
            out_info = to_MNI(image, segmented=out_info.nipype_results['segment'])
            norm_recon_map = out_info['anat']


        # doing the plots
        if do_plot:
            time_elapsed = time.time() - start_time
            print('\n[INFO,  t={:.2f}s] plotting the map...'.format(time_elapsed))
            plot_dir = join(cwd, 'tmp_t2', 'plot')
            if not exists(plot_dir):
                makedirs(plot_dir)
            if do_normalise_after:
                t2_img = norm_recon_map
            else:
                t2_img = recon_map
            plot_thresholded_qmap(img=t2_img,
                                thresh="95",
                                map=f"{sub_name}_T2map",
                                interactive=True,
                                coords=(10,56,43),
                                output_folder=plot_dir)

        time_elapsed = time.time() - start_time
        print('\n[INFO,  t={:.2f}s] DONE'.format(time_elapsed))


        # move derived files out and delete tmp_t2 directory
        final_recon_map = join(SAVE_TO, f'{sub_name}_run-0{run_count+1}_space-{space}_T2map.nii.gz')
        if do_normalise_after:
            system('gzip {}'.format(norm_recon_map))
            recon_map = norm_recon_map + '.gz'
        shutil.move(recon_map, final_recon_map)

        if do_plot:
            for fi in listdir(plot_dir):
                plot_name = join(plot_dir, fi)
                final_plot_name = join(SAVE_TO, final_recon_map.split(sep)[-1].split('.')[0]
                                        + '_' + fi.split('_')[-1])
                shutil.move(plot_name, final_plot_name)

        if not keep_tmp:
            shutil.rmtree(join(SAVE_TO, 'tmp_t2'))

        time_elapsed = time.time() - start_time
        print('\n[INFO,  t={:.2f}s] created {} \n\n'.format(time_elapsed, final_recon_map))
        run_count = run_count + 1



# only one of do_normalise_before and do_normalise_after should be True
# both can be False
def t1_pipeline(do_normalise_before=False,
                do_segment=True, do_normalise_after=False,
                do_plot=True,  keep_tmp=True ,
                sub_name='sub-11', sess_num='ses-17', 
                root_path='/neurospin/ibc'):

    DATA_LOC = join(root_path, 'sourcedata', sub_name, sess_num)
    SAVE_TO = join(root_path, 'derivatives', sub_name, sess_num)

    if do_normalise_before or do_normalise_after:
        space = "MNI152"
    else:
        space = "individual"

    if not exists(SAVE_TO):
        makedirs(SAVE_TO)

    start_time = time.time()

    # data files
    data_dir = DATA_LOC

    niftis = []
    jsons = []
    for fi in listdir(join(data_dir, 'anat')):
        if fi.split('_')[-1] == 'T1map.nii.gz':
            niftis.append(join(data_dir, 'anat', fi))
        elif fi.split('_')[-1] == 'T1map.json':
            jsons.append(join(data_dir, 'anat', fi))
        elif fi.split('_')[-1] == 'B1map.nii.gz':
            b1_map_nifti = join(data_dir, 'anat', fi)
        elif fi.split('_')[-1] == 'B1map.json':
            b1_map_json = join(data_dir, 'anat', fi)
        else:
            continue
    niftis.sort()
    jsons.sort()


    # preprocessing directory setup
    time_elapsed = time.time() - start_time
    print('[INFO,  t={:.2f}s] copying necessary files...'.format(time_elapsed))
    cwd = SAVE_TO
    preproc_dir = join(cwd, 'tmp_t1', 'preproc')
    if not exists(preproc_dir):
        makedirs(preproc_dir)

    # copy data files to tmp_t1 directory
    cnt = 0
    for nii, json in zip(niftis, jsons):
        system('cp {} {}'.format(nii, preproc_dir))
        niftis[cnt] = join(preproc_dir, nii.split(sep)[-1])
        system('gunzip -df {}'.format(niftis[cnt]))
        niftis[cnt] = join(preproc_dir, nii.split(sep)[-1].split('.')[0] + '.nii')
        system('cp {} {}'.format(json, preproc_dir))
        jsons[cnt] = join(preproc_dir, json.split(sep)[-1])
        cnt += 1
    system('cp {} {}'.format(b1_map_nifti, preproc_dir))
    b1_map_nifti = join(preproc_dir, b1_map_nifti.split(sep)[-1])
    system('gunzip -df {}'.format(b1_map_nifti))
    b1_map_nifti = join(preproc_dir, b1_map_nifti.split(sep)[-1].split('.')[0] + '.nii')
    system('cp {} {}'.format(b1_map_json, preproc_dir))
    b1_map_json = join(preproc_dir, b1_map_json.split(sep)[-1])


    # preprocessing step: spatial normalization to MNI space
    # of T1 maps
    if do_normalise_before:
        time_elapsed = time.time() - start_time
        print('[INFO,  t={:.2f}s] segmenting highest flip angle image'.format(time_elapsed))
        image = niftis[-1]
        out_info = segment(image, True)
        # save normalised segments
        segments = [join(preproc_dir, f"w{out_info[segment].split('/')[-1]}") for segment in ['gm', 'wm']]
        time_elapsed = time.time() - start_time
        print('[INFO,  t={:.2f}s] transforming images to MNI space...'.format(time_elapsed))
        normed_niftis = []
        cnt = 0
        for nii in niftis:
            image = nii
            if cnt == len(niftis)-1:
                b1_map = b1_map_nifti
                out_info = to_MNI(image, data=out_info, func=b1_map)
                normed_b1_map = out_info['func'][0]
            else:
                out_info = to_MNI(image, data=out_info)
            normed_niftis.append(out_info['anat'])
            time_elapsed = time.time() - start_time
            print('[INFO,  t={:.2f}s] \t transformed {}'.format(time_elapsed, nii.split(sep)[-1]))
            cnt = cnt + 1

    # preprocessing step: segmenting largest flip angle image
    if do_segment:
        time_elapsed = time.time() - start_time
        if do_normalise_before:
            print('[INFO,  t={:.2f}s] creating a mask'.format(time_elapsed))
            image = normed_niftis[-1]
            mni = compute_brain_mask(image)
            closed_mni = closing(mni, 12)
            union = intersect_masks([mni, closed_mni], threshold=0)
        else:
            print('[INFO,  t={:.2f}s] segmenting highest flip angle image'.format(time_elapsed))
            image = niftis[-1]
            out_info = segment(image, True)
            segments = [out_info['gm'], out_info['wm']]
            time_elapsed = time.time() - start_time
            print('[INFO,  t={:.2f}s] \t segmented {}'.format(time_elapsed, image.split(sep)[-1]))

            # preprocessing step: creating a mask
            print('[INFO,  t={:.2f}s] creating a mask using segments'.format(time_elapsed))
            add = math_img("img1 + img2", img1=segments[0], img2=segments[1])
            if sub_name=='sub-08':
                full = compute_epi_mask(add, exclude_zeros=True)
            else:
                full = compute_epi_mask(add)
            insides = compute_background_mask(full, opening=12)
            union = intersect_masks([full, insides], threshold=0)

        mask_file = join(preproc_dir, 'mask.nii')
        union.to_filename(mask_file)


    # estimation directory setup
    time_elapsed = time.time() - start_time
    print('[INFO,  t={:.2f}s] starting estimation...'.format(time_elapsed))
    recon_dir = join(cwd, 'tmp_t1', 'recon')
    if not exists(recon_dir):
        makedirs(recon_dir)
    jsons_str = ' '.join(jsons)

    if do_normalise_before:
        niftis_str = ' '.join(normed_niftis)
        b1_map_str = normed_b1_map
    else:
        niftis_str = ' '.join(niftis)
        b1_map_str = b1_map_nifti

    if do_segment == False:
        mask_file = None

    # estimation: parameter extraction
    system(f"python3 ../scripts/qmri_t1_map_b1_params.py\
        -v 1\
        -s {sub_name}\
        -o {recon_dir}\
        -g {jsons_str}\
        -b {b1_map_json}")

    # estimation: t1 estimation
    system(f"python3 ../scripts/qmri_t1_map_b1.py\
        -v 1\
        -s {sub_name}\
        -o {recon_dir}\
        -g {niftis_str}\
        -b {b1_map_str}\
        -r {join(recon_dir,f'{sub_name}_t1_map_b1.json')}\
        -d fit\
        -m {mask_file}")

    recon_map = join(recon_dir, f'{sub_name}_T1map.nii.gz')

    # postprocessing: normalization of reconstructed t1 map
    if do_normalise_after:
        postproc_dir = join(cwd, 'tmp_t1', 'postproc')
        if not exists(postproc_dir):
            makedirs(postproc_dir)
        recon_map = join(recon_dir, f'{sub_name}_T1map.nii.gz')
        system('cp {} {}'.format(recon_map, postproc_dir))
        time_elapsed = time.time() - start_time
        print('[INFO,  t={:.2f}s] normalizing reconstructed t1 map...'.format(time_elapsed))
        image = join(postproc_dir, f'{sub_name}_T1map.nii.gz')
        system('gunzip -df {}'.format(image))
        image = join(postproc_dir, f'{sub_name}_T1map.nii')
        out_info = to_MNI(image, segmented=out_info.nipype_results['segment'])
        norm_recon_map = out_info['anat']


    # doing the plots
    if do_plot:
        time_elapsed = time.time() - start_time
        print('\n[INFO,  t={:.2f}s] plotting the map...'.format(time_elapsed))
        plot_dir = join(cwd, 'tmp_t1', 'plot')
        if not exists(plot_dir):
            makedirs(plot_dir)
        if do_normalise_after:
            t1_img = norm_recon_map
        else:
            t1_img = join(recon_dir, f'{sub_name}_T1map.nii.gz')
        plot_thresholded_qmap(img=t1_img,
                            thresh="99",
                            map=f"{sub_name}_T1map_fit",
                            interactive=True,
                            coords=(10,56,43),
                            output_folder=plot_dir)

    time_elapsed = time.time() - start_time
    print('\n[INFO,  t={:.2f}s] DONE'.format(time_elapsed))

    # move derived files out and delete tmp_t1 directory
    final_recon_map = join(SAVE_TO, f'{sub_name}_space-{space}_T1map.nii.gz')
    if do_normalise_after:
        system('gzip {}'.format(norm_recon_map))
        recon_map = norm_recon_map + '.gz'
    shutil.move(recon_map, final_recon_map)

    if do_plot:
        for fi in listdir(plot_dir):
            plot_name = join(plot_dir, fi)
            final_plot_name = join(SAVE_TO, final_recon_map.split(sep)[-1].split('.')[0]
                                    + '_' + fi.split('_')[-1])
            shutil.move(plot_name, final_plot_name)

    if not keep_tmp:
        shutil.rmtree(join(SAVE_TO, 'tmp_t1'))

    time_elapsed = time.time() - start_time
    print('\n[INFO,  t={:.2f}s] created {} \n\n'.format(time_elapsed, final_recon_map))
