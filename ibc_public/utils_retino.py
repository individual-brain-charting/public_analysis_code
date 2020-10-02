import os
import numpy as np
import nibabel as nib
NEGINF = -np.inf
ALL_REG = ['sin_ring_pos', 'cos_ring_pos', 'sin_ring_neg',  'cos_ring_neg',
           'sin_wedge_pos', 'cos_wedge_pos', 'sin_wedge_neg',
           'cos_wedge_neg']


def combine_phase(phase_pos, phase_neg, offset=0, hemo=None):
    """ Combine the phases estimated in two directions"""
    if hemo is None:
        # estimate hemodynamic delay
        hemo = 0.5 * (phase_pos + phase_neg)
        hemo += np.pi * (hemo < 0) # - np.pi / 4)
        hemo += np.pi * (hemo < 0) # - np.pi / 4)

    # first phase estimate
    pr1 = phase_pos - hemo
    pr2 = hemo - phase_neg
    pr2[(pr1 - pr2) > np.pi] += (2 * np.pi)
    pr2[(pr1 - pr2) > np.pi] += (2 * np.pi)
    pr1[(pr2 - pr1) > np.pi] += (2 * np.pi)
    pr1[(pr2 - pr1) > np.pi] += (2 * np.pi)
    phase = 0.5 * (pr1 + pr2)

    # add the offset and bring back to [-pi, +pi]
    phase += offset
    phase += 2 * np.pi * (phase < - np.pi)
    phase += 2 * np.pi * (phase < - np.pi)
    phase -= 2 * np.pi * (phase > np.pi)
    phase -= 2 * np.pi * (phase > np.pi)
    return phase, hemo


def phase_maps(data, offset_ring=0, offset_wedge=0, do_wedge=True,
               do_ring=True):
    """ Compute the phase for each functional map

    Parameters
    ----------
    data: dictionary with keys 'sin_wedge_pos', 'sin_wedge_neg',
          'cos_wedge_neg', 'cos_ring_pos', 'sin_ring_neg', 'cos_wedge_pos',
          'sin_ring_pos', 'cos_ring_neg'
         arrays of shape (n_nodes) showing fMRI activations
         for different retino conditions

    offset_ring: float,
        offset value to apply to the ring phase

    offset_wedge: float,
        offset value to apply to the wedge phase

    do_wedge: bool,
        should we do the ring phase estimation or not

    do_ring: bool,
        should we do the ring phase estimation or not

    mesh: path or mesh instance, optional
        underlying mesh model
    """
    phase_ring, phase_wedge, hemo = None, None, None
    if do_ring:
        phase_ring_pos = np.arctan2(data['sin_ring_pos'], data['cos_ring_pos'])
        phase_ring_neg = np.arctan2(data['sin_ring_neg'], data['cos_ring_neg'])
        phase_ring, hemo_ring = combine_phase(
            phase_ring_pos, phase_ring_neg, offset_ring, hemo=hemo)
        hemo = hemo_ring

    if do_wedge:
        phase_wedge_pos = np.arctan2(data['sin_wedge_pos'],
                                     data['cos_wedge_pos'])
        phase_wedge_neg = np.arctan2(data['sin_wedge_neg'],
                                     data['cos_wedge_neg'])
        phase_wedge, hemo_wedge = combine_phase(
            phase_wedge_pos, phase_wedge_neg, offset_wedge)
        hemo = hemo_wedge

    if do_ring and do_wedge:
        hemo = 0.5 * (hemo_ring + hemo_wedge)

    return phase_wedge, phase_ring, hemo


def angular_maps(side, contrast_path, mask_img, mesh_path=None,
                 all_reg=ALL_REG, threshold=3.1,
                 offset_wedge=0, offset_ring=0,
                 do_wedge=True, do_ring=True, do_phase_unwrapping=False):
    """
    Parameters
    ----------
    side: {'left', 'right', False}
    all_reg: list of strings,
            identifiers of the contrast files used in angular mapping
    threshold: float, optional
               threshold defining the brain regions
               where the analysis is performed
    offset_wedge: float, optional
                  offset to be applied to wedge angle
    offset_ring float, optional
                  offset to be applied to ring angle
    """
    if side is False:
        stat_map = os.path.join(contrast_path, 'effects_of_interest_z_map.nii')

        # create an occipital data_mask
        mask = nib.load(stat_map).get_data() > threshold

        # load and mask the data
        data = {}
        for r in all_reg:
            contrast_file = os.path.join(contrast_path, '%s_con.nii' % r)
            data[r] = nib.load(contrast_file).get_data()[mask]
        do_phase_unwrapping = False
        mesh = None
    else:
        pass

    # Then compute the activation phase in these regions
    phase_wedge, phase_ring, hemo = phase_maps(
        data, offset_ring, offset_wedge, do_wedge, do_ring,
        do_phase_unwrapping, mesh=mesh, mask=mask)

    # write the results
    data_, id_ = [hemo, mask[mask > 0]], ['hemo', 'mask']
    if do_ring:
        data_.append(phase_ring)
        id_.append('phase_ring')
    if do_wedge:
        data_.append(phase_wedge)
        id_.append('phase_wedge')

    if side is False:
        for (x, name) in zip(data_, id_):
            wdata = np.zeros(nib.load(stat_map).shape)
            wdata[mask > 0] = x
            wim = nib.Nifti1Image(wdata, nib.load(stat_map).affine)
            nib.save(wim, os.path.join(contrast_path, '%s.nii' % name))


# Compute fixed effects_maps for effects of interest -> retinotopic maps
