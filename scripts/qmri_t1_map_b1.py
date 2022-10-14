##########################################################################
# NSAp - Copyright (C) CEA, 2016 - 2021
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# Sytem import
import argparse
import os
import nibabel
import math
import json
import numpy as np

# Qmri import
from qmri.t1.t1_io import load_sequence
from qmri.t1.t1_io import save_sequence
from qmri.t1.interpolation import resample_image
from qmri.t1.despot1 import QuantitativeT1Reconstruction



# Script documentation
doc = """
T1 map reconstruction
~~~~~~~~~~~~~~~~~~~~~

Reconstruct the T1 map from a B1 image and N GRE images acquired at different
flip angles.
Two reconstruction methods are available:

* 'analytic': an analytic method, only available for 2 flip angles (fast).
* 'fit': a joint T1, M0 fit (slower but better).

Record Json files, following the bids (http://bids.neuroimaging.io/) standard,
are expected to describe the B1 and GRE sequences.
The files must specified the flip angles and repetiton times of both sequences.

::

{
    <subject_code_in_study>: {
        <B1>: {
            "FlipAngle": <flip_angle>,
            "RepetitonTime": <repetiton_time>
        },
        <GRE>: {
            "FlipAngle": <flip_angles>,
            "RepetitonTime": <repetiton_times>
        }
    }
}

This file can be generated automatically using the 'qmri_t1_map_b1_parameters'
command.
"""

def is_file(filearg):
    """ Type for argparse - checks that file exists but does not open.
    """
    if not os.path.isfile(filearg):
        raise argparse.ArgumentError(
            "The file '{0}' does not exist!".format(filearg))
    return filearg


def is_directory(dirarg):
    """ Type for argparse - checks that directory exists.
    """
    if not os.path.isdir(dirarg):
        raise argparse.ArgumentError(
            "The directory '{0}' does not exist!".format(dirarg))
    return dirarg


parser = argparse.ArgumentParser(description=doc)
parser.add_argument(
    "-v", "--verbose", dest="verbose", type=int, choices=[0, 1], default=0,
    help="increase the verbosity level: 0 silent, 1 verbose.")
parser.add_argument(
    "-s", "--subjectid", dest="subjectid", required=True,
    help="the subject code in study.")
parser.add_argument(
    "-o", "--outdir", dest="outdir", required=True, metavar="PATH",
    help="the reconstruction directory.", type=is_directory)
parser.add_argument(
    "-g", "--niigres", dest="niigres", nargs="+", required=True,
    help="the gre nifti images.", type=is_file)
parser.add_argument(
    "-b", "--niib1", dest="niib1", metavar="FILE", required=True,
    help="the flip angle map (1/10degrees).", type=is_file)
parser.add_argument(
    "-r", "--record", dest="record", metavar="FILE", required=True,
    help="a record containing the B1 and GRE sequence parameters.",
    type=is_file)
parser.add_argument(
    "-d", "--method", dest="method", choices=("analytic", "fit"),
    help="the reconstruction method.")
parser.add_argument(
    "-m", "--mask", dest="mask", metavar="FILE",
    help=("if activated generate a mask with FSL BET that is applied on the "
          "T1 map."))
parser.add_argument(
    "-t", "--thresh", dest="thresh", default=0.5, type=float,
    help="fractional intensity threshold (0->1), smaller values give larger "
    "brain outline estimates.")
parser.add_argument(
    "-k", "--spoiling-correction", action="store_true",
    help="if set, perform a T1 spoiling correction.")
parser.add_argument(
    "-a", "--average-t2", type=float, default=85.5,
    help="the average T2 value in ms.")

args = parser.parse_args()


"""
Welcome message and checks.
"""
verbose = args.verbose
if verbose > 0:
    print("[INFO] Starting T1 map computation...")
b1niifile = args.niib1
greniifiles = args.niigres
recordfile = args.record
method = args.method
mask_file = args.mask
subjectid = args.subjectid
working_dir = args.outdir
spoiling_correction = args.spoiling_correction
average_t2 = args.average_t2
if verbose > 0:
    print("[INFO] b1nii: '{0}'.".format(b1niifile))
    print("[INFO] greniis: '{0}'.".format(greniifiles))
    print("[INFO] record: '{0}'.".format(recordfile))
    print("[INFO] method: '{0}'.".format(method))
    print("[INFO] spoiling correction: {0}.".format(spoiling_correction))
    print("[INFO] average t2 for spoiling correction: '{0}'.".format(average_t2))
    print("[INFO] working directory: '{0}'.".format(working_dir))
with open(recordfile) as open_file:
    record = json.load(open_file)
if (len(greniifiles) != len(record[subjectid]["GRE"]["FlipAngle"]) or
        len(greniifiles) != len(record[subjectid]["GRE"]["RepetitionTime"])):
    raise ValueError("Can't find GRE images meta information.")
if (not isinstance(record[subjectid]["B1"]["FlipAngle"], float) or
        not isinstance(record[subjectid]["B1"]["RepetitionTime"], float)):
    raise ValueError("Can't find B1 images meta information.")


"""
Load sequences.
"""
b1array, b1affine = load_sequence(b1niifile)
b1fa = record[subjectid]["B1"]["FlipAngle"]
b1tr = record[subjectid]["B1"]["RepetitionTime"]
gre_sequences = []
grefas = record[subjectid]["GRE"]["FlipAngle"]
gretrs = record[subjectid]["GRE"]["RepetitionTime"]
for fimage, fa, tr in zip(greniifiles, grefas, gretrs):
        gre_sequences.append((load_sequence(fimage)[0], fa, tr))

if verbose > 0:
    print("[INFO] b1fa: {0}.".format(b1fa))
    print("[INFO] b1tr: {0}.".format(b1tr))
    print("[INFO] grefas: {0}.".format(grefas))
    print("[INFO] gretrs: {0}.".format(gretrs))

"""
Get a relative measure of the flip angle spatial distribution
centered at the therical acquisition flip angle.
"""
relative_b1array = b1array * math.pi / 1800 / math.radians(b1fa)
relative_b1file = os.path.join(working_dir,
                               "{0}_B1map.nii.gz".format(subjectid))
save_sequence(relative_b1array, b1affine, relative_b1file)


"""
Resample the B1 image to match the GREs geometry using linear interpolation.
"""
resampled_b1file = os.path.join(
    working_dir, "resampled_" + os.path.basename(relative_b1file))

moving_b1array, moving_b1affine = load_sequence(relative_b1file)
target_grearray, target_greaffine = load_sequence(greniifiles[0])
resampled_b1array = resample_image(
        moving_b1array, target_grearray, moving_b1affine, target_greaffine,
        interp_order=1)
save_sequence(resampled_b1array, target_greaffine, resampled_b1file)


"""
Calculate T1 map from DESPOT data unsing an analytic method or a fit.
"""
b1array, b1affine = load_sequence(resampled_b1file)
maskarray, maskaffine = load_sequence(mask_file)
t1_constructor = QuantitativeT1Reconstruction(
    gre_sequences, None, None, None, None, None,  b1_map_array=b1array,
    mask_array=maskarray, spoiling_correction=spoiling_correction,
    average_t2=average_t2, outdir=working_dir)
if method == "fit":
    if len(gre_sequences) <= 2:
        raise ValueError("Expect more than two flip angles to perform the fit "
                         "method.")
    t1array, m0array, foptarray = t1_constructor.get_m0_t1_maps()
else:
    if len(gre_sequences) != 2:
        raise ValueError("Expect two flip angles to perform the analytic "
                         "method.")
    m0array, foptarray = (None, None)
    t1array = t1_constructor.get_t1_map_analytic_2_point()
wrongest_array = (np.isnan(t1array) | np.isinf(t1array)).astype(int)
t1file = os.path.join(working_dir, "{0}_T1map.nii.gz".format(subjectid))
wrongestfile = os.path.join(
    working_dir, "{0}_wrongest_T1map.nii.gz".format(subjectid))
m0file, foptfile = (None, None)
save_sequence(t1array, b1affine, t1file)
save_sequence(wrongest_array, b1affine, wrongestfile)
if m0array is not None:
    m0file = os.path.join(working_dir, "{0}_M0.nii.gz".format(subjectid))
    save_sequence(m0array, b1affine, m0file)
if foptarray is not None:
    foptfile = os.path.join(working_dir, "{0}_fopt.nii.gz".format(subjectid))
    save_sequence(foptarray, b1affine, foptfile)
if verbose > 0:
    print("[RESULT] t1: {0}".format(t1file))
    print("[RESULT] wrongly estimated t1: {0}".format(wrongestfile))
    print("[RESULT] m0: {0}".format(m0file)) 
    print("[RESULT] fopt: {0}".format(foptfile)) 
    
