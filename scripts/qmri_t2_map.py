##########################################################################
# NSAp - Copyright (C) CEA, 2016
# Alexandre Vignaud - Yann Leprince
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# Sytem import
from __future__ import print_function
import argparse
import os
import numpy

# Qmri import
from qmri.t2.t2_io import get_serie_echo_times
from qmri.t2.t2_io import load_sequence
from qmri.t2.t2_io import save_sequence
from qmri.t2.decay_fit import decay_fit


# Script documentation
doc = """
T2 map reconstruction
~~~~~~~~~~~~~~~~~~~~~

Reconstruct the t2 map from a multi-constrast t2 relaxometry sequence.

Command:

python $HOME/git/qmri/qmri/scripts/t2_map \
    -v 1 \
    -s sub-11
    -o /neurospin/tmp/agrigis/qmri/processed \
    -n /neurospin/tmp/agrigis/qmri/data/s19.nii.gz \
    -d /neurospin/tmp/agrigis/qmri/data/000019_relaxometry-T2-tra-2mm-multise-12contrastes \
    -c /etc/fsl/5.0/fsl.sh \
    -t 0.3 \
    -m
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
    "-n", "--niirelaxo", dest="niirelaxo", metavar="FILE", required=True,
    help="the relaxometry nifti image.", type=is_file)
parser.add_argument(
    "-m", "--mask", dest="mask", metavar="FILE",
    help=("the mask file."))
parser.add_argument(
    "-t", "--thresh", dest="thresh", default=0.5, type=float,
    help="fractional intensity threshold (0->1), smaller values give larger "
    "brain outline estimates.")
args = parser.parse_args()


# Welcome message and checks
verbose = args.verbose
if verbose > 0:
    print("[INFO] T2 map computation using the analytic method...")
niirelaxo = args.niirelaxo
subjectid = args.subjectid
mask_file = args.mask
if verbose > 0:
    print("[INFO] niirelaxo: '{0}'.".format(niirelaxo))

# Create the working directory
working_dir = args.outdir
if verbose > 0:
    print("[INFO] Working directory: '{0}'.".format(working_dir))

# Load sequence and sequence parameters
echo_times = get_serie_echo_times(niirelaxo)
relaxoarray, relaxoaffine = load_sequence(niirelaxo)
if len(echo_times) != relaxoarray.shape[-1]:
    raise Exception("Wrong echo number in '{0}'.".format(echo_times))
teparms = []
for key in sorted(echo_times.keys()):
    teparms.append(echo_times[key])
teparms = numpy.asarray(teparms)

# Estimate the t2
t2array = decay_fit(relaxoarray, teparms, maskfile=mask_file)
t2file = os.path.join(working_dir, "{0}_T2map.nii.gz".format(subjectid))
save_sequence(t2array, relaxoaffine, t2file)
print("[INFO] t2file: '{0}'".format(t2file)) 
    
