##########################################################################
# NSAp - Copyright (C) CEA, 2016 - 2021
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# Sytem import
import argparse
import json
import os

# Qmri import
from qmri.t1.t1_io import load_b1_sequence_parameters
from qmri.t1.t1_io import load_gre_sequence_parameters


# Script documentation
doc = """
T1 map reconstruction parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get the B1 and GRE sequences associated parameters. 
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
    help="the destination directory.", type=is_directory)
parser.add_argument(
    "-g", "--jsongres", dest="jsongres", nargs="+", required=True,
    help="one json file of each GRE sequence.", type=is_file)
parser.add_argument(
    "-b", "--jsonb1", dest="jsonb1", metavar="FILE", required=True,
    help="one json file of the b1 sequence.", type=is_file)
args = parser.parse_args()


"""
Welcome message and checks
"""
verbose = args.verbose
if verbose > 0:
    print("[INFO] Starting B1 and GRE sequences parameters extraction...")
b1jsonfile = args.jsonb1
grejsonfiles = args.jsongres
working_dir = args.outdir
subjectid = args.subjectid
if verbose > 0:
    print("[INFO] b1jsonfile: '{0}'.".format(b1jsonfile))
    print("[INFO] grejsonfiles: '{0}'.".format(grejsonfiles))
    print("[INFO] working directory: '{0}'.".format(working_dir))


"""
Load sequences and sequences parameters
"""
b1fa, b1tr = load_b1_sequence_parameters(b1jsonfile)
gre_sequences = []
for fjson in grejsonfiles:
    gre_sequences.append(load_gre_sequence_parameters(fjson))
grefas = [x[0] for x in gre_sequences]
gretrs = [x[1] for x in gre_sequences]
if verbose > 0:
    print("[INFO] b1fa: {0}.".format(b1fa))
    print("[INFO] b1tr: {0}.".format(b1tr))
    print("[INFO] grefas: {0}.".format(grefas))
    print("[INFO] gretrs: {0}.".format(gretrs))


"""
Save the result following bids (http://bids.neuroimaging.io/) standard.
"""
record = {
    subjectid: {
        "B1": {
            "FlipAngle": b1fa,
            "RepetitionTime": b1tr
        },
        "GRE": {
            "FlipAngle": grefas,
            "RepetitionTime": gretrs
        }
    }
}
record_file = os.path.join(working_dir, "{0}_t1_map_b1.json".format(subjectid))
with open(record_file, "wt") as open_file:
    json.dump(record, open_file, indent=4)
if verbose > 0:
    print("[INFO] record: {0}.".format(record_file))



