import os
from glob import glob
import pandas as pd
from ibc_public.connectivity.utils_fc_estimation import get_ses_modality
from nilearn.connectome import sym_matrix_to_vec

### create sc_data dataframe for native space
cache = DATA_ROOT = "/storage/store2/work/haggarwa/"
n_parcels = 200
sub_ses, _ = get_ses_modality("DWI")
sc_data_native = []
for sub, session in sub_ses.items():
    data = {"subject": sub, "measure": "SC", "task": "SC"}
    path = os.path.join(DATA_ROOT, sub, session, "dwi")
    csv = glob(
        os.path.join(
            path,
            f"*connectome_schaefer{n_parcels}_individual_siftweighted.csv",
        )
    )
    matrix = pd.read_csv(csv[0], header=None).to_numpy()
    print(matrix.shape)
    matrix = sym_matrix_to_vec(matrix, discard_diagonal=True)
    data["connectivity"] = matrix
    sc_data_native.append(data)

sc_data_native = pd.DataFrame(sc_data_native)
sc_data_native.to_pickle(
    os.path.join(DATA_ROOT, f"sc_data_native_{n_parcels}")
)

### create sc_data dataframe for MNI space
sub_ses, _ = get_ses_modality("DWI")
sc_data_mni = []
for sub, session in sub_ses.items():
    data = {"subject": sub, "measure": "SC", "task": "SC"}
    path = os.path.join(DATA_ROOT, sub, session, "dwi")
    csv = glob(
        os.path.join(path, "*connectome_schaefer400_MNI152_siftweighted.csv")
    )
    matrix = pd.read_csv(csv[0], header=None).to_numpy()
    print(matrix.shape)
    matrix = sym_matrix_to_vec(matrix, discard_diagonal=True)
    data["connectivity"] = matrix
    sc_data_mni.append(data)

sc_data_mni = pd.DataFrame(sc_data_mni)
sc_data_mni.to_pickle(os.path.join(DATA_ROOT, "sc_data_mni_new"))
