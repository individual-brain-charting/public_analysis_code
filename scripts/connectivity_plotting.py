import os
from glob import glob
import pandas as pd
import seaborn as sns
from ibc_public import utils_connectivity as fc
from nilearn.connectome import sym_matrix_to_vec

sns.set_theme(context="talk", style="whitegrid")

# cache, results, output directory
cache = DATA_ROOT = "/storage/store/work/haggarwa/"
results_dir = f"fc_classification_20230920-163739"
results_dir = os.path.join(DATA_ROOT, results_dir)
output_dir = os.path.join(DATA_ROOT, "final_plots", "supp")

# load the results
dfs = []
sub_dirs = os.listdir(results_dir)
for dir in sub_dirs:
    path = os.path.join(results_dir, dir)
    csvs = glob(os.path.join(path, "*.csv"))
    for csv in csvs:
        df = pd.read_csv(csv)
        dfs.append(df)
df = pd.concat(dfs)

# clean up the results
cols_to_clean = [
    "LinearSVC_predicted_class",
    "true_class",
    "train_sets",
    "test_sets",
    "Dummy_predicted_class",
]
for col in cols_to_clean:
    df[col] = df[col].apply(fc._clean)

# calculate chance level
df = fc.chance_level(df)

# plot the results
fc.do_plots(df, output_dir)


# create sc_data dataframe for native space
sub_ses, _ = fc.get_ses_modality("DWI")
sc_data_native = []
for sub, session in sub_ses.items():
    data = {"subject": sub, "measure": "SC", "task": "SC"}
    path = os.path.join(DATA_ROOT, sub, session, "dwi")
    csv = glob(
        os.path.join(
            path, "*connectome_schaefer400_individual_siftweighted.csv"
        )
    )
    matrix = pd.read_csv(csv[0], header=None).to_numpy()
    print(matrix.shape)
    matrix = sym_matrix_to_vec(matrix, discard_diagonal=True)
    data["connectivity"] = matrix
    sc_data_native.append(data)

sc_data_native = pd.DataFrame(sc_data_native)
sc_data_native.to_pickle(os.path.join(DATA_ROOT, "sc_data_native_new"))

# create sc_data dataframe for native space
sub_ses, _ = fc.get_ses_modality("DWI")
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
