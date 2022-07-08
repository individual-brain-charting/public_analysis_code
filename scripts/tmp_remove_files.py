import os

# root location of sourcedata and derivatives
DATA_ROOT = '/neurospin/ibc/'

# relaxometry session numbers for each subject
sub_sess = {'sub-01': 'ses-21', 'sub-04': 'ses-20', 'sub-05': 'ses-22',
'sub-06': 'ses-20', 'sub-07': 'ses-20', 'sub-08': 'ses-35',
'sub-09': 'ses-19', 'sub-11': 'ses-17', 'sub-12': 'ses-17',
'sub-13': 'ses-20', 'sub-14': 'ses-20', 'sub-15': 'ses-18'}


for sub, sess in sub_sess.items():
    path = os.path.join(DATA_ROOT, 'sourcedata', sub, sess, 'anat')
    all_files = os.listdir(path)

    for file in all_files:
        file_specs = file.split("_")
        if ((file_specs[-1] in ['T2star.json', 'T2star.nii.gz']) and 
        (file_specs[-2] in ['run-01', 'run-02'])):
            os.remove(os.path.join(path, file))


