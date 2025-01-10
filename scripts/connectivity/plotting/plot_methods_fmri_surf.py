import os
import seaborn as sns
from nilearn.plotting import view_img_on_surf
from nilearn import image

sns.set_theme()
sns.set_style("whitegrid")
sns.set_context("talk")

### plot fmri image for methods
cache = DATA_ROOT = "/storage/store/work/haggarwa/"
DATA_ROOT2 = "/storage/store2/work/haggarwa/"
output_dir = os.path.join(DATA_ROOT2, "fmri_methods")
os.makedirs(output_dir, exist_ok=True)

fmri_file = "/storage/store2/data/ibc/derivatives/sub-04/ses-12/func/wrdcsub-04_ses-12_task-MTTNS_dir-pa_run-01_bold.nii.gz"

mean_fmri = image.mean_img(fmri_file)

view_img_on_surf(mean_fmri, surf_mesh="fsaverage").save_as_html(
    "output_dir/fmri.html"
)
