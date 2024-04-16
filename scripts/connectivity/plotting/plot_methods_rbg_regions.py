from nilearn.plotting import plot_glass_brain
from nilearn import datasets
import numpy as np
import matplotlib.pyplot as plt

cache = "/storage/store2/work/haggarwa/"

atlas = datasets.fetch_atlas_schaefer_2018(
    data_dir=cache, n_rois=100, resolution_mm=2
)
display = plot_glass_brain(None, display_mode="r")
colors = ["tab:red", "tab:blue", "tab:green", "tab:purple"]
parcels = [[51, 61], [95, 98], [98, 100]]

for parcel, color in zip(parcels, colors):
    display.add_contours(atlas.maps, filled=True, levels=parcel, colors=color)

display.savefig("glass_brain.png", dpi=600)
display.close()


fig, ax = plt.subplots()
n = 100
ax.plot(np.arange(n), np.random.random_sample(n), color="tab:red")
ax.plot(
    np.arange(n),
    (np.random.random_sample(n) + 1),
    color="tab:green",
)
ax.plot(np.arange(n), (np.random.random_sample(n) + 2), color="tab:blue")
ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
plt.savefig("line.png", dpi=600, transparent=True, bbox_inches="tight")
plt.close()
