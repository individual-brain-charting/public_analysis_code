"""
This script gathers individual retinotopic maps into one single image
"""
import glob
import os
from os.path import join as pjoin
from ibc_public.utils_data import DERIVATIVES
import matplotlib.pyplot as plt
import numpy as np

wedge_maps = sorted(glob.glob(pjoin(
    DERIVATIVES, '*', '*', 'res_fsaverage7_retinotopy_ffx', 'stat_maps',
    'phase_wedge.png')))
ring_maps = sorted(glob.glob(pjoin(
    DERIVATIVES, '*', '*', 'res_fsaverage7_retinotopy_ffx', 'stat_maps',
    'phase_ring.png')))

n_subjects = len(wedge_maps)

"""
x, y = np.linspace(-1, 1, 101)[np.newaxis], np.linspace(-1, 1, 101)[np.newaxis].T
r, t = np.sqrt(x ** 2 + y ** 2), np.arctan2(x, y)
r, t = r[r < 1], t[r < 1]
plt.figure(figsize=(1, 1), facecolor='w', edgecolor='w')
plt.plot(r * np.cos(t), r * np.sin(t), r, '.')
"""

import matplotlib.tri as tri
import math
n_angles = 100
n_radii = 20
min_radius = 0.0
radii = np.linspace(min_radius, 1., n_radii)

angles = np.linspace(0, 2 * math.pi, n_angles, endpoint=False)
angles = np.repeat(angles[..., np.newaxis], n_radii, axis=1)
angles[:, 1::2] += math.pi / n_angles

x = (radii * np.sin(angles)).flatten()
y = (-radii * np.cos(angles)).flatten()
z1 = ((radii > 0) * angles).flatten()
z2 = (radii * (angles > -1)).flatten()

# Create the Triangulation; no triangles so Delaunay triangulation created.
triang = tri.Triangulation(x, y)

# Mask off unwanted triangles.
xmid = x[triang.triangles].mean(axis=1)
ymid = y[triang.triangles].mean(axis=1)
mask = np.where(xmid * xmid + ymid * ymid < min_radius * min_radius, 1, 0)
triang.set_mask(mask)

# Illustrate Gouraud shading.
wedge = plt.figure(figsize=(1, 1))
plt.gca().set_aspect('equal')
plt.tripcolor(triang, z1, shading='gouraud', cmap='RdBu_r')
plt.axis('off')
plt.savefig('/tmp/wedge.png', dpi=300)

ring = plt.figure(figsize=(1, 1))
plt.gca().set_aspect('equal')
plt.tripcolor(triang, z2, shading='gouraud', cmap='RdBu_r')
plt.axis('off')
plt.savefig('/tmp/ring.png', dpi=300)

plt.figure(figsize=(8, 6), facecolor='w', edgecolor='w')

delta = 2./ n_subjects
for i, img in enumerate(wedge_maps):
    ax = plt.axes([np.mod(i, 2) * .25, (i // 2) * delta, .25, delta])
    img = plt.imread(img)
    ax.imshow(img)
    plt.axis('off')

for i, img in enumerate(ring_maps):
    ax = plt.axes([0.5 + np.mod(i, 2) * .25, (i // 2) * delta, .25, delta])
    img = plt.imread(img)
    ax.imshow(img)
    plt.axis('off')

ax = plt.axes([.2, .45, .1, .1])
ax.imshow(plt.imread('/tmp/wedge.png'))
plt.axis('off')
ax = plt.axes([.7, .45, .1, .1])
ax.imshow(plt.imread('/tmp/ring.png'))
plt.axis('off')
plt.plot([.5, .5], [-1, 2], linewidth=2, color='k')
plt.axis('off')


write_dir = '/neurospin/tmp/bthirion'
plt.savefig(os.path.join(write_dir, 'retino_montage.pdf'),
            facecolor='w', dpi=300)

plt.show(block=False)
