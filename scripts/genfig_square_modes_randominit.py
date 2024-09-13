# Roughly following the algorithm from "Breaking Good: Fracture Modes for Realtime Destruction" (Sellan et al 2022)
import sys
sys.path.insert(0, '../src/')

from l1hessian import hessian_l1_modes, hessian_L2_modes
import gpytoolbox as gp
import numpy as np
import matplotlib.pyplot as plt

regen_np = True if int(sys.argv[1]) == 1 else False
quick = True if int(sys.argv[2]) == 1 else False

# tunable params
hesstype = "vfef"
mu = 1000000
k = 8
eps = 1e-10
levels = 15
seed = 1 # set to -1 for actually random
cmapname = "YlGnBu_r"
l2cmapname = "YlOrRd_r"
steincmapname = "PuRd_r"

# file locations
outpng = "../results/pngs/square_modes_ours_randominit_seed" + str(seed) + "mu" + str(mu) + ".png"
l2outpng = "../results/pngs/square_modes_l2_randominit_seed" + str(seed) + "mu" + str(mu) + ".png"
outnpy = "../results/npys/square_modes_randominit_seed" + str(seed) + ".npy"

modelname = "irreg_sheet"
modelfilename = "../models/irreg_mesh_128_minangle45.obj"
V, F = gp.read_mesh(modelfilename)

# divide by max M entry
hel = gp.halfedge_lengths(V, F)
hel /= np.amax(gp.massmatrix(V, F).data)

if regen_np:
    modes, eigs = hessian_l1_modes(hel, F, k=k, mu=mu, eps=eps, hesstype=hesstype, randinit=True, seed=seed, verbose=True)
    l2modes, l2eigs = hessian_L2_modes(hel, F, k=k, mu=1.0, eps=eps, randinit=True, seed=seed, verbose=True)
    with open(outnpy, 'wb') as f:
        np.save(f, modes)
        np.save(f, eigs)
        np.save(f, l2modes)
        np.save(f, l2eigs)

with open(outnpy, 'rb') as f:
    print(outnpy)
    modes, eigs = np.load(f), np.load(f)
    l2modes, l2eigs = np.load(f), np.load(f)

modesmin, modesmax = np.amin(modes), np.amax(modes)
l2modesmin, l2modesmax = np.amin(l2modes), np.amax(l2modes)

# draw all the modes in a single figure
r, c = 2, k//2
fig, axs = plt.subplots(2, k//2)
fig.set_size_inches((4, 2))
fig.tight_layout()

# our modes
plt.subplots_adjust(left=0, bottom=0.06, right=1, top=0.98, wspace=0.02, hspace=0.15)
for i in range(k):
    j = (i//c, i%c)
    axs[j].tricontourf(V[:, 0], V[:, 1], F, modes[:, i], levels=levels, vmin=modesmin, vmax=modesmax, cmap=cmapname)
    axs[j].tricontour(V[:, 0], V[:, 1], F, modes[:, i], levels=levels, vmin=modesmin, vmax=modesmax, linewidths=0.2, linestyles='solid', colors='k')
    axs[j].text(0.5, -0.1, r"$\lambda=$%.5f" % eigs[i], fontsize=5, ha='center')
    
    # set axis style
    axs[j].axis('off')
    axs[j].set_aspect('equal', 'box')
    axs[j].get_xaxis().set_visible(False)
    axs[j].get_yaxis().set_visible(False)

plt.savefig(outpng, dpi=300, bbox_inches='tight', pad_inches=0)

plt.clf()

r, c = 2, k//2
fig, axs = plt.subplots(2, k//2)
fig.set_size_inches((4, 2))
fig.tight_layout()

# L2 modes
plt.subplots_adjust(left=0, bottom=0.06, right=1, top=0.98, wspace=0.02, hspace=0.15)
for i in range(k):
    j = (i//c, i%c)
    axs[j].tricontourf(V[:, 0], V[:, 1], F, l2modes[:, i], levels=levels, vmin=l2modesmin, vmax=l2modesmax, cmap=l2cmapname)
    axs[j].tricontour(V[:, 0], V[:, 1], F, l2modes[:, i], levels=levels, vmin=l2modesmin, vmax=l2modesmax, linewidths=0.2, linestyles='solid', colors='k')
    print(l2eigs)
    axs[j].text(0.5, -0.1, r"$\lambda=$%.5f" % l2eigs[i], fontsize=5, ha='center')
    
    # set axis style
    axs[j].axis('off')
    axs[j].set_aspect('equal', 'box')
    axs[j].get_xaxis().set_visible(False)
    axs[j].get_yaxis().set_visible(False)

plt.savefig(l2outpng, dpi=300, bbox_inches='tight', pad_inches=0)

plt.clf()