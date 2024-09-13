import sys
sys.path.insert(0, '../src/')

from l1hessian import hessian_l1_solve
import gpytoolbox as gp
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from matplotlib import colormaps
import time

regen_np = True if int(sys.argv[1]) == 1 else False
quick = True if int(sys.argv[2]) == 1 else False

def heat_distance(V, F, si):
    # compute geodesic distance from points si.  based on Crane et al.
    t = np.mean(gp.halfedge_lengths(V, F))**2
    M, L = gp.massmatrix(V, F), -gp.cotangent_laplacian(V, F)
    mask = np.zeros((V.shape[0], 1))
    mask[si] = 1
    heatu = sp.linalg.spsolve(M-t*L, mask)[:, None]
    G = gp.grad_intrinsic(gp.halfedge_lengths_squared(V, F), F)
    vecs = G @ heatu
    vecs = vecs / np.tile(np.linalg.norm(np.reshape(vecs, (F.shape[0], 2), order="F"), axis=1, keepdims=True), reps=(2, 1))
    Mx = 0.5*sp.diags(np.hstack([gp.doublearea(V, F), gp.doublearea(V, F)]))
    phi = G.T @ Mx @ vecs
    
    u = sp.linalg.spsolve(L, phi)
    u = (u - np.amin(u)) / (np.amax(u)-np.amin(u))
    
    return u

heat_ridge = lambda V, F, si: np.where(heat_distance(V, F, si) < 0.5, 2*heat_distance(V, F, si), 2-2*heat_distance(V, F, si))
heat_ridge_alpha = lambda V, F, si, alpha: np.where(heat_distance(V, F, si) < alpha, heat_distance(V, F, si), 2*alpha-heat_distance(V, F, si))

# file locations
outpng = "../results/pngs/person_fninterp_levels.png"
outnpy = "../results/npys/person_fninterp.npy"

# tunable params
hesstype = "vfef"
mu = 100
eps = 1e-12
levels = 50
sample_count = 80

modelname = "person"
modelfilename = "../models/good_person2.obj"

V, F = gp.read_mesh(modelfilename)
V, F = gp.subdivide(V, F, method='loop', iters=1)

# get hel
hel = gp.halfedge_lengths(V, F)
hel /= np.amax(hel)

rng = np.random.default_rng(0)

if regen_np:
    orig_fn = heat_ridge_alpha(V, F, 848, 0.4)
    
    si = rng.choice(np.array(V.shape[0]), size=sample_count, replace=False)
    
    sv = orig_fn[si]
    
    st = time.time()
    recovered_fn = hessian_l1_solve(hel, F, mode=hesstype, y=si, k=sv)
    et = time.time()
    print("ours", et-st)
    st = time.time()
    lapl2_fn = gp.min_quad_with_fixed(gp.biharmonic_energy_intrinsic(hel**2, F, bc='curved_hessian'), k=si, y=sv) # bc='curved_hessian'?
    et = time.time()
    print("lapl2", et-st)
    st = time.time()
    steinetal2018_fn = hessian_l1_solve(hel, F, mode="stein et al 2018", y=si, k=sv, V=V)
    et = time.time()
    print("stein", et-st)
    
    
    with open(outnpy, 'wb') as f:
        np.save(f, orig_fn)
        np.save(f, lapl2_fn)
        np.save(f, steinetal2018_fn)
        np.save(f, recovered_fn)

with open(outnpy, 'rb') as f:
    orig_fn, lapl2_fn, steinetal2018_fn, recovered_fn = np.load(f), np.load(f), np.load(f), np.load(f)

r, c = 1, 4
fig, axs = plt.subplots(r, c)
fig.set_size_inches((9, 5))
fig.tight_layout()

plt.subplots_adjust(top=1,
                    bottom=0,
                    left=0,
                    right=1,
                    hspace=0.0,
                    wspace=0.0)

cmapnames = ["Greys_r", "YlOrRd_r", "PuRd_r", "YlGnBu_r"]
fns = [orig_fn, lapl2_fn, steinetal2018_fn, recovered_fn]

axs[0].tricontourf(V[:, 0], V[:, 1], F, fns[0], levels=levels, cmap=cmapnames[0]) #cmap='plasma')
axs[0].tricontour(V[:, 0], V[:, 1], F, fns[0], levels=levels, linewidths=0.2, linestyles='solid', colors='k')
axs[0].triplot(V[:, 0], V[:, 1], F, color=(0.5, 0.5, 0.75), linewidth=0.1)

axs[1].tricontourf(V[:, 0], V[:, 1], F, fns[1], levels=levels, cmap=cmapnames[1]) #cmap='plasma')
axs[1].tricontour(V[:, 0], V[:, 1], F, fns[1], levels=levels, linewidths=0.2, linestyles='solid', colors='k')

axs[2].tricontourf(V[:, 0], V[:, 1], F, fns[2], levels=levels, cmap=cmapnames[2]) #cmap='plasma')
axs[2].tricontour(V[:, 0], V[:, 1], F, fns[2], levels=levels, linewidths=0.2, linestyles='solid', colors='k')

axs[3].tricontourf(V[:, 0], V[:, 1], F, fns[3], levels=levels, cmap=cmapnames[3]) #cmap='plasma')
axs[3].tricontour(V[:, 0], V[:, 1], F, fns[3], levels=levels, linewidths=0.2, linestyles='solid', colors='k')

for i in range(c):
    fnsi_rescaled = (fns[i][si]-fns[i].min())/(fns[i].max()-fns[i].min()) # rescale to have vmin, vmax the same as the actual fn
    colorsi = colormaps[cmapnames[i]](fnsi_rescaled)
    axs[i].scatter(V[si, 0], V[si, 1], c=colorsi, marker='o', edgecolor='r', s=10, linewidths=0.7)
    
    # set axis style
    axs[i].axis('off')
    axs[i].set_aspect('equal', 'box')
    axs[i].get_xaxis().set_visible(False)
    axs[i].get_yaxis().set_visible(False)

plt.savefig(outpng, dpi=600, bbox_inches='tight', pad_inches=0)
    