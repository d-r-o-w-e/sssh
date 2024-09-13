import sys
sys.path.insert(0, '../src/')

from l1hessian import hessian_l1_solve
import gpytoolbox as gp
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
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
outpng = "../results/pngs/octo_fnholefill.png"
outnpy = "../results/npys/octo_fnholefill.npy"

# tunable params
hesstype = "vfef"
mu = 100
eps = 1e-12
levels = 50

modelname = "octo"
modelfilename = "../models/good_octo_3.obj"

V, F = gp.read_mesh(modelfilename)

V = V - np.mean(V, axis=0, keepdims=True)
V = V / np.amax(np.linalg.norm(V, axis=1))

# get hel
hel = gp.halfedge_lengths(V, F)
hel /= np.amax(hel)

if regen_np:
    orig_fn = heat_ridge_alpha(V, F, 848, 0.57)
    
    si = np.nonzero(np.logical_and.reduce([np.linalg.norm(V - np.array([[0.3, 0, 0]]), axis=1) > 0.13,
                                           np.linalg.norm(V - np.array([[-0.4, -0.22, 0]]), axis=1) > 0.07,
                                           np.linalg.norm(V - np.array([[-0.13, 0.28, 0]]), axis=1) > 0.1]))[0]
    sv = orig_fn[si]
    
    st = time.time()
    recovered_fn = hessian_l1_solve(hel, F, mode=hesstype, y=si, k=sv)
    et = time.time()
    print("ours", et - st)
    st = time.time()
    lapl2_fn = gp.min_quad_with_fixed(gp.biharmonic_energy_intrinsic(hel**2, F, bc='curved_hessian'), k=si, y=sv)
    et = time.time()
    print("L2", et - st)
    st = time.time()
    steinetal2018_fn = hessian_l1_solve(hel, F, mode="stein et al 2018", y=si, k=sv, V=V)
    et = time.time()
    print("stein", et - st)
    
    with open(outnpy, 'wb') as f:
        np.save(f, orig_fn)
        np.save(f, lapl2_fn)
        np.save(f, steinetal2018_fn)
        np.save(f, recovered_fn)

with open(outnpy, 'rb') as f:
    orig_fn, lapl2_fn, steinetal2018_fn, recovered_fn = np.load(f), np.load(f), np.load(f), np.load(f)

r, c = 1, 4
fig, axs = plt.subplots(r, c)
fig.set_size_inches((20, 5))
fig.tight_layout()

plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

circle1 = [plt.Circle((0.3, 0), 0.13, fill=False, ls='--', color='r') for i in range(c)]
circle2 = [plt.Circle((-0.4, -0.22), 0.07, fill=False, ls='--', color='r') for i in range(c)]
circle3 = [plt.Circle((-0.13, 0.28), 0.1, fill=False, ls='--', color='r') for i in range(c)]

axs[0].tricontourf(V[:, 0], V[:, 1], F, orig_fn, levels=levels, cmap="Greys_r") #cmap='plasma')
axs[0].tricontour(V[:, 0], V[:, 1], F, orig_fn, levels=levels, linewidths=0.2, linestyles='solid', colors='k')
axs[0].triplot(V[:, 0], V[:, 1], F, color=(0.5, 0.5, 0.75), linewidth=0.07)

axs[1].tricontourf(V[:, 0], V[:, 1], F, lapl2_fn, levels=levels, cmap="YlOrRd_r") #cmap='plasma')
axs[1].tricontour(V[:, 0], V[:, 1], F, lapl2_fn, levels=levels, linewidths=0.2, linestyles='solid', colors='k')

axs[2].tricontourf(V[:, 0], V[:, 1], F, steinetal2018_fn, levels=levels, cmap="PuRd_r") #cmap='plasma')
axs[2].tricontour(V[:, 0], V[:, 1], F, steinetal2018_fn, levels=levels, linewidths=0.2, linestyles='solid', colors='k')

axs[3].tricontourf(V[:, 0], V[:, 1], F, recovered_fn, levels=levels, cmap="YlGnBu_r") #cmap='plasma')
axs[3].tricontour(V[:, 0], V[:, 1], F, recovered_fn, levels=levels, linewidths=0.2, linestyles='solid', colors='k')

for i in range(c):
    # add circle outlines
    axs[i].add_patch(circle1[i])
    axs[i].add_patch(circle2[i])
    axs[i].add_patch(circle3[i])
    
    # set axis style
    axs[i].axis('off')
    axs[i].set_aspect('equal', 'box')
    axs[i].get_xaxis().set_visible(False)
    axs[i].get_yaxis().set_visible(False)

plt.savefig(outpng, dpi=600, bbox_inches='tight', pad_inches=0)
    