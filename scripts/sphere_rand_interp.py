import sys
sys.path.insert(0, '../src/')

from l1hessian import hessian_l1_solve
import gpytoolbox as gp
import numpy as np
import scipy.sparse as sp
import polyscope as ps

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

ps.set_program_name("sphere random point scalar interpolation")
ps.init()

# mesh choices
modelfilename = "../models/ico_highres.obj"

V, F = gp.read_mesh(modelfilename)

nv = 200 # number of vertices to sample from the selected function

# select a ground truth function
u_gt = heat_ridge(V, F, 100)

# select a few random vertices (using the same seed)
rng = np.random.default_rng(0)
si = np.unique(rng.integers(0, V.shape[0], size=(nv,)))

# get the sample values
sv = u_gt[si]

# optimise each of them
hel = gp.halfedge_lengths(V, F)
u_vfef = hessian_l1_solve(hel, F, y=si, k=sv, mode="vfef")
u_stein_et_al_2018 = hessian_l1_solve(hel, F, y=si, k=sv, mode="stein et al 2018", V=V)
u_lapl2 = gp.min_quad_with_fixed(gp.biharmonic_energy_intrinsic(hel**2, F), k=si, y=sv)
try:
    u_stein_hess2 = gp.min_quad_with_fixed(gp.biharmonic_energy_intrinsic(hel**2, F, bc='curved_hessian'), k=si, y=sv)
except AssertionError:
    print("Gpytoolbox C++ bindings for bc='curved_hessian' likely not accessible to conda.  Setting L2 Hessian energy to 0.")
    u_stein_hess2 = np.zeros_like(u_lapl2)

mesh = ps.register_surface_mesh("test mesh", V, F, enabled=True)
ps.register_point_cloud("sampled points", V[si, :], radius=0.002, enabled=True, color=(1,0,0))

# show all of the energies
mesh.add_scalar_quantity("Ground truth function", u_gt, defined_on='vertices', isolines_enabled=True)
mesh.add_scalar_quantity("Our intrinsic L1 Hessian (Rowe et al 2024)", u_vfef, defined_on='vertices', isolines_enabled=True)
mesh.add_scalar_quantity("Extrinsic L1 Hessian (Stein et al 2018)", u_stein_et_al_2018, defined_on='vertices', isolines_enabled=True)
mesh.add_scalar_quantity("L2 Laplacian energy", u_lapl2, defined_on='vertices', isolines_enabled=True)
mesh.add_scalar_quantity("L2 (curved) Hessian energy (Stein et al 2020)", u_stein_hess2, defined_on='vertices', isolines_enabled=True)

ps.show()