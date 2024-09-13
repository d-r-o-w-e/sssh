# Triceratops holefilling
import sys
sys.path.insert(0, '../src/')

from l1hessian import hessian_l1_solve
import gpytoolbox as gp
import numpy as np
import blendertoolbox as bt
import bpy
import os
from igl import heat_geodesic
from matplotlib import colormaps
import time

regen_np = True if int(sys.argv[1]) == 1 else False
quick = True if int(sys.argv[2]) == 1 else False

def binsearch(gt, solve_alpha, minbound, maxbound, ary=5, tol=1e-3):
    
    currmin, currmax = minbound, maxbound
    
    while (currmax-currmin) > tol:
        print(currmin, currmax)
        guesses = np.linspace(currmin,currmax, ary, endpoint=True)
        losses = np.array([np.sum((gt-solve_alpha(guesses[i]))**2) for i in range(ary)])
        # print(losses)
        i_best = np.argmin(losses)
        currmin, currmax = np.clip(guesses[i_best-1], minbound, maxbound), np.clip(guesses[i_best+1], minbound, maxbound)
    
    print("final loss:", str(np.sum((gt-solve_alpha((currmax+currmin)/2.0))**2)))
    
    return (currmax+currmin)/2.0

def render_cherries(V, F, outpath, numpy_scalar, cmapname, vminmax=None):
    tx = "../results/pngs/" + cmapname + "_isolinemap40.png"
    
    res = [720, 720] if quick else [2160, 2160] # recommend >1080 for paper figures
    num_samples = 500 # recommend >200 for paper figures
    meshpos = (0.56, -0.11, 0.59) # UI: click mesh > Transform > Location
    meshrot = (68, 0, 67) # UI: click mesh > Transform > Rotation
    meshscale = (1.1, 1.1, 1.1) # UI: click mesh > Transform > Scale
    lightangle = (6, -30, -155) # UI: click Sun > Transform > Rotation

    bt.blenderInit(res[0], res[1], num_samples, 1.5)

    bt.invisibleGround(shadowBrightness=0.9)

    # dummy points for color selection
    if not (vminmax is None):
        V = np.vstack([V, np.array([[10000, 0, 0],
                                    [10000, 0, 0]])])
        numpy_scalar = np.concatenate([numpy_scalar, np.array([vminmax[0], vminmax[1]])])
    
    mesh = bt.readNumpyMesh(V, F, meshpos, meshrot, meshscale)
    
    # mapping of the thing
    mesh = bt.vertexScalarToUV(mesh, numpy_scalar) # using bt vertexscalartoUV
    
    # set the colors without isolines if its the noisy version
    if outpath[-9:-4] == "noisy":
        # vscalars = (V[:, 2]-V[:, 2].min())/(V[:, 2].max()-V[:, 2].min())
        cmap = colormaps[cmapname]
        vertex_colors = cmap(numpy_scalar)
        color_type = 'vertex'
        mesh = bt.setMeshColors(mesh, vertex_colors, color_type)
        meshVColor = bt.colorObj([], 0.5, 1.0, 1.0, 0.0, 0.0)
        bt.setMat_VColor(mesh, meshVColor)
    else:
        meshColor = bt.colorObj((0,0,0,1), 0.5, 1.3, 1.0, 0.0, 0.4)
        bt.setMat_texture(mesh, tx, meshColor)
    
    bpy.ops.object.shade_smooth() 
    
    camLocation = (3, 0, 2)
    lookAtLocation = (0,0,0.5)
    focalLength = 45 # (UI: click camera > Object Data > Focal Length)
    cam = bt.setCamera(camLocation, lookAtLocation, focalLength)
    
    strength = 2
    shadowSoftness = 0.3
    sun = bt.setLight_sun(lightangle, strength, shadowSoftness)
    
    bt.setLight_ambient(color=(0.3,0.3,0.3,1))
    
    bpy.ops.wm.save_mainfile(filepath=os.getcwd() + '/test.blend')
    
    bt.renderImage(outpath, cam)

def vertexScalarToUV(mesh_obj, vertex_scalars, vminmax=None):
    """
    This function takes a vertex scalar data and set to vertex UV (useful for render isoline)

    Inputs
    mesh_obj: bpy.object of the mesh
    C: |V| numpy array of vertex scalars

    Outputs
    mesh_obj
    """
    mesh = mesh_obj.data
    nV = len(mesh.vertices)
    nC = len(vertex_scalars.flatten())

    # guess the type of colors
    if nC != nV:
        raise ValueError('Error in "vertexScalarToUV": input color format must be eithe |V| array of vertex colors')

    uv_layer = mesh.uv_layers.new(name="funcUV")

    if vminmax is None:
        return bt.vertexScalarToUV(mesh_obj, vertex_scalars)
    else:
        C = np.clip(np.copy(vertex_scalars.flatten()), vminmax[0], vminmax[1])
        C -= vminmax[0]
        C /= (vminmax[1]+1e-16)

    for face in mesh.polygons:
        for vIdx, loopIdx in zip(face.vertices, face.loop_indices):
            uv_layer.data[loopIdx].uv = (C[vIdx], 0)
    return mesh_obj

def heat_distance(V, F, si):
    # just call igl
    u = heat_geodesic(V, F, t=np.mean(gp.halfedge_lengths(V, F))**2, gamma=np.array(si, dtype=F.dtype))
    u = (u - np.amin(u)) / (np.amax(u)-np.amin(u))
    return u

heat_ridge = lambda V, F, si: np.where(heat_distance(V, F, si) < 0.5, 2*heat_distance(V, F, si), 2-2*heat_distance(V, F, si))
heat_ridge_alpha = lambda V, F, si, alpha: np.where(heat_distance(V, F, si) < alpha, heat_distance(V, F, si), 2*alpha-heat_distance(V, F, si))

# file locations
outpng_gt = "../results/pngs/cherries_fndenoise_gt_repimage.png"
outpng_noisy = "../results/pngs/cherries_fndenoise_noisy_repimage.png"
outpng_lapl2 = "../results/pngs/cherries_fndenoise_lapl2_repimage.png"
outpng_steinetal2018 = "../results/pngs/cherries_fndenoise_steinetal2018_repimage.png"
outpng_ours = "../results/pngs/cherries_fndenoise_ours_repimage.png"
outnpy = "../results/npys/cherries_fndenoise_repimage.npy"

# tunable params
hesstype = "vfef"
mu = 100
eps = 1e-12
levels = 50
cmaps = ["Greys_r", "YlOrRd_r", "PuRd_r", "YlGnBu_r"]
re_search = False

modelname = "cherries"

modelfilename = "../models/cherries1.obj"

V, F = gp.read_mesh(modelfilename)

V = V - np.mean(V, axis=0, keepdims=True)
V = V / np.amax(np.linalg.norm(V, axis=1))

# get hel
hel = gp.halfedge_lengths(V, F)
hel /= np.amax(hel)

rng = np.random.default_rng(0)

if regen_np:
    orig_fn = heat_ridge_alpha(V, F, np.array([253]), 0.7)
    
    u0 = orig_fn + 0.1*rng.uniform(-1, 1, size=V.shape[0])
    alpha_ours = 6.1479
    alpha_lapl2 = 0.01452
    alpha_stein = 16.17
    
    if re_search:
        print("alpha")
        alpha = binsearch(orig_fn, lambda alpha: hessian_l1_solve(hel, F, alpha=alpha, u0=u0, mode=hesstype), minbound=0, maxbound=1000)
        print("found best alpha: " + str(alpha))
        print("alpha1")
        alpha1 = binsearch(orig_fn, lambda alpha1: gp.min_quad_with_fixed(Q=2*(gp.biharmonic_energy_intrinsic(hel**2, F, bc="curved_hessian") + alpha1*gp.massmatrix_intrinsic(hel**2, F)), 
                                                                              c=-2*alpha1*(gp.massmatrix_intrinsic(hel**2, F) @ u0[:, None]))[:, 0], 0, 10)
        print("found best alpha1: " + str(alpha1))
        print("alpha2")
        alpha2 = binsearch(orig_fn, lambda alpha2: hessian_l1_solve(hel, F, alpha=alpha2, u0=u0, mode="stein et al 2018", V=V), 0.1, 200)
        print("found best alpha2: " + str(alpha2))
    
    print("our solve")
    st = time.time()
    recovered_fn = hessian_l1_solve(hel, F, alpha=alpha_ours, u0=u0, mode=hesstype)
    et = time.time()
    print("ours: " + str(et - st))
    print("lapl2 solve")
    st = time.time()
    lapl2_fn = gp.min_quad_with_fixed(Q=2*(gp.biharmonic_energy_intrinsic(hel**2, F, bc="curved_hessian") + alpha_lapl2*gp.massmatrix_intrinsic(hel**2, F)), 
                                      c=-2*alpha_lapl2*(gp.massmatrix_intrinsic(hel**2, F) @ u0[:, None]))[:, 0]
    et = time.time()
    print("lapl2: " + str(et - st))
    print("stein et al solve")
    st = time.time()
    steinetal2018_fn = hessian_l1_solve(hel, F, alpha=alpha_stein, u0=u0, mode="stein et al 2018", V=V)
    et = time.time()
    print("stein: " + str(et - st))
    
    with open(outnpy, 'wb') as f:
        np.save(f, orig_fn)
        np.save(f, u0)
        np.save(f, lapl2_fn)
        np.save(f, steinetal2018_fn)
        np.save(f, recovered_fn)

with open(outnpy, 'rb') as f:
    orig_fn, u0, lapl2_fn, steinetal2018_fn, recovered_fn = np.load(f), np.load(f), np.load(f), np.load(f), np.load(f)

vmm = (orig_fn.min(), orig_fn.max())
render_cherries(V, F, outpng_gt, orig_fn, cmaps[0], vminmax=vmm)
render_cherries(V, F, outpng_noisy, u0, cmaps[0], vminmax=vmm)
render_cherries(V, F, outpng_lapl2, lapl2_fn, cmaps[1], vminmax=vmm)
render_cherries(V, F, outpng_steinetal2018, steinetal2018_fn, cmaps[2], vminmax=vmm)
render_cherries(V, F, outpng_ours, recovered_fn, cmaps[3], vminmax=vmm)