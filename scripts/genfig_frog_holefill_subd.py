
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
import time

regen_np = True if int(sys.argv[1]) == 1 else False
quick = True if int(sys.argv[2]) == 1 else False

def render_frog(V, F, outpath, numpy_scalar, si, cmapname, vminmax=None):
    
    tx = "../results/pngs/" + cmapname + "_isolinemap60.png"
    tx_grey = "../results/pngs/Greys_r_isolinemap60.png"
    
    res = [720, 720] if quick else [1080, 1080] # recommend >1080 for paper figures
    num_samples = 200 # recommend >200 for paper figures
    meshpos = (-0.44, 0.1, 0.37) # UI: click mesh > Transform > Location
    meshrot = (90, 0, 203) # UI: click mesh > Transform > Rotation
    meshscale = (1.2, 1.2, 1.2) # UI: click mesh > Transform > Scale
    lightangle = (6, -30, -155) # UI: click Sun > Transform > Rotation

    bt.blenderInit(res[0], res[1], num_samples, 1.5)

    bt.invisibleGround(shadowBrightness=0.9)

    sif = np.nonzero(np.all(np.isin(F, si), axis=1))[0]
    nsif = np.setdiff1d(np.arange(F.shape[0]), sif)

    print(F.shape)
    print(sif.shape)
    print(nsif.shape)

    Vsi, Fsi, Isi, Jsi = gp.remove_unreferenced(V, F[sif, :], return_maps=True)
    # add two vertices at infinity to set vmin, vmax
    Vsi = np.vstack([Vsi, np.array([[1000, 0, 0],
                                    [0, 1000, 0]])])
    mesh = bt.readNumpyMesh(Vsi, Fsi, meshpos, meshrot, meshscale)
    mesh = bt.vertexScalarToUV(mesh, np.concatenate([numpy_scalar[Jsi], np.array(vminmax)])) # using bt vertexscalartoUV
    meshColor = bt.colorObj((0,0,0,1), 0.5, 1.3, 1.0, 0.0, 0.4)
    bt.setMat_edgeWithTexture(mesh, 0.001, (0.1, 0.1, 0.1, 0), tx_grey, bt.colorObj([], 0.5, 1.3, 1.0, 0.0, 0.4))
    
    Vnsi, Fnsi, Insi, Jnsi = gp.remove_unreferenced(V, F[nsif, :], return_maps=True)
    # add two vertices at infinity to set vmin, vmax
    Vnsi = np.vstack([Vnsi, np.array([[1000, 0, 0],
                                      [0, 1000, 0]])])
    mesh2 = bt.readNumpyMesh(Vnsi, Fnsi, meshpos, meshrot, meshscale)
    mesh2 = bt.vertexScalarToUV(mesh2, np.concatenate([numpy_scalar[Jnsi], np.array(vminmax)])) # using bt vertexscalartoUV
    meshColor2 = bt.colorObj((0,0,0,1), 0.5, 1.3, 1.0, 0.0, 0.4)
    bt.setMat_edgeWithTexture(mesh2, 0.001, (0.1, 0.1, 0.1, 0), tx, bt.colorObj([], 0.5, 1.3, 1.0, 0.0, 0.4))
    
    # render boundary as a pointcloud if the rest is grey
    if cmapname == "Greys_r":
        ptColor = bt.colorObj((1, 0, 0, 1), 0.5, 1.3, 1.0, 0.0, 0.0)
        bt.drawBoundaryLoop(mesh2, 0.003, ptColor)
    
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
outpng_gt = "../results/pngs/frog_fnholefill_gt.png"
outpng_ours = "../results/pngs/frog_fnholefill_ours_subd0.png"
outpng_ours1 = "../results/pngs/frog_fnholefill_ours_subd1.png"
outpng_ours2 = "../results/pngs/frog_fnholefill_ours_subd2.png"
outnpy = "../results/npys/frog_fnholefill.npy"

# tunable params
hesstype = "vfef"
mu = 100
eps = 1e-12
levels = 50
cmaps = ["Greys_r", "YlOrRd_r", "PuRd_r", "YlGnBu_r"] # "YlOrRd_r"

modelname = "frog"

modelfilename = "../models/frog_dec.obj"

V, F = gp.read_mesh(modelfilename)

V = V - np.mean(V, axis=0, keepdims=True)
V = V / np.amax(np.linalg.norm(V, axis=1))

# subdivided meshes
V1, F1 = gp.subdivide(V, F, method='loop', iters=1)
V2, F2 = gp.subdivide(V, F, method='loop', iters=2)

# get hel
hel = gp.halfedge_lengths(V, F)
hel1 = gp.halfedge_lengths(V1, F1)
hel2 = gp.halfedge_lengths(V2, F2)

if regen_np:
    fn2 = heat_ridge_alpha(V2, F2, np.array([0]), 0.45)
    fn1 = fn2[:V1.shape[0]]
    orig_fn = fn2[:V.shape[0]]
    
    get_si = lambda V: np.nonzero(np.logical_and.reduce([np.linalg.norm(V - np.array([[-0.24, 0.104, 0.087]]), axis=1) > 0.2, # side
                                                         np.linalg.norm(V - np.array([[-0.29, 0.29, -0.55]]), axis=1) > 0.2]))[0] # head
    
    si = get_si(V)
    si1 = get_si(V1)
    si2 = get_si(V2)
    
    sv = orig_fn[si]
    sv1 = fn1[si1]
    sv2 = fn2[si2]
    
    # original mesh
    st = time.time()
    recovered_fn = hessian_l1_solve(hel, F, mode=hesstype, y=si, k=sv)
    et = time.time()
    print("ours time: " + str(et - st))
    st = time.time()
    lapl2_fn = gp.min_quad_with_fixed(gp.biharmonic_energy_intrinsic(hel**2, F, bc="curved_hessian"), k=si, y=sv)
    et = time.time()
    print("lapl2 time: " + str(et - st))
    
    # first subd
    st = time.time()
    recovered_fn1 = hessian_l1_solve(hel1, F1, mode=hesstype, y=si1, k=sv1)
    et = time.time()
    print("ours time: " + str(et - st))
    st = time.time()
    lapl2_fn1 = gp.min_quad_with_fixed(gp.biharmonic_energy_intrinsic(hel1**2, F1, bc="curved_hessian"), k=si1, y=sv1)
    et = time.time()
    print("lapl2 time: " + str(et - st))
    
    # second subd
    st = time.time()
    recovered_fn2 = hessian_l1_solve(hel2, F2, mode=hesstype, y=si2, k=sv2)
    et = time.time()
    print("ours time: " + str(et - st))
    st = time.time()
    lapl2_fn2 = gp.min_quad_with_fixed(gp.biharmonic_energy_intrinsic(hel2**2, F2, bc="curved_hessian"), k=si2, y=sv2)
    et = time.time()
    print("lapl2 time: " + str(et - st))
    
    with open(outnpy, 'wb') as f:
        # original
        np.save(f, si)
        np.save(f, orig_fn)
        np.save(f, recovered_fn)
        
        # subd 1
        np.save(f, si1)
        np.save(f, fn1)
        np.save(f, recovered_fn1)
        
        # subd 2
        np.save(f, si2)
        np.save(f, fn2)
        np.save(f, recovered_fn2)

with open(outnpy, 'rb') as f:
    si, orig_fn, recovered_fn = np.load(f), np.load(f), np.load(f)
    si1, fn1, recovered_fn1 = np.load(f), np.load(f), np.load(f)
    si2, fn2, recovered_fn2 = np.load(f), np.load(f), np.load(f)

vmm = (fn2.min(), fn2.max())
render_frog(V2, F2, outpng_gt, fn2, si2, cmaps[0], vminmax=vmm)
render_frog(V, F, outpng_ours, recovered_fn, si, cmaps[3], vminmax=vmm)
render_frog(V1, F1, outpng_ours1, recovered_fn1, si1, cmaps[3], vminmax=vmm)
render_frog(V2, F2, outpng_ours2, recovered_fn2, si2, cmaps[3], vminmax=vmm)