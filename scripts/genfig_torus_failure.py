import sys
sys.path.insert(0, '../src/')

from l1hessian import hessian_l1_solve
import gpytoolbox as gp
import numpy as np
import blendertoolbox as bt
import bpy
import os
from igl import heat_geodesic

regen_np = True if int(sys.argv[1]) == 1 else False
quick = True if int(sys.argv[2]) == 1 else False

def render_torus(V, F, outpath, numpy_scalar, si, cmapname, vminmax=None):
    
    tx = "../results/pngs/" + cmapname + "_isolinemap60.png"
    tx_grey = "../results/pngs/Greys_r_isolinemap60.png"
    
    res = [720, 720] if quick else [1080, 1080] # recommend >1080 for paper figures
    num_samples = 200 # recommend >200 for paper figures
    meshpos = (2.16, -0.03, 0.49) # UI: click mesh > Transform > Location
    meshrot = (90, 0, 0) # UI: click mesh > Transform > Rotation
    meshscale = (3, 3, 3) # UI: click mesh > Transform > Scale
    lightangle = (6, -30, -155) # UI: click Sun > Transform > Rotation

    bt.blenderInit(res[0], res[1], num_samples, 1.5)

    bt.invisibleGround(shadowBrightness=0.9)

    sif = np.nonzero(np.all(np.isin(F, si), axis=1))[0]
    nsif = np.setdiff1d(np.arange(F.shape[0]), sif)
    
    Vsi, Fsi, Isi, Jsi = gp.remove_unreferenced(V, F[sif, :], return_maps=True)
    # add two vertices at infinity to set vmin, vmax
    Vsi = np.vstack([Vsi, np.array([[1000, 0, 0],
                                    [0, 1000, 0]])])
    mesh = bt.readNumpyMesh(Vsi, Fsi, meshpos, meshrot, meshscale)
    mesh = bt.vertexScalarToUV(mesh, np.clip(np.concatenate([numpy_scalar[Jsi], np.array(vminmax)]), vminmax[0], vminmax[1])) # using bt vertexscalartoUV
    meshColor = bt.colorObj((0,0,0,1), 0.5, 1.3, 1.0, 0.0, 0.4)
    bt.setMat_texture(mesh, tx_grey, meshColor)
    
    Vnsi, Fnsi, Insi, Jnsi = gp.remove_unreferenced(V, F[nsif, :], return_maps=True)
    # add two vertices at infinity to set vmin, vmax
    Vnsi = np.vstack([Vnsi, np.array([[1000, 0, 0],
                                      [0, 1000, 0]])])
    mesh2 = bt.readNumpyMesh(Vnsi, Fnsi, meshpos, meshrot, meshscale)
    mesh2 = bt.vertexScalarToUV(mesh2, np.clip(np.concatenate([numpy_scalar[Jnsi], np.array(vminmax)]), vminmax[0], vminmax[1])) # using bt vertexscalartoUV
    meshColor2 = bt.colorObj((0,0,0,1), 0.5, 1.3, 1.0, 0.0, 0.4)
    bt.setMat_texture(mesh2, tx, meshColor2)            
    
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
outpng_gt = "../results/pngs/torusfailure_gt.png"
outpng_ours = "../results/pngs/torusfailure_ours.png"
outnpy = "../results/npys/torusfailure.npy"

# tunable params
hesstype = "vfef"
mu = 100
eps = 1e-12
levels = 100
cmaps = ["Greys_r", "YlOrRd_r", "PuRd_r", "YlGnBu_r"]

modelname = "torus"

modelfilename = "../models/torus.obj"

V, F = gp.read_mesh(modelfilename)
V, F = gp.subdivide(V, F, method='loop', iters=1)

V = V - np.mean(V, axis=0, keepdims=True)
V = V / np.amax(np.linalg.norm(V, axis=1))

# get hel
hel = gp.halfedge_lengths(V, F)
hel /= np.amax(hel)

if regen_np:
    orig_fn = heat_ridge_alpha(V, F, np.array([0]), 0.85) # previously
    
    si = np.nonzero(np.linalg.norm(V - np.array([[-0.6, 0.09, 0]]), axis=1) > 0.2)[0]
    
    sv = orig_fn[si]
    
    recovered_fn = hessian_l1_solve(hel, F, mode=hesstype, y=si, k=sv)
    
    with open(outnpy, 'wb') as f:
        np.save(f, si)
        np.save(f, orig_fn)
        np.save(f, recovered_fn)

with open(outnpy, 'rb') as f:
    si, orig_fn, recovered_fn = np.load(f), np.load(f), np.load(f)

vmm = (0.51, 1)
render_torus(V, F, outpng_gt, orig_fn, si, cmaps[0], vminmax=vmm)
render_torus(V, F, outpng_ours, recovered_fn, si, cmaps[3], vminmax=vmm)