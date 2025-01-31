import sys
sys.path.insert(0, '../src/')

from l1hessian import hessian_l1_solve
import gpytoolbox as gp
import numpy as np
import scipy.sparse as sp
import os
import blendertoolbox as bt
import bpy
import time

regen = True if int(sys.argv[1]) == 1 else False
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

class colorObj(object):
    def __init__(self, RGBA, \
    H = 0.5, S = 1.0, V = 1.0,\
    B = 0.0, C = 0.0):
        self.H = H # hue
        self.S = S # saturation
        self.V = V # value
        self.RGBA = RGBA
        self.B = B # birghtness
        self.C = C # contrast

def render_plane(V, F, outpath, numpy_scalar, si, cmapname, vminmax=None):
    
    tx = "../results/pngs/" + cmapname + "_isolinemap30.png"
    
    res = [720, 720] if quick else [1080, 1080] # recommend >1080 for paper figures
    num_samples = 200 # recommend >200 for paper figures
    meshpos = (-0.42, -0.12, 0.02) # UI: click mesh > Transform > Location
    meshrot = (0, 0, -39) # UI: click mesh > Transform > Rotation
    meshscale = (0.2, 0.2, 0.2) # UI: click mesh > Transform > Scale
    lightangle = (6, -30, -155) # UI: click Sun > Transform > Rotation

    bt.blenderInit(res[0], res[1], num_samples, 1.5)

    bt.invisibleGround(shadowBrightness=0.9)

    mesh = bt.readNumpyMesh(V, F, meshpos, meshrot, meshscale)
    
    # mapping of the thing
    print("nans:")
    print(np.count_nonzero(np.isnan(numpy_scalar)))
    print("min, max:")
    print(np.amin(numpy_scalar), np.amax(numpy_scalar))
    
    # only use vminmax for ours
    if cmapname == "YlGnBu_r":
        mesh = vertexScalarToUV(mesh, numpy_scalar, vminmax) # using custom vertexscalartoUV
    else:
        mesh = bt.vertexScalarToUV(mesh, numpy_scalar) # using bt vertexscalartoUV
    
    meshColor = bt.colorObj((0,0,0,1), 0.5, 1.3, 1.0, 0.0, 0.4)
    bt.setMat_texture(mesh, tx, meshColor)
    
    # render the pointcloud too
    ptcloud = bt.readNumpyPoints(V[si, :],meshpos,meshrot,meshscale)
    
    ptColor = bt.colorObj((1, 0, 0, 1), 0.5, 1.3, 1.0, 0.0, 0.0)
    ptSize = 0.1
    bt.setMat_pointCloud(ptcloud, ptColor, ptSize)
    
    bpy.ops.object.shade_flat() 
    
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

    C = np.copy(vertex_scalars.flatten())
    vminmax = (C.min(), C.max()+1e-16) if (vminmax is None) else vminmax
    if vminmax is None:
        vminmax = vminmax if np.abs(vminmax[1]-vminmax[0]) > 1e-10 else (-1, 1)
    C = (C-vminmax[0])/np.abs(vminmax[1]-vminmax[0])

    for face in mesh.polygons:
        for vIdx, loopIdx in zip(face.vertices, face.loop_indices):
            uv_layer.data[loopIdx].uv = (C[vIdx], 0)
    return mesh_obj

# mesh choices

hesstype = "vfef"
k = 8
eps = 1e-10
cmaps = ["Greys_r", "YlOrRd_r", "PuRd_r", "YlGnBu_r"]

modelname = "plane_holes"
modelfilename = "../models/plane_holes.obj"

sample_model = "../models/plane.obj"

Vsamp, Fsamp = gp.read_mesh(sample_model)

V, F = gp.read_mesh(modelfilename)
V, F = gp.subdivide(V, F, method='loop')
print(str(V.shape[0]) + " vertices")

# divide by max hel
hel = gp.halfedge_lengths(V, F)
hel /= np.amax(gp.massmatrix(V, F).data)

outpng_gt = "../results/pngs/plane_interp_gt.png"
outpng_lapl2 = "../results/pngs/plane_interp_lapl2.png"
outpng_steinetal2018 = "../results/pngs/plane_interp_steinetal2018.png"
outpng_ours = "../results/pngs/plane_interp_ours.png"
outnpy = "../results/npys/plane_interp.npy"

f = -np.abs(np.sum(V * np.array([[0, 1, 0]]), axis=1))

si = np.array([18982, 3908, 13219, 33729])
sv = np.array([1, 1, -1, -1])

if regen:
    st = time.time()
    recovered_fn = hessian_l1_solve(hel, F, mode=hesstype, y=si, k=sv)
    et = time.time()
    print("ours", et-st)
    st = time.time()
    lapl2_fn = gp.min_quad_with_fixed(gp.biharmonic_energy_intrinsic(hel**2, F, bc='curved_hessian'), k=si, y=sv) # bc='curved_hessian'?
    et = time.time()
    print("L2", et-st)

    with open(outnpy, 'wb') as fi:
        np.save(fi, recovered_fn)
        np.save(fi, lapl2_fn)

with open(outnpy, 'rb') as fi:
    recovered_fn = np.load(fi)
    lapl2_fn = np.load(fi)

vmm = (-1.1,1.1)
render_plane(V, F, outpng_gt, f, si, cmaps[0], vminmax=vmm)
render_plane(V, F, outpng_lapl2, lapl2_fn, si, cmaps[1], vminmax=vmm)
render_plane(V, F, outpng_ours, recovered_fn, si, cmaps[3], vminmax=vmm)