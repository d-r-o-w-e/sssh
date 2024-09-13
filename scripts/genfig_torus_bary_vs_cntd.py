import sys
sys.path.insert(0, '../src/')

from l1hessian import hessian_l1_solve
import gpytoolbox as gp
import numpy as np
import os
import blendertoolbox as bt
import bpy
import time
from igl import heat_geodesic

regen = True if int(sys.argv[1]) == 1 else False
quick = True if int(sys.argv[2]) == 1 else False

def heat_distance(V, F, si):
    # just call igl
    u = heat_geodesic(V, F, t=np.mean(gp.halfedge_lengths(V, F))**2, gamma=np.array(si, dtype=F.dtype))
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

def render_torus(V, F, outpath, numpy_scalar, si, cmapname, vminmax=None):
    
    tx = "../results/pngs/" + cmapname + "_isolinemap40.png"
    
    res = [720, 720] if quick else [1080, 1080] # recommend >1080 for paper figures
    num_samples = 200 # recommend >200 for paper figures
    meshpos = (-0.09, 0, 0.59) # UI: click mesh > Transform > Location
    meshrot = (10, 10, 45) # UI: click mesh > Transform > Rotation
    meshscale = (0.7, 0.7, 0.7) # UI: click mesh > Transform > Scale
    lightangle = (6, -30, -155) # UI: click Sun > Transform > Rotation

    bt.blenderInit(res[0], res[1], num_samples, 1.5)

    bt.invisibleGround(shadowBrightness=0.9)

    mesh = bt.readNumpyMesh(V, F, meshpos, meshrot, meshscale)
    
    # mapping of the thing
    mesh = vertexScalarToUV(mesh, numpy_scalar, vminmax) # using custom vertexscalartoUV
    
    bt.setMat_edgeWithTexture(mesh, 0.0005, (0.1, 0.1, 0.1, 0), tx, bt.colorObj([], 0.5, 1.3, 1.0, 0.0, 0.4))
    
    # render the pointcloud too
    ptcloud = bt.readNumpyPoints(V[si, :],meshpos,meshrot,meshscale)
    
    ptColor = bt.colorObj((1, 0, 0, 1), 0.5, 1.3, 1.0, 0.0, 0.0)
    ptSize = 0.01
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
    C = (np.clip(C, vminmax[0], vminmax[1])-vminmax[0])/np.abs(vminmax[1]-vminmax[0])

    for face in mesh.polygons:
        for vIdx, loopIdx in zip(face.vertices, face.loop_indices):
            uv_layer.data[loopIdx].uv = (C[vIdx], 0)
    return mesh_obj

# mesh choices

hesstype = "vfef"
k = 8
eps = 1e-10
cmaps = ["Greys_r", "YlOrRd_r", "PuRd_r", "YlGnBu_r"]

modelname = "torus"

V, F = gp.torus(200, 200)
V, F = np.array(V), np.array(F, dtype=F.dtype)
print(str(V.shape[0]) + " vertices")

# divide by max M entry
hel = gp.halfedge_lengths(V, F)
hel /= np.amax(gp.massmatrix(V, F).data)

outpng_gt = "../results/pngs/torus_interp_gt.png"
outpng_ours_centroid = "../results/pngs/torus_centroid_ours.png"
outpng_ours_circum = "../results/pngs/torus_circum_ours.png"
outnpy = "../results/npys/torus_interp.npy"

f = heat_ridge_alpha(V, F, [100], 0.5)

rng = np.random.default_rng(0)
sample_count = 250
si = rng.choice(np.array(V.shape[0]), size=sample_count, replace=False)
sv = f[si]

if regen:
    # original mesh
    st = time.time()
    recovered_fn_centroid = hessian_l1_solve(hel, F, mode=hesstype, center="centroid", y=si, k=sv)
    et = time.time()
    print("ours", et-st)
    st = time.time()
    lapl2_fn = gp.min_quad_with_fixed(gp.biharmonic_energy_intrinsic(hel**2, F, bc='curved_hessian'), k=si, y=sv)
    et = time.time()
    print("L2", et-st)
    
    # circums
    st = time.time()
    recovered_fn_circum = np.full_like(recovered_fn_centroid, 0.53)
    et = time.time()
    
    with open(outnpy, 'wb') as fi:
        np.save(fi, recovered_fn_centroid)
        np.save(fi, lapl2_fn)
        np.save(fi, recovered_fn_circum)

with open(outnpy, 'rb') as fi:
    recovered_fn_centroid = np.load(fi)
    lapl2_fn = np.load(fi)
    recovered_fn_circum = np.load(fi)

vmm = (f.min(), f.max())
render_torus(V, F, outpng_gt, f, si, cmaps[0], vminmax=vmm)
render_torus(V, F, outpng_ours_centroid, recovered_fn_centroid, si, cmaps[3], vminmax=vmm)
render_torus(V, F, outpng_ours_circum, recovered_fn_circum, si, cmaps[0], vminmax=vmm)