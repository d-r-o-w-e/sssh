import sys
sys.path.insert(0, '../src/')

from l1hessian import hessian_l1_solve, vfef, hessian_l1_modes
import gpytoolbox as gp
import numpy as np
import scipy.sparse as sp
import os
import blendertoolbox as bt
import bpy
import time
from igl import heat_geodesic, intrinsic_delaunay_triangulation, edge_lengths

regen = True if int(sys.argv[1]) == 1 else False
quick = True if int(sys.argv[2]) == 1 else False

def heat_distance(V, F, si):
    # compute geodesic distance from points si.  based on keenan's paper
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

def heat_distance_intrinsic(hel, F, si):
    # compute geodesic distance from points si
    hel, F = intrinsic_delaunay_triangulation(hel, F)
    
    vshape = np.amax(F)+1
    
    t = np.mean(hel)**2
    M, L = gp.massmatrix_intrinsic(hel**2, F), -gp.cotangent_laplacian_intrinsic(hel**2, F)
    mask = np.zeros((vshape, 1))
    mask[si] = 1
    heatu = sp.linalg.spsolve(M-t*L, mask)[:, None]
    G = gp.grad_intrinsic(hel**2, F)
    vecs = G @ heatu
    vecs = vecs / np.tile(np.linalg.norm(np.reshape(vecs, (F.shape[0], 2), order="F"), axis=1, keepdims=True), reps=(2, 1))
    Mx = 0.5*sp.diags(np.hstack([gp.doublearea_intrinsic(hel**2, F), gp.doublearea_intrinsic(hel**2, F)]))
    phi = G.T @ Mx @ vecs
    
    u = sp.linalg.spsolve(L, phi)
    u = (u - np.amin(u)) / (np.amax(u)-np.amin(u))
    
    return u

def heat_distance(V, F, si):
    # just call igl.  at this size mesh it's unstable
    # print(np.array(si, dtype=F.dtype))
    hel = gp.halfedge_lengths(V, F)
    u = heat_geodesic(V, F, t=np.mean(hel)**2, gamma=np.array(si, dtype=F.dtype))
    u = (u - np.amin(u)) / (np.amax(u)-np.amin(u))
    return u

heat_ridge = lambda V, F, si: np.where(heat_distance(V, F, si) < 0.5, 2*heat_distance(V, F, si), 2-2*heat_distance(V, F, si))
heat_ridge_alpha = lambda V, F, si, alpha: np.where(heat_distance(V, F, si) < alpha, heat_distance(V, F, si), 2*alpha-heat_distance(V, F, si))
heat_ridge_intrinsic = lambda hel, F, si: np.where(heat_distance_intrinsic(hel, F, si) < 0.5, 2*heat_distance_intrinsic(hel, F, si), 2-2*heat_distance_intrinsic(hel, F, si))
heat_ridge_alpha_intrinsic = lambda hel, F, si, alpha: np.where(heat_distance_intrinsic(hel, F, si) < alpha, heat_distance_intrinsic(hel, F, si), 2*alpha-heat_distance_intrinsic(hel, F, si))

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

def render_rocket(V, F, outpath, numpy_scalar, si, cmapname, vminmax=None):
    
    tx = "../results/pngs/" + cmapname + "_isolinemap40.png"
    
    res = [720, 720] if quick else [2160, 2160] # recommend >1080 for paper figures
    num_samples = 200 # recommend >200 for paper figures
    meshpos = (0.73, 0.02, 0.1) # UI: click mesh > Transform > Location
    meshrot = (90, 0, -19) # UI: click mesh > Transform > Rotation
    meshscale = (0.09, 0.09, 0.09) # UI: click mesh > Transform > Scale
    lightangle = (6, -30, -155) # UI: click Sun > Transform > Rotation

    bt.blenderInit(res[0], res[1], num_samples, 1.5)

    bt.invisibleGround(shadowBrightness=0.9)

    mesh = bt.readNumpyMesh(V, F, meshpos, meshrot, meshscale)
    
    # mapping of the thing
    mesh = vertexScalarToUV(mesh, numpy_scalar, vminmax) # using my own vertexscalartoUV
    
    meshColor = bt.colorObj((0,0,0,1), 0.5, 1.3, 1.0, 0.0, 0.4)
    setMat_texture_AO(mesh, tx, meshColor, AOStrength=100)
    
    # render the pointcloud too
    ptcloud = bt.readNumpyPoints(V[si, :],meshpos,meshrot,meshscale)
    
    ptColor = bt.colorObj((1, 0, 0, 1), 0.5, 1.3, 1.0, 0.0, 0.0)
    ptSize = 0.06
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

def render_w_edges(V, F, outpath, numpy_scalar, si, cmapname, vminmax=None):
    
    tx = "../results/pngs/" + cmapname + "_isolinemap40.png"
    
    res = [720, 720] if quick else [1080, 1080] # recommend >1080 for paper figures
    num_samples = 200 # recommend >200 for paper figures
    meshpos = (0.85, 0.25, -1.41) # UI: click mesh > Transform > Location
    meshrot = (90, 0, -19) # UI: click mesh > Transform > Rotation
    meshscale = (0.4, 0.4, 0.4) # UI: click mesh > Transform > Scale
    lightangle = (6, -30, -155) # UI: click Sun > Transform > Rotation

    bt.blenderInit(res[0], res[1], num_samples, 1.5)

    bt.invisibleGround(shadowBrightness=0.9)

    mesh = bt.readNumpyMesh(V, F, meshpos, meshrot, meshscale)
    
    # mapping of the thing
    mesh = vertexScalarToUV(mesh, numpy_scalar, vminmax) # using custom vertexscalartoUV
    
    setMat_edgeWithTexture_AO(mesh, 0.002, (0.1, 0.1, 0.1, 0), tx, bt.colorObj([], 0.5, 1.3, 1.0, 0.0, 0.4), AOStrength=100)
    
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
    C = (C-vminmax[0])/np.abs(vminmax[1]-vminmax[0])

    for face in mesh.polygons:
        for vIdx, loopIdx in zip(face.vertices, face.loop_indices):
            uv_layer.data[loopIdx].uv = (C[vIdx], 0)
    return mesh_obj

def setMat_edgeWithTexture_AO(mesh, edgeThickness, edgeRGBA, texturePath, textureHSVBC, AOStrength=10):
    # set mat edge with texture with ao parameter 
    
    meshRGBA = (1,1,1,0)

    # initialize material node graph
    mat = bpy.data.materials.new('MeshMaterial')
    mesh.data.materials.append(mat)
    mesh.active_material = mat
    mat.use_nodes = True
    tree = mat.node_tree

    # add edge wire rendering
    WIRE_RGB = tree.nodes.new('ShaderNodeRGB')
    WIRE_RGB.outputs[0].default_value = edgeRGBA
    WIRE_RGB.location.x -= 200
    WIRE_RGB.location.y -= 400

    WIRE = tree.nodes.new(type="ShaderNodeWireframe")
    WIRE.inputs[0].default_value = edgeThickness
    WIRE.location.x -= 200
    WIRE.location.y += 200

    MIX = tree.nodes.new('ShaderNodeMixRGB')
    MIX.blend_type = 'MIX'
	
    # add texture
    TI = tree.nodes.new('ShaderNodeTexImage')
    absTexturePath = os.path.abspath(texturePath)
    TI.image = bpy.data.images.load(absTexturePath)
    TI.location.x -= 700

    # set color using Hue/Saturation node
    HSVNode = tree.nodes.new('ShaderNodeHueSaturation')
    HSVNode.inputs['Saturation'].default_value = textureHSVBC.S
    HSVNode.inputs['Value'].default_value = textureHSVBC.V
    HSVNode.inputs['Hue'].default_value = textureHSVBC.H
    HSVNode.location.x -= 400

    # set color brightness/contrast
    BCNode = tree.nodes.new('ShaderNodeBrightContrast')
    BCNode.inputs['Bright'].default_value = textureHSVBC.B
    BCNode.inputs['Contrast'].default_value = textureHSVBC.C
    BCNode.location.x -= 200

	# set principled BSDF
    PRI = tree.nodes["Principled BSDF"]
    PRI.inputs['Roughness'].default_value = 0.3
    PRI.inputs['Sheen Tint'].default_value = [0, 0, 0, 1]
    PRI.inputs['Specular IOR Level'].default_value = 0.2
    PRI.inputs['IOR'].default_value = 1.45
    PRI.inputs['Transmission Weight'].default_value = 0
    PRI.inputs['Coat Roughness'].default_value = 0
    # AO
    # add Ambient Occlusion
    tree.nodes.new('ShaderNodeAmbientOcclusion')
    tree.nodes.new('ShaderNodeGamma')
    tree.nodes["Gamma"].inputs["Gamma"].default_value = AOStrength
    tree.nodes["Ambient Occlusion"].inputs["Distance"].default_value = 10.0
    tree.nodes["Gamma"].location.x -= 600

    # link AO node
    tree.links.new(tree.nodes["Ambient Occlusion"].outputs['Color'], MIX.inputs['Color2'])
    tree.links.new(tree.nodes["Ambient Occlusion"].outputs['AO'], tree.nodes['Gamma'].inputs['Color'])
    tree.links.new(tree.nodes["Gamma"].outputs['Color'], MIX.inputs['Color1'])
    tree.links.new(BCNode.outputs['Color'], tree.nodes['Ambient Occlusion'].inputs['Color'])
    
	# link everything
    tree.links.new(BCNode.outputs['Color'], PRI.inputs['Base Color'])
    tree.links.new(TI.outputs['Color'], HSVNode.inputs['Color'])
    tree.links.new(HSVNode.outputs['Color'], BCNode.inputs['Color'])
    tree.links.new(BCNode.outputs['Color'], MIX.inputs[1])
    tree.links.new(WIRE.outputs[0], MIX.inputs[0])
    tree.links.new(WIRE_RGB.outputs[0], MIX.inputs[2])
    tree.links.new(MIX.outputs[0], PRI.inputs[0])
    tree.links.new(PRI.outputs[0], tree.nodes['Material Output'].inputs['Surface'])

def setMat_texture_AO(mesh, texturePath, meshColor, alpha= 1.0, colorspace_settting='sRGB', AOStrength=10):
    mat = bpy.data.materials.new('MeshMaterial')
    mesh.data.materials.append(mat)
    mesh.active_material = mat
    mat.use_nodes = True
    tree = mat.node_tree
    
    MIX = tree.nodes.new('ShaderNodeMixRGB')
    MIX.blend_type = 'MIX'

    # set principled BSDF
    PRI = tree.nodes["Principled BSDF"]
    PRI.inputs['Roughness'].default_value = 1.0
    PRI.inputs['Sheen Tint'].default_value = [0, 0, 0, 1]
    PRI.inputs['Alpha'].default_value = alpha

    TI = tree.nodes.new('ShaderNodeTexImage')
    absTexturePath = os.path.abspath(texturePath)
    TI.image = bpy.data.images.load(absTexturePath)
    TI.image.colorspace_settings.name = colorspace_settting

    # set color using Hue/Saturation node
    HSVNode = tree.nodes.new('ShaderNodeHueSaturation')
    HSVNode.inputs['Saturation'].default_value = meshColor.S
    HSVNode.inputs['Value'].default_value = meshColor.V
    HSVNode.inputs['Hue'].default_value = meshColor.H

    # set color brightness/contrast
    BCNode = tree.nodes.new('ShaderNodeBrightContrast')
    BCNode.inputs['Bright'].default_value = meshColor.B
    BCNode.inputs['Contrast'].default_value = meshColor.C

    # AO
    # add Ambient Occlusion
    tree.nodes.new('ShaderNodeAmbientOcclusion')
    tree.nodes.new('ShaderNodeGamma')
    tree.nodes["Gamma"].inputs["Gamma"].default_value = AOStrength
    tree.nodes["Ambient Occlusion"].inputs["Distance"].default_value = 10.0
    tree.nodes["Gamma"].location.x -= 600

    # link AO node
    tree.links.new(tree.nodes["Ambient Occlusion"].outputs['Color'], MIX.inputs['Color1'])
    tree.links.new(tree.nodes["Ambient Occlusion"].outputs['AO'], tree.nodes['Gamma'].inputs['Color'])
    tree.links.new(tree.nodes["Gamma"].outputs['Color'], MIX.inputs['Color2'])
    tree.links.new(BCNode.outputs['Color'], tree.nodes['Ambient Occlusion'].inputs['Color'])
    
    # link everything
    tree.links.new(BCNode.outputs['Color'], PRI.inputs['Base Color'])
    tree.links.new(TI.outputs['Color'], HSVNode.inputs['Color'])
    tree.links.new(HSVNode.outputs['Color'], BCNode.inputs['Color'])
    tree.links.new(BCNode.outputs['Color'], MIX.inputs[1])
    tree.links.new(MIX.outputs[0], PRI.inputs[0])
    tree.links.new(PRI.outputs[0], tree.nodes['Material Output'].inputs['Surface'])

# mesh choices

hesstype = "vfef"
k = 8
eps = 1e-10
cmaps = ["Greys_r", "YlOrRd_r", "PuRd_r", "YlGnBu_r"]

modelname = "rocketship"
modelfilename = "../models/rocket_worse.obj"

V, F = gp.read_mesh(modelfilename)
V, F = gp.subdivide(V, F, method='loop')
V1 = V.copy()
print(str(V.shape[0]) + " vertices")

hel = edge_lengths(V, F)
hel_int, F_int = intrinsic_delaunay_triangulation(hel, F)

outpng_gt = "../results/pngs/rocketship_interp_gt.png"
outpng_ours = "../results/pngs/rocketship_irreg_ours.png"
outpng_ours1 = "../results/pngs/rocketship_itri_ours.png"
outpng_gt_edges = "../results/pngs/rocketship_interp_gtzoom.png"
outnpy = "../results/npys/rocketship_interp.npy"

f = heat_ridge_alpha_intrinsic(hel_int, F_int, [205], 0.51)

rng = np.random.default_rng(0)
sample_count = 350
si = rng.choice(np.array(V.shape[0]), size=sample_count, replace=False)
sv = f[si]

if regen:
    # original mesh
    st = time.time()
    recovered_fn = hessian_l1_solve(hel, F, mode=hesstype, y=si, k=sv)
    et = time.time()
    print("ours", et-st)
    st = time.time()
    lapl2_fn = gp.min_quad_with_fixed(gp.biharmonic_energy_intrinsic(hel**2, F, bc='curved_hessian'), k=si, y=sv) # bc='curved_hessian'?
    et = time.time()
    print("L2", et-st)
    
    # intr_tri
    st = time.time()
    recovered_fn1 = hessian_l1_solve(hel_int, F_int, mode=hesstype, y=si, k=sv)
    et = time.time()
    print("ours", et-st)
    st = time.time()
    lapl2_fn1 = gp.min_quad_with_fixed(gp.biharmonic_energy_intrinsic(hel_int**2, F_int, bc='curved_hessian'), k=si, y=sv) # bc='curved_hessian'?
    et = time.time()
    print("L2", et-st)

    with open(outnpy, 'wb') as fi:
        np.save(fi, recovered_fn)
        np.save(fi, lapl2_fn)
        
        np.save(fi, recovered_fn1)
        np.save(fi, lapl2_fn1)

with open(outnpy, 'rb') as fi:
    recovered_fn = np.load(fi)
    lapl2_fn = np.load(fi)
    
    recovered_fn1 = np.load(fi)
    lapl2_fn1 = np.load(fi)

vmm = (f.min(), f.max())
render_rocket(V, F, outpng_gt, f, si, cmaps[0], vminmax=vmm)
render_rocket(V, F, outpng_ours, recovered_fn, si, cmaps[3], vminmax=vmm)
render_rocket(V, F, outpng_ours1, recovered_fn1, si, cmaps[3], vminmax=vmm)
render_w_edges(V, F, outpng_gt_edges, f, si, cmaps[0], vminmax=vmm)