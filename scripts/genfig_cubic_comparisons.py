import sys
sys.path.insert(0, '../src/')

from l1hessian import vfef
import gpytoolbox as gp
import numpy as np
import scipy.sparse as sp
import polyscope
import os
import blendertoolbox as bt
import bpy
from matplotlib import colormaps
from scipy.spatial.transform import Rotation
from igl import gaussian_curvature

regen = True if int(sys.argv[1]) == 1 else False
quick = True if int(sys.argv[2]) == 1 else False

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

def setMat_edge(mesh, \
				edgeThickness, \
				edgeColor, \
				meshColor = (0.7,0.7,0.7,1), \
				AOStrength = 1.0):
	mat = bpy.data.materials.new('MeshMaterial')
	mesh.data.materials.append(mat)
	mesh.active_material = mat
	mat.use_nodes = True
	tree = mat.node_tree

	# set principled BSDF
	tree.nodes["Principled BSDF"].inputs['Roughness'].default_value = 0.3
	tree.nodes["Principled BSDF"].inputs['Sheen Tint'].default_value = [0, 0, 0, 1]

	# add Ambient Occlusion
	tree.nodes.new('ShaderNodeAmbientOcclusion')
	tree.nodes.new('ShaderNodeGamma')
	MIXRGB = tree.nodes.new('ShaderNodeMixRGB')
	MIXRGB.blend_type = 'MULTIPLY'
	tree.nodes["Gamma"].inputs["Gamma"].default_value = AOStrength
	tree.nodes["Gamma"].location.x -= 600
	tree.nodes["Ambient Occlusion"].inputs["Distance"].default_value = 10.0
	tree.nodes["Ambient Occlusion"].inputs["Color"].default_value = meshColor
	tree.links.new(tree.nodes["Ambient Occlusion"].outputs['Color'], MIXRGB.inputs['Color1'])
	tree.links.new(tree.nodes["Ambient Occlusion"].outputs['AO'], tree.nodes['Gamma'].inputs['Color'])
	tree.links.new(tree.nodes["Gamma"].outputs['Color'], MIXRGB.inputs['Color2'])
	tree.links.new(MIXRGB.outputs['Color'], tree.nodes['Principled BSDF'].inputs['Base Color'])

	# add edge wireframe
	tree.nodes.new(type="ShaderNodeWireframe")
	wire = tree.nodes[-1]
	wire.inputs[0].default_value = edgeThickness
	wire.location.x -= 200
	wire.location.y -= 200
	tree.nodes.new(type="ShaderNodeBsdfDiffuse")
	mat_wire = tree.nodes[-1]
	HSVNode = tree.nodes.new('ShaderNodeHueSaturation')
	HSVNode.inputs['Color'].default_value = edgeColor.RGBA
	HSVNode.inputs['Saturation'].default_value = edgeColor.S
	HSVNode.inputs['Value'].default_value = edgeColor.V
	HSVNode.inputs['Hue'].default_value = edgeColor.H
	HSVNode.location.x -= 200
	# set color brightness/contrast
	BCNode = tree.nodes.new('ShaderNodeBrightContrast')
	BCNode.inputs['Bright'].default_value = edgeColor.B
	BCNode.inputs['Contrast'].default_value = edgeColor.C
	BCNode.location.x -= 400

	tree.links.new(HSVNode.outputs['Color'],BCNode.inputs['Color'])
	tree.links.new(BCNode.outputs['Color'],mat_wire.inputs['Color'])

	tree.nodes.new('ShaderNodeMixShader')
	tree.links.new(wire.outputs[0], tree.nodes['Mix Shader'].inputs[0])
	tree.links.new(mat_wire.outputs['BSDF'], tree.nodes['Mix Shader'].inputs[2])
	tree.links.new(tree.nodes["Principled BSDF"].outputs['BSDF'], tree.nodes['Mix Shader'].inputs[1])
	tree.links.new(tree.nodes["Mix Shader"].outputs['Shader'], tree.nodes['Material Output'].inputs['Surface'])

def render_hand(modelfile, logenfile, ugkfile, ndnfile, outpath):
    res = [720, 720] if quick else [1080, 1080] # recommend >1080 for paper figures
    num_samples = 200 # recommend >200 for paper figures
    meshpos = (0.74, 0.07, 1.03) # UI: click mesh > Transform > Location
    meshrot = (-54, -179, -118) # UI: click mesh > Transform > Rotation
    meshscale = (1.75, 1.75, 1.75) # UI: click mesh > Transform > Scale
    lightangle = (6, -30, -155) # UI: click Sun > Transform > Rotation

    bt.blenderInit(res[0], res[1], num_samples, 1.5)

    bt.invisibleGround(shadowBrightness=0.9)
    
    V, F = gp.read_mesh(modelfile)
    R = Rotation.from_rotvec([-45, 0, 0], degrees=True).as_matrix()
    V = (R @ V.T).T
    mesh = bt.readNumpyMesh(V, F, meshpos, meshrot, meshscale)
    
    vertex_scalars = np.load(logenfile)
    print("max and min of the current log_energy_density file")
    print(np.amin(vertex_scalars))
    print(np.amax(vertex_scalars))
    vertex_scalars = (vertex_scalars - (-6))/(12)
    cmap = colormaps["YlGnBu_r"]
    fcolors = cmap(vertex_scalars)
    
    color_type = 'face'
    mesh = bt.setMeshColors(mesh, fcolors, color_type)
    meshVColor = bt.colorObj([], 0.5, 1.0, 1.0, 0.0, 0.0)
    setMat_VColor(mesh, meshVColor)

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
    
    # now render Gaussian curvature
    
    bt.blenderInit(res[0], res[1], num_samples, 1.5)

    bt.invisibleGround(shadowBrightness=0.9)
    
    V, F = gp.read_mesh(modelfile)
    R = Rotation.from_rotvec([-45, 0, 0], degrees=True).as_matrix()
    V = (R @ V.T).T
    mesh = bt.readNumpyMesh(V, F, meshpos, meshrot, meshscale)
    
    vertex_scalars = np.load(ugkfile)
    print("max, mean, min of the current ugk file")
    print(np.amin(vertex_scalars))
    print(np.mean(vertex_scalars))
    print(np.amax(vertex_scalars))
    vertex_scalars = (vertex_scalars + 100)/(200)
    cmap = colormaps["RdBu_r"]
    fcolors = cmap(vertex_scalars)
    
    color_type = 'vertex'
    mesh = bt.setMeshColors(mesh, fcolors, color_type)
    meshVColor = bt.colorObj([], 0.5, 1.0, 1.0, 0.0, 0.0)
    setMat_VColor(mesh, meshVColor)
    
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
    
    bt.renderImage(outpath[:-4] + "_ugk.png", cam)
    
    
    # now render normal difference to neighbors
    
    bt.blenderInit(res[0], res[1], num_samples, 1.5)

    bt.invisibleGround(shadowBrightness=0.9)
    
    V, F = gp.read_mesh(modelfile)
    R = Rotation.from_rotvec([-45, 0, 0], degrees=True).as_matrix()
    V = (R @ V.T).T
    mesh = bt.readNumpyMesh(V, F, meshpos, meshrot, meshscale)
    
    vertex_scalars = np.load(ndnfile)
    print("max, mean, min of the current ndn file")
    print(np.amin(vertex_scalars))
    print(np.mean(vertex_scalars))
    print(np.amax(vertex_scalars))
    vertex_scalars = 0.3*np.ones_like(vertex_scalars)
    cmap = colormaps["Purples_r"]
    fcolors = cmap(vertex_scalars)
    
    color_type = 'face'
    mesh = bt.setMeshColors(mesh, fcolors, color_type)
    meshVColor = bt.colorObj([], 0.5, 1.0, 1.0, 0.0, 0.0)
    setMat_VColor(mesh, meshVColor)
    
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
    
    bt.renderImage(outpath[:-4] + "_ndn.png", cam)

def unintegrated_gaussian_curvature(V, F):
    return gaussian_curvature(V, F)*(1.0/(gp.massmatrix(V, F).diagonal()))

def normal_difference_to_neighbors(V, F):
    tt, tti = gp.triangle_triangle_adjacency(F)
    normals = gp.per_face_normals(V, F)
    
    diffs = np.zeros(F.shape[0])
    for f in range(F.shape[0]):
        normalcount = 0
        normaldiffsum = 0
        for i in range(3):
            if tt[f, i] == -1:
                continue
            normalcount += 1
            normaldiffsum += np.linalg.norm(normals[tt[f, i], :] - normals[f, :])
        diffs[f] = normaldiffsum / float(normalcount)
    
    return diffs
        
def setMat_VColor(mesh, meshVColor):
    # modded to add roughness
	mat = bpy.data.materials.new('MeshMaterial')
	mesh.data.materials.append(mat)
	mesh.active_material = mat
	mat.use_nodes = True
	tree = mat.node_tree

	# read vertex attribute
	tree.nodes.new('ShaderNodeAttribute')
	tree.nodes[-1].attribute_name = "Col"
	HSVNode = tree.nodes.new('ShaderNodeHueSaturation')
	tree.links.new(tree.nodes["Attribute"].outputs['Color'], HSVNode.inputs['Color'])
	HSVNode.inputs['Saturation'].default_value = meshVColor.S
	HSVNode.inputs['Value'].default_value = meshVColor.V
	HSVNode.inputs['Hue'].default_value = meshVColor.H
	HSVNode.location.x -= 200

	# set color brightness/contrast
	BCNode = tree.nodes.new('ShaderNodeBrightContrast')
	BCNode.inputs['Bright'].default_value = meshVColor.B
	BCNode.inputs['Contrast'].default_value = meshVColor.C
	BCNode.location.x -= 400
 
    # add Ambient Occlusion
	tree.nodes.new('ShaderNodeAmbientOcclusion')
	tree.nodes.new('ShaderNodeGamma')
	MIXRGB = tree.nodes.new('ShaderNodeMixRGB')
	MIXRGB.blend_type = 'MULTIPLY'
	tree.nodes["Gamma"].inputs["Gamma"].default_value = 0.5
	tree.nodes["Ambient Occlusion"].inputs["Distance"].default_value = 10.0
	tree.nodes["Gamma"].location.x -= 600

	# set principled BSDF
	tree.nodes["Principled BSDF"].inputs['Roughness'].default_value = 0.5
	tree.nodes["Principled BSDF"].inputs['Sheen Tint'].default_value = [0, 0, 0, 1]
	tree.links.new(HSVNode.outputs['Color'], BCNode.inputs['Color'])
	tree.links.new(BCNode.outputs['Color'], tree.nodes['Principled BSDF'].inputs['Base Color'])


# mesh choices

experiments = [("cubichand", "../models/cubic_hand.obj", 1, 1)]

hesstype = "vfef"

for e in experiments:
    
    modelname, modelfilename, fidelity_weight, iterations = e
    
    # first, generate the original mesh files
    
    logendensfilename = "../results/npys/cubiccomparison_" + modelname + ".npy"
    ugkfilename = "../results/npys/cubiccomparison_ugk_" + modelname + ".npy"
    ndnfilename = "../results/npys/cubiccomparison_ndn_" + modelname + ".npy"
    
    V, F = gp.read_mesh(modelfilename)
    
    meshscale = np.amax(np.linalg.norm(V, axis=1))
    
    hel = gp.halfedge_lengths(V, F)
    hel = hel/np.amax(hel)
    
    if regen: 
        u0x, u0y, u0z = V[:, 0], V[:, 1], V[:, 2]
        H = vfef(hel, F)
        Mfinv = 2.0/gp.doublearea_intrinsic(hel**2, F)[:, None] # inverse mass matrix for visualisation, assumes M diagonal 
        energy_density = np.linalg.norm(np.reshape(H @ u0x, (-1, 4)), axis=1, keepdims=True) + \
            np.linalg.norm(np.reshape(H @ u0y, (-1, 4)), axis=1, keepdims=True) + \
            np.linalg.norm(np.reshape(H @ u0z, (-1, 4)), axis=1, keepdims=True)
        energy_density = (Mfinv * energy_density)[:, 0]
        log_energy_density = np.log(energy_density) 
        np.save(logendensfilename, log_energy_density)
        
        ugk = unintegrated_gaussian_curvature(V, F)
        np.save(ugkfilename, ugk)
        
        ndn = normal_difference_to_neighbors(V, F)
        np.save(ndnfilename, ndn)
    
    print("here")
    
if modelname in {"cubichand", "hand"}:
    render_hand(modelfilename, logendensfilename, ugkfilename, ndnfilename, "../results/pngs/cubiccomparison_" + modelname + ".png")