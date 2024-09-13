import sys
sys.path.insert(0, '../src/')

from l1hessian import hessian_l1_solve, vfef
import gpytoolbox as gp
import numpy as np
import os
import blendertoolbox as bt
import bpy
from matplotlib import colormaps
from scipy.spatial.transform import Rotation
from igl import gaussian_curvature
import mosek

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

def render_sphere(modelfile, logenfile, ugkfile, ndnfile, outpath):
    res = [720, 720] if quick else [1080, 1080] # recommend >1080 for paper figures
    num_samples = 200 # recommend >200 for paper figures
    meshpos = (0.3, 0, 0.75) # UI: click mesh > Transform > Location
    meshrot = (0, -25, 0) # UI: click mesh > Transform > Rotation
    meshscale = (0.7,0.7,0.7) # UI: click mesh > Transform > Scale
    meshc = [0.25, 0.25, 0.25]
    lightangle = (6, -30, -155) # UI: click Sun > Transform > Rotation

    bt.blenderInit(res[0], res[1], num_samples, 1.5)

    bt.invisibleGround(shadowBrightness=0.9)

    mesh = bt.readMesh(modelfile, meshpos, meshrot, meshscale)
    
    edgeThickness = 0.001
    edgeColor = bt.colorObj((0,0,0,0),0.5, 1.0, 1.0, 0.0, 0.0)
    meshRGBA = tuple(meshc + [1.0])
    AOStrength = 1.0
    setMat_edge(mesh, edgeThickness, edgeColor, meshRGBA, AOStrength)
    
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

def render_cube(modelfile, logenfile, ugkfile, ndnfile, outpath):
    res = [720, 720] if quick else [1080, 1080] # recommend >1080 for paper figures
    num_samples = 200 # recommend >200 for paper figures
    meshpos = (0.6, 0.11, 0.87) # UI: click mesh > Transform > Location
    meshrot = (-54, 13, -109) # UI: click mesh > Transform > Rotation
    meshscale = (0.5, 0.5, 0.5) # UI: click mesh > Transform > Scale
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
    vertex_scalars = (vertex_scalars - (-6))/(6)
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

    # mesh = bt.readMesh(modelfile, meshpos, meshrot, meshscale)
    
    V, F = gp.read_mesh(modelfile)
    R = Rotation.from_rotvec([-45, 0, 0], degrees=True).as_matrix()
    V = (R @ V.T).T
    mesh = bt.readNumpyMesh(V, F, meshpos, meshrot, meshscale)
    
    vertex_scalars = np.load(ndnfile)
    print("max, mean, min of the current ndn file")
    print(np.amin(vertex_scalars))
    print(np.mean(vertex_scalars))
    print(np.amax(vertex_scalars))
    vertex_scalars = (vertex_scalars - vertex_scalars.min()) / ((vertex_scalars.max() +np.mean(vertex_scalars))/2.0 - vertex_scalars.min())
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

def render_hand(modelfile, logenfile, ugkfile, ndnfile, outpath):
    res = [720, 720] if quick else [1080, 1080] # recommend >1080 for paper figures
    num_samples = 200 # recommend >200 for paper figures
    meshpos = (0.86, 0.07, 0.08) # UI: click mesh > Transform > Location
    meshrot = (-54, -179, -109) # UI: click mesh > Transform > Rotation
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
    vertex_scalars = 0.3*np.ones_like(vertex_scalars) # to make it constant
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

    # mesh = bt.readMesh(modelfile, meshpos, meshrot, meshscale)
    
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
    vertex_scalars = (vertex_scalars - vertex_scalars.min()) / ((vertex_scalars.max() +np.mean(vertex_scalars))/2.0 - vertex_scalars.min())
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

def render_koala(modelfile, logenfile, ugkfile, ndnfile, outpath):
    res = [720, 720] if quick else [1080, 1080] # recommend >1080 for paper figures
    num_samples = 200 # recommend >200 for paper figures
    meshpos = (-0.16, 0.23, 0.39) # UI: click mesh > Transform > Location
    meshrot = (135, 0, 49) # UI: click mesh > Transform > Rotation
    meshscale = (0.25, 0.25, 0.25) # UI: click mesh > Transform > Scale
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
    vertex_scalars = (vertex_scalars - (-4))/(6)
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

    # mesh = bt.readMesh(modelfile, meshpos, meshrot, meshscale)
    
    V, F = gp.read_mesh(modelfile)
    R = Rotation.from_rotvec([-45, 0, 0], degrees=True).as_matrix()
    V = (R @ V.T).T
    mesh = bt.readNumpyMesh(V, F, meshpos, meshrot, meshscale)
    
    vertex_scalars = np.load(ugkfile)
    print("max, mean, min of the current ugk file")
    print(np.amin(vertex_scalars))
    print(np.mean(vertex_scalars))
    print(np.amax(vertex_scalars))
    vertex_scalars = (vertex_scalars + 10)/(20)
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
    vertex_scalars = (vertex_scalars - vertex_scalars.min()) / ((vertex_scalars.max() +np.mean(vertex_scalars))/2.0 - vertex_scalars.min())
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

def render_springer(modelfile, logenfile, ugkfile, ndnfile, outpath):
    res = [720, 720] if quick else [1080, 1080] # recommend >1080 for paper figures
    num_samples = 200 # recommend >200 for paper figures
    meshpos = (0.37, 0, 1.0) # UI: click mesh > Transform > Location
    meshrot = (-225, 0, -136) # UI: click mesh > Transform > Rotation
    meshscale = (1.0, 1.0, 1.0) # UI: click mesh > Transform > Scale
    lightangle = (6, -30, -155) # UI: click Sun > Transform > Rotation

    bt.blenderInit(res[0], res[1], num_samples, 1.5)

    bt.invisibleGround(shadowBrightness=0.9)
    
    V, F = gp.read_mesh(modelfile)
    R = Rotation.from_rotvec([-45, 0, 0], degrees=True).as_matrix()
    V = (R @ V.T).T
    mesh = bt.readNumpyMesh(V, F, meshpos, meshrot, meshscale)
    
    vertex_scalars = np.load(logenfile)
    vertex_scalars = np.nan_to_num(vertex_scalars)
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
    vertex_scalars = (vertex_scalars - vertex_scalars.min()) / ((vertex_scalars.max() +np.mean(vertex_scalars))/2.0 - vertex_scalars.min())
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

def render_nefertiti(modelfile, logenfile, ugkfile, ndnfile, outpath):
    res = [720, 720] if quick else [1080, 1080] # recommend >1080 for paper figures
    num_samples = 200 # recommend >200 for paper figures
    meshpos = (0.63, 0.07, 0.87) # UI: click mesh > Transform > Location
    meshrot = (135, 0, 71) # change for revision
    meshscale = (0.0035, 0.0035, 0.0035) # UI: click mesh > Transform > Scale
    lightangle = (6, -30, -155) # UI: click Sun > Transform > Rotation

    bt.blenderInit(res[0], res[1], num_samples, 1.5)

    bt.invisibleGround(shadowBrightness=0.9)
    
    V, F = gp.read_mesh(modelfile)
    R = Rotation.from_rotvec([-45, 0, 0], degrees=True).as_matrix()
    V = (R @ V.T).T
    mesh = bt.readNumpyMesh(V, F, meshpos, meshrot, meshscale)
    
    vertex_scalars = np.load(logenfile)
    vertex_scalars = np.nan_to_num(vertex_scalars)
    print("max and min of the current log_energy_density file")
    print(np.amin(vertex_scalars))
    print(np.mean(vertex_scalars))
    print(np.amax(vertex_scalars))
    ub, lb = 6., -6.
    vertex_scalars = np.clip((vertex_scalars - (lb))/np.abs(ub-lb), 0, 1)
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
    vertex_scalars = (vertex_scalars - vertex_scalars.min()) / ((vertex_scalars.max() +np.mean(vertex_scalars))/2.0 - vertex_scalars.min())
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

experiments = []

# step size experiments
jump = 0.01 # take timesteps at some fidelity weight to get to "jump" amount of time
experiments += [("nefertiti" + str(a), "../models/nefertiti-lowres.obj", a, int(2*a*jump)) for a in (1250, 1000, 500, 50)]

# other flow experiments
experiments += [("cube", "../models/cube_remesh.obj", 100, 50)]
experiments += [("hand", "../models/hand.obj", 500, 43)]
experiments += [("koala", "../models/koala.obj", 500, 40)]
experiments += [("springer1", "../models/springer1.obj", 500, 20)]

# discretization dependence
experiments += [("icosphere", "../models/ico_highres.obj", 500, 250),
               ("uvsphere", "../models/uvsphere_highres.obj", 500, 250),
               ("cubesphere", "../models/cubesphere_highres.obj", 500, 250)]

hesstype = "vfef"

for e in experiments:
    
    modelname, modelfilename, fidelity_weight, iterations = e
    
    # first, generate the original mesh files
    
    logendensfilename = "../results/npys/initflow_" + modelname + ".npy"
    ugkfilename = "../results/npys/initflow_ugk_" + modelname + ".npy"
    ndnfilename = "../results/npys/initflow_ndn_" + modelname + ".npy"
    
    V, F = gp.read_mesh(modelfilename)
    
    meshscale = np.amax(np.linalg.norm(V, axis=1))
    
    # scale some models
    if modelname[:9] == "nefertiti":
        V /= meshscale
    
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
        
    
    if modelname in {"cube"}:
        render_cube(modelfilename, logendensfilename, ugkfilename, ndnfilename, "../results/pngs/initflow_" + modelname + ".png")
    elif modelname in {"icosphere", "uvsphere", "cubesphere"}:
        render_sphere(modelfilename, logendensfilename, ugkfilename, ndnfilename, "../results/pngs/initflow_" + modelname + ".png")
    elif modelname in {"hand"}:
        render_hand(modelfilename, logendensfilename, ugkfilename, ndnfilename, "../results/pngs/initflow_" + modelname + ".png")
    elif modelname in {"koala"}:
        render_koala(modelfilename, logendensfilename, ugkfilename, ndnfilename, "../results/pngs/initflow_" + modelname + ".png")
    elif modelname in {"springer1"}:
        render_springer(modelfilename, logendensfilename, ugkfilename, ndnfilename, "../results/pngs/initflow_" + modelname + ".png")
    elif modelname[:9] in {"nefertiti"}:
        render_nefertiti(modelfilename, logendensfilename, ugkfilename, ndnfilename, "../results/pngs/initflow_" + modelname + ".png")
    
    if regen:
        for i in range(iterations):
            print(modelname + " iteration " + str(i))
            
            # get hel of current iteration; # rescale to be within (0,1] range
            hel = gp.halfedge_lengths(V, F)
            hel = hel/np.amax(hel)
            
            # get the coordinates as vertex functions
            u0x, u0y, u0z = V[:, 0], V[:, 1], V[:, 2]
            
            H = vfef(hel, F)
            Mfinv = 2.0/gp.doublearea_intrinsic(hel**2, F)[:, None] # inverse mass matrix for visualisation, assumes M diagonal 
            energy_density = np.linalg.norm(np.reshape(H @ u0x, (-1, 4)), axis=1, keepdims=True) + \
                np.linalg.norm(np.reshape(H @ u0y, (-1, 4)), axis=1, keepdims=True) + \
                np.linalg.norm(np.reshape(H @ u0z, (-1, 4)), axis=1, keepdims=True)
            energy_density = (Mfinv * energy_density)[:, 0]
            log_energy_density = np.log(energy_density)
            
            ugk = unintegrated_gaussian_curvature(V, F)
            
            ndn = normal_difference_to_neighbors(V, F)
            
            # solve for the new coordinates as vertex functions
            try:
                u1x = hessian_l1_solve(hel, F, u0=u0x, mode=hesstype, alpha=fidelity_weight)
                u1y = hessian_l1_solve(hel, F, u0=u0y, mode=hesstype, alpha=fidelity_weight)
                u1z = hessian_l1_solve(hel, F, u0=u0z, mode=hesstype, alpha=fidelity_weight)
            except ValueError:
                print("breaking at iteration " + str(i) + "due to ValueError.")
                u1x = u0x
                u1y = u0y
                u1z = u0z
                V = np.hstack([u1x[:, None], u1y[:, None], u1z[:, None]])
                break
            except mosek.Error:
                print("breaking at iteration " + str(i) + "due to mosek solver error.")
                u1x = u0x
                u1y = u0y
                u1z = u0z
                V = np.hstack([u1x[:, None], u1y[:, None], u1z[:, None]])
                break
            
            # stack them into new vertex positions
            V = np.hstack([u1x[:, None], u1y[:, None], u1z[:, None]])
        
        # evaluate the energy density __AFTER__ the loop is over
        hel = gp.halfedge_lengths(V, F)
        hel = hel/np.amax(hel)
        # get the coordinates as vertex functions
        u0x, u0y, u0z = V[:, 0], V[:, 1], V[:, 2]
        H = vfef(hel, F)
        Mfinv = 2.0/gp.doublearea_intrinsic(hel**2, F)[:, None] # inverse mass matrix for visualisation, assumes M diagonal 
        energy_density = np.linalg.norm(np.reshape(H @ u0x, (-1, 4)), axis=1, keepdims=True) + \
            np.linalg.norm(np.reshape(H @ u0y, (-1, 4)), axis=1, keepdims=True) + \
            np.linalg.norm(np.reshape(H @ u0z, (-1, 4)), axis=1, keepdims=True)
        energy_density = (Mfinv * energy_density)[:, 0]
        log_energy_density = np.log(energy_density)
        ugk = unintegrated_gaussian_curvature(V, F)
        ndn = normal_difference_to_neighbors(V, F)
        
        if modelname[:9] == "nefertiti":
            V *= meshscale
        
    # after those iterations, save as an obj and rerender
    new_modelfilename = "../results/objs/flow_" + modelname + "_" + str(fidelity_weight) + ".obj"
    if regen: gp.write_mesh(new_modelfilename, V, F)
    new_logendensfilename = "../results/npys/flow_" + modelname + "_" + str(fidelity_weight) + ".npy"
    if regen: np.save(new_logendensfilename, log_energy_density)
    new_ugkfilename = "../results/npys/flow_ugk_" + modelname + ".npy"
    if regen: np.save(new_ugkfilename, ugk)
    new_ndnfilename = "../results/npys/flow_ndn_" + modelname + ".npy"
    if regen: np.save(new_ndnfilename, ndn)
    
    if modelname in {"cube"}:
        render_cube(new_modelfilename, new_logendensfilename, new_ugkfilename, new_ndnfilename, "../results/pngs/endflow_" + modelname + ".png")
    elif modelname in {"icosphere", "uvsphere", "cubesphere"}:
        render_sphere(new_modelfilename, new_logendensfilename, new_ugkfilename, new_ndnfilename, "../results/pngs/endflow_" + modelname + ".png")
    elif modelname in {"hand"}:
        render_hand(new_modelfilename, new_logendensfilename, new_ugkfilename, new_ndnfilename, "../results/pngs/endflow_" + modelname + ".png")
    elif modelname in {"koala"}:
        render_koala(new_modelfilename, new_logendensfilename, new_ugkfilename, new_ndnfilename, "../results/pngs/endflow_" + modelname + ".png")
    elif modelname in {"springer1"}:
        render_springer(new_modelfilename, new_logendensfilename, new_ugkfilename, new_ndnfilename, "../results/pngs/endflow_" + modelname + ".png")
    elif modelname[:9] in {"nefertiti"}:
        render_nefertiti(new_modelfilename, new_logendensfilename, new_ugkfilename, new_ndnfilename, "../results/pngs/endflow_" + modelname + ".png")