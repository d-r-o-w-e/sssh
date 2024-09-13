import sys
sys.path.insert(0, '../src/')

from l1hessian import at_hess1_segment, thresh_segmentation_from_v
import gpytoolbox as gp
import numpy as np
import os
import blendertoolbox as bt
import bpy
from matplotlib import colormaps

regen = True if int(sys.argv[1]) == 1 else False
quick = True if int(sys.argv[2]) == 1 else False

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

def render_seg(Vis, Fis, adjs, outpath):
    res = [720, 720] if quick else [2160, 2160] # recommend >1080 for paper figures
    num_samples = 200 # recommend >200 for paper figures
    
    meshpos = (-0.61, 0.16, 0.73) # cube
    meshrot = (-270, 0, -20) # cube
    meshscale = (0.03, 0.03, 0.03) # cube
    
    lightangle = (6, -30, -155) # UI: click Sun > Transform > Rotation

    bt.blenderInit(res[0], res[1], num_samples, 1.5)

    bt.invisibleGround(shadowBrightness=0.9)

    rng = np.random.default_rng(3)

    classes = 10
    cmapname = "tab10"

    colorlist = [-1]*len(Vis)
    for i in range(len(Vis)):
        cidx = i % 10
        cidx_orig = i % 10
        while cidx in [colorlist[a] for a in adjs[i]]:
            cidx = (cidx + 1) % 10
            if cidx_orig == cidx:
                1
                raise ValueError("found unresolvable adjacencies; consider changing the colormap, or commenting this line out")
        colorlist[i] = cidx

    for i in range(len(Vis)):
        Vi, Fi = Vis[i], Fis[i]
        
        mesh = bt.readNumpyMesh(Vi, Fi, meshpos, meshrot, meshscale)
        
        for f in mesh.data.polygons:
            f.use_smooth=True
        
        cmap = colormaps[cmapname] 
        color = cmap(colorlist[i]*(1.0/classes) + 1/(2*classes))
        
        meshColor = bt.colorObj(color, 0.5, 1.0, 1.0, 0.0, 2.0)
        bt.setMat_singleColor(mesh, meshColor, 0)
    
    camLocation = (3, 0, 2)
    lookAtLocation = (0,0,0.5)
    focalLength = 45 # (UI: click camera > Object Data > Focal Length)
    cam = bt.setCamera(camLocation, lookAtLocation, focalLength)
    
    strength = 2
    shadowSoftness = 0.3
    sun = bt.setLight_sun(lightangle, strength, shadowSoftness)
    
    bt.setLight_ambient(color=(0.3, 0.3, 0.3, 1)) 
    
    bpy.ops.wm.save_mainfile(filepath=os.getcwd() + '/test.blend')
    
    bt.renderImage(outpath, cam)

def render_seg_geom(modelfile, v, outpath):
    res = [720, 720] if quick else [2160, 2160] # recommend >1080 for paper figures
    num_samples = 200 # recommend >200 for paper figures
    
    meshpos = (-0.61, 0.16, 0.73) # cube
    meshrot = (-270, 0, -20) # cube
    meshscale = (0.03, 0.03, 0.03) # cube
    
    lightangle = (6, -30, -155) # UI: click Sun > Transform > Rotation

    bt.blenderInit(res[0], res[1], num_samples, 1.5)

    bt.invisibleGround(shadowBrightness=0.9)

    mesh = bt.readMesh(modelfile, meshpos, meshrot, meshscale)
    
    cmapname = "BuGn_r"
    cmap = colormaps[cmapname]
    vertex_colors = cmap(v)
    color_type = 'face'
    mesh = bt.setMeshColors(mesh, vertex_colors, color_type)
    meshVColor = bt.colorObj([], 0.5, 1.0, 1.0, 0.0, 0.0)
    bt.setMat_VColor(mesh, meshVColor)
    
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

def render_building_itself(modelfile, tx, outpath):
    res = [720, 720] if quick else [2160, 2160] # recommend >1080 for paper figures
    num_samples = 200 # recommend >200 for paper figures
    
    meshpos = (-0.61, 0.16, 0.73) # cube
    meshrot = (-270, 0, -20) # cube
    meshscale = (0.03, 0.03, 0.03) # cube
    
    lightangle = (6, -30, -155) # UI: click Sun > Transform > Rotation

    bt.blenderInit(res[0], res[1], num_samples, 1.5)

    bt.invisibleGround(shadowBrightness=0.9)

    mesh = bt.readMesh(modelfile, meshpos, meshrot, meshscale)
    
    meshColor = bt.colorObj((0,0,0,1), 0.5, 1.0, 1.0, 0.0, 0.0)
    bt.setMat_texture(mesh, tx, meshColor)
    
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

# mesh choices
meshname = "building_CUT"

filename_npy = "../results/npys/segmentation_cut_" + meshname + ".npy"
filename_png = "../results/pngs/segmentation_cut_" + meshname + ".png"
filename_geom_obj = "../results/objs/seg_geom_" + meshname + ".obj"
filename_geom_png = "../results/pngs/seg_geom_" + meshname + ".png"
filename_original_tx = "../results/pngs/original_" + meshname + ".png"

modelfilename = "../models/building_manageable_CUT.obj"

V, F = gp.read_mesh(modelfilename)

# set offset to make it better
xrange = np.amax(V[:, 0]) - np.amin(V[:, 0])

# building CUT
hesstype = "vfef"
lam = 0.03
alpha = 0.5
eps1 = 0.1 # starting epsilon in A-T formula
eps2 = 0.001
eps3 = 1e-5 # inner loop stopping criterion epsilon; multiplied by mesh surface area in the function
n = 30
cut_thresh = 0.925
merge_limit = 20

print("lam:", lam)
print("alpha:", alpha)
print("eps1:", eps1)
print("eps2:", eps2)
print("using eps3 = " + str(eps3))
print("n:", n)
print("cut_thresh:", cut_thresh)
print("merge_limit:", merge_limit)

if regen:
    U_mesh, v = at_hess1_segment(V, F, lam=lam, alpha=alpha, eps1=eps1, eps2=eps2, eps3=eps3, n=n, verbose=True)
    print("CUTTING MESH")
    Vis, Fis, cf = thresh_segmentation_from_v(V, F, v, cut_thresh=cut_thresh, merge_limit=merge_limit)
    
    gp.write_mesh(filename_geom_obj, U_mesh, F)
    np.save(filename_npy, np.array([Vis, Fis, cf, v], dtype=object), allow_pickle=True)

U_mesh = gp.read_mesh(filename_geom_obj)[0]
VisFis = np.load(filename_npy, allow_pickle=True)
Vis, Fis, cf, v = VisFis[0], VisFis[1], VisFis[2], VisFis[3]

# get adjs from cf
tt, tti = gp.triangle_triangle_adjacency(F)
adjs = []
for i in range(np.amax(cf)+1):
    adjs += [list(set(cf[tt[cf==i, :].flatten()]) - {i})]

render_seg(Vis, Fis, adjs, filename_png)
render_seg_geom(filename_geom_obj, v[:, 0], filename_geom_png)
render_building_itself(modelfilename, "../models/sham_test_building_texture.jpeg", filename_original_tx)