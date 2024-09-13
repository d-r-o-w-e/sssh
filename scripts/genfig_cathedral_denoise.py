import sys
sys.path.insert(0, '../src/')

from l1hessian import hessian_l1_solve
import gpytoolbox as gp
import numpy as np
import scipy.sparse as sp
import blendertoolbox as bt
import bpy
from matplotlib import colormaps
import os
import time

regen_np = True if int(sys.argv[1]) == 1 else False
quick = True if int(sys.argv[2]) == 1 else False

def setMat_VColor(mesh, meshVColor):
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

	# set principled BSDF
	tree.nodes["Principled BSDF"].inputs['Roughness'].default_value = 0.5
	tree.nodes["Principled BSDF"].inputs['Sheen Tint'].default_value = [0, 0, 0, 1]
	tree.nodes["Principled BSDF"].inputs['Specular IOR Level'].default_value = 0.5
	tree.nodes["Principled BSDF"].inputs['IOR'].default_value = 1.45
	tree.nodes["Principled BSDF"].inputs['Transmission Weight'].default_value = 0
	tree.nodes["Principled BSDF"].inputs['Coat Roughness'].default_value = 0

	# add Ambient Occlusion
	tree.nodes.new('ShaderNodeAmbientOcclusion')
	tree.nodes.new('ShaderNodeGamma')
	MIXRGB = tree.nodes.new('ShaderNodeMixRGB')
	MIXRGB.blend_type = 'MULTIPLY'
	tree.nodes["Gamma"].inputs["Gamma"].default_value = 0
	tree.nodes["Ambient Occlusion"].inputs["Distance"].default_value = 10.0
	tree.nodes["Gamma"].location.x -= 600

	# link all the nodes
	tree.links.new(HSVNode.outputs['Color'], BCNode.inputs['Color'])
	tree.links.new(BCNode.outputs['Color'], tree.nodes['Ambient Occlusion'].inputs['Color'])
	tree.links.new(tree.nodes["Ambient Occlusion"].outputs['Color'], MIXRGB.inputs['Color1'])
	tree.links.new(tree.nodes["Ambient Occlusion"].outputs['AO'], tree.nodes['Gamma'].inputs['Color'])
	tree.links.new(tree.nodes["Gamma"].outputs['Color'], MIXRGB.inputs['Color2'])
	tree.links.new(MIXRGB.outputs['Color'], tree.nodes['Principled BSDF'].inputs['Base Color'])

def binsearch(gt, solve_alpha, minbound, maxbound, ary=5, tol=1e-3):
    
    currmin, currmax = minbound, maxbound
    
    while (currmax-currmin) > tol:
        print(currmin, currmax)
        guesses = np.linspace(currmin,currmax, ary, endpoint=True)
        losses = np.array([np.sum((gt-solve_alpha(guesses[i]))**2, axis=print(i)) for i in range(ary)])
        i_best = np.argmin(losses)
        currmin, currmax = np.clip(guesses[i_best-1], minbound, maxbound), np.clip(guesses[i_best+1], minbound, maxbound)
    
    print("final loss:", str(np.sum((gt-solve_alpha((currmax+currmin)/2.0))**2)))
    
    return (currmax+currmin)/2.0

def render_cathedral(V, F, cmapname, outpath):
    
    res = [720, 720] if quick else [1080, 1080] # recommend >1080 for paper figures
    num_samples = 200 # recommend >200 for paper figures
    meshpos = (-0.45, 0.25, 0) # UI: click mesh > Transform > Location
    meshrot = (0, 0, 45) # UI: click mesh > Transform > Rotation
    meshscale = (3.0, 3.0, 3.0) # UI: click mesh > Transform > Scale
    lightangle = (6, -30, -155) # UI: click Sun > Transform > Rotation

    bt.blenderInit(res[0], res[1], num_samples, 1.5)

    bt.invisibleGround(shadowBrightness=0.9)
    
    mesh = bt.readNumpyMesh(V, F, meshpos, meshrot, meshscale)
    
    cmap = colormaps[cmapname]
    vscalars = (V[:, 2]-V[:, 2].min())/(V[:, 2].max()-V[:, 2].min())
    
    vertex_colors = cmap(vscalars)
    color_type = 'vertex'
    mesh = bt.setMeshColors(mesh, vertex_colors, color_type)
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

def add_walls(V, F):
    
    bE = gp.boundary_loops(F)[0]
    bE = np.reshape(np.concatenate([bE, bE[1:], bE[0:1]]), (-1, 2), order='F')
    orig_vshape = V.shape[0]
    newV = np.copy(V[bE[:, 0], :])
    newV[:, 2] = 0
    newF = np.zeros((bE.shape[0]*2, 3), dtype=int)
    for i in range(bE.shape[0]):
        newF[2*i:2*i+2, :] = np.array([[bE[i, 0], orig_vshape+i, bE[i, 1]],
                                       [orig_vshape+i, orig_vshape+((i+1) % bE.shape[0]), bE[i, 1]]])
    return np.vstack([V, newV]), np.vstack([F, newF])


cathedral = lambda x: (x[:,0]<0.4)*(x[:,0]>-0.4)*(x[:,1]<-0.2)*(x[:,1]>-0.6)*(0.2-np.abs(x[:,1]+0.4)) \
                + np.maximum(2.*(0.15-np.linalg.norm(x-np.array([[0.2,0.4]]),ord=np.inf,axis=-1))*((0.15-np.linalg.norm(x-np.array([[0.2,0.4]]),ord=np.inf,axis=-1))>0.) \
                + 2.*(0.15-np.linalg.norm(x-np.array([[-0.2,0.4]]),ord=np.inf,axis=-1))*((0.15-np.linalg.norm(x-np.array([[-0.2,0.4]]),ord=np.inf,axis=-1))>0.),
                (x[:,0]>-0.2)*(x[:,0]<0.2)*(x[:,1]<0.65)*(x[:,1]>-0.8)*(0.2-np.abs(x[:,0]))) \
                + (x[:,0]>-0.2)*(x[:,0]<0.2)*(x[:,1]<-0.8)*(0.2-np.linalg.norm(x-np.array([[0.,-0.8]]),axis=-1))*((0.2-np.linalg.norm(x-np.array([[0.,-0.8]]),axis=-1))>0.) \
                + ((0.2-np.linalg.norm(x-np.array([[0.,-0.4]]),ord=np.inf,axis=-1))>0.)*(0.2-np.linalg.norm(x-np.array([[0.,-0.4]]),ord=np.inf,axis=-1))

# file locations
outgt = "../results/pngs/cathedral_fndenoise_gt.png"
outnoisy = "../results/pngs/cathedral_fndenoise_noisy.png"
outours = "../results/pngs/cathedral_fndenoise_ours.png"
outstein = "../results/pngs/cathedral_fndenoise_stein.png"
outlapl2 = "../results/pngs/cathedral_fndenoise_lapl2.png"
outl1laplacian = "../results/pngs/cathedral_fndenoise_l1_laplacian.png"
outl0laplacian = "../results/pngs/cathedral_fndenoise_l0_laplacian.png" # this is l1 of the _edge_ laplacian
outnpy = "../results/npys/cathedral_fndenoise.npy"

# tunable params
hesstype = "vfef"
mu = 100
eps = 1e-12
levels = 50
re_search = False

modelname = "cathedral"
modelfilename = "../models/cathedral.obj"

V, F = gp.read_mesh(modelfilename)
V, F = gp.subdivide(V, F, method='loop', iters=1)

# get hel
hel = gp.halfedge_lengths(V, F)
hel /= np.amax(hel)

rng = np.random.default_rng(0)
heightfield = cathedral(2*V[:, :2]-0.005*np.array([[1, 0]]))/2.0
if regen_np:
    
    u0 = 0.03*rng.uniform(-1, 1, size=V.shape[0]) + heightfield
    
    alpha = 100 # ours
    alpha1 = 0.05 # lapl2
    alpha2 = 100 # stein
    alpha3 = 75 # formerly l1 laplacian
    alpha4 = 25 # edge laplacian
    
    # binary search to find reasonable initial parameters; fix them later
    if re_search:
        print("alpha")
        alpha = binsearch(heightfield, lambda alpha: hessian_l1_solve(hel, F, alpha=alpha, u0=u0, mode=hesstype), minbound=0, maxbound=1000)
        print("found best alpha: " + str(alpha))
        print("alpha1")
        alpha1 = binsearch(heightfield, lambda alpha1: gp.min_quad_with_fixed(Q=2*(gp.biharmonic_energy_intrinsic(hel**2, F, bc="curved_hessian") + alpha1*gp.massmatrix_intrinsic(hel**2, F)), 
                                                                              c=-2*alpha1*(gp.massmatrix_intrinsic(hel**2, F) @ u0[:, None]))[:, 0], 0, 10)
        print("found best alpha1: " + str(alpha1))
        print("alpha2")
        alpha2 = binsearch(heightfield, lambda alpha2: hessian_l1_solve(hel, F, alpha=alpha2, u0=u0, mode="stein et al 2018", V=V), 0.0001, 1000)
        print("found best alpha2: " + str(alpha2))
        print("alpha3")
        alpha3 = binsearch(heightfield, lambda alpha2: hessian_l1_solve(hel, F, alpha=alpha2, u0=u0, mode="l1_laplacian"), minbound=0.0001, maxbound=1000)
        print("found best alpha3: " + str(alpha3))
        
    
    orig_fn = u0
    print("our solve")
    st = time.time()
    recovered_fn = hessian_l1_solve(hel, F, alpha=alpha, u0=u0, mode=hesstype)
    et = time.time()
    print("our time:" + str(et-st))
    print("lapl2 solve")
    st = time.time()
    lapl2_fn = gp.min_quad_with_fixed(Q=2*(gp.biharmonic_energy_intrinsic(hel**2, F, bc="curved_hessian") + alpha1*gp.massmatrix_intrinsic(hel**2, F)), 
                                      c=-2*alpha1*(gp.massmatrix_intrinsic(hel**2, F) @ u0[:, None]))[:, 0]
    et = time.time()
    print("lapl2 time:" + str(et-st))
    print("stein et al solve")
    st = time.time()
    steinetal2018_fn = hessian_l1_solve(hel, F, alpha=alpha2, u0=u0, mode="stein et al 2018", V=V)
    et = time.time()
    print("stein time:" + str(et-st))
    print("l1 laplacian solve")
    l1_laplacian_fn = hessian_l1_solve(hel, F, alpha=alpha2, u0=u0, mode="l1_laplacian") # not in paper
    print("edge laplacian solve")
    l0_laplacian_fn = hessian_l1_solve(hel, F, V=V, alpha=alpha4, u0=u0, mode="l1_edgelaplacian")
    
    with open(outnpy, 'wb') as f:
        np.save(f, orig_fn)
        np.save(f, recovered_fn)
        np.save(f, lapl2_fn)
        np.save(f, steinetal2018_fn)
        np.save(f, l1_laplacian_fn)
        np.save(f, l0_laplacian_fn)

with open(outnpy, 'rb') as f:
    orig_fn, recovered_fn, lapl2_fn, steinetal2018_fn, l1_laplacian_fn, l0_laplacian_fn = np.load(f), np.load(f), np.load(f), np.load(f), np.load(f), np.load(f)

shiftheight = 0.1
shifter = shiftheight

V_orig, V_ours, V_stein, V_lapl2, V_l1_laplacian, V_l0_laplacian = V.copy(), V.copy(), V.copy(), V.copy(), V.copy(), V.copy()
V_orig[:, 2] = orig_fn + shifter
V_ours[:, 2] = recovered_fn + shifter
V_stein[:, 2] = steinetal2018_fn + shifter
V_lapl2[:, 2] = lapl2_fn + shifter 
V_l1_laplacian[:, 2] = l1_laplacian_fn + shifter
V_l0_laplacian[:, 2] = l0_laplacian_fn + shifter
V[:, 2] = heightfield + shifter

V_orig, F_orig = add_walls(V_orig, F)
V_ours, F_ours = add_walls(V_ours, F)
V_stein, F_stein = add_walls(V_stein, F)
V_lapl2, F_lapl2 = add_walls(V_lapl2, F)
V_l1_laplacian, F_l1_laplacian = add_walls(V_l1_laplacian, F)
V_l0_laplacian, F_l0_laplacian = add_walls(V_l0_laplacian, F)
V, F = add_walls(V, F)

render_cathedral(V_l0_laplacian, F_l0_laplacian, "Purples_r", outl0laplacian)
render_cathedral(V, F, "Greys_r", outgt)
render_cathedral(V_orig, F_orig, "Greys_r", outnoisy)
render_cathedral(V_ours, F_ours, "YlGnBu_r", outours)
render_cathedral(V_stein, F_stein, "PuRd_r", outstein)
render_cathedral(V_lapl2, F_lapl2, "YlOrRd_r", outlapl2)
render_cathedral(V_l1_laplacian, F_l1_laplacian, "Purples_r", outl1laplacian)