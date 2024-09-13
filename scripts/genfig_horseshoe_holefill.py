# TODO: MAKE THIS WORK
import sys
sys.path.insert(0, '../src/')

from l1hessian import hessian_l1_solve
from l1hessian import vfef
import gpytoolbox as gp
import numpy as np
import os
from scipy.spatial.transform import Rotation
from hole_filling_liepa.core import fill_hole_liepa, find_boundary_loops
import blendertoolbox as bt
import bpy
from matplotlib import colormaps

regen = True if int(sys.argv[1]) == 1 else False
quick = True if int(sys.argv[2]) == 1 else False

def not_si(V, si):
    return np.setdiff1d(np.arange(V.shape[0]), si)

def not_si_and_b(V, nb, sample_idx):
    nsi = not_si(V, sample_idx)
    
    bv = set(nsi) # boundary vertices
    for i in range(nsi.shape[0]):
        bv |= set(nb[nsi[i]])
    return np.sort(np.array(list(bv)))

def mesh_restrict(V, F, si):
    sis = set(list(si))
    F_fine = []
    for i in range(F.shape[0]):
        if (F[i, 0] in sis) and (F[i, 1] in sis) and (F[i, 2] in sis):
            F_fine += [i]
    Fnew = F[np.array(F_fine), :]
    return gp.remove_unreferenced(V, Fnew)

def slice_plane_indices(V, F, c, bias):
    Hi = (V@c - bias > 0)[:, 0].nonzero()[0]
    print(Hi.shape)
    return sisv_from_hole(V, F, Hi)

def sisv_from_hole(V, F, Hi):
    # Hi is the index of all the vertices in the hole.  get si (n-h,) as the indices outside of this region, and sv (n-h, 3) as the position values
    si = np.setdiff1d(np.arange(V.shape[0]), Hi)
    return si, V[si, :]

def build_neighbs(V, F):
    neighbs = {i:set() for i in range(V.shape[0])}
    for j in range(F.shape[0]):
        neighbs[F[j, 0]] |= {F[j, 1], F[j, 2]}
        neighbs[F[j, 1]] |= {F[j, 2], F[j, 0]}
        neighbs[F[j, 2]] |= {F[j, 0], F[j, 1]}
    return {i:list(neighbs[i]) for i in neighbs.keys()}

def cut_ball_indices(V, F, c, r):
    # c 3x1, r float
    Hi = (np.linalg.norm(V-c.T, axis=1) < r).nonzero()
    return sisv_from_hole(V, F, Hi)

def render_horseshoe(modelfile, logenfile, outpath):
    res = [720, 720] if quick else [1080, 1080] # recommend >1080 for paper figures
    num_samples = 200 # recommend >200 for paper figures
    meshpos = (0.33, -0.02, 0.33) # UI: click mesh > Transform > Location
    meshrot = (950, 151, -605) # UI: click mesh > Transform > Rotation
    meshscale = (0.9, 0.9, 0.9) # UI: click mesh > Transform > Scale
    lightangle = (6, -30, -155) # UI: click Sun > Transform > Rotation

    bt.blenderInit(res[0], res[1], num_samples, 1.5)

    bt.invisibleGround(shadowBrightness=0.9)

    mesh = bt.readMesh(modelfile, meshpos, meshrot, meshscale)
    
    vsif = np.load(logenfile, allow_pickle=True)
    vertex_scalars, sif = vsif[0].astype(float), vsif[1].astype(int)
    print("max and min of the current log_energy_density file")
    print(np.amin(vertex_scalars))
    print(np.amax(vertex_scalars))
    vertex_scalars = np.clip((vertex_scalars - (-7))/(14), 0, 1)
    vertex_scalars = 0.5*np.ones_like(vertex_scalars)
    # vertex_scalars = (vertex_scalars - vertex_scalars.min()) / (vertex_scalars.max() - vertex_scalars.min())
    # vertex_scalars /= np.amax(vertex_scalars)
    cmap_inner = colormaps["YlGnBu_r"]
    # cmap_outer = colormaps["Greys_r"]
    
    # fcolors = np.where(sif[:, None], cmap_inner(vertex_scalars), cmap_outer(vertex_scalars))
    fcolors = np.where(sif[:, None], cmap_inner(vertex_scalars), 0.4)
    
    color_type = 'face'
    mesh = bt.setMeshColors(mesh, fcolors, color_type)
    meshVColor = bt.colorObj([], 0.5, 1.0, 1.0, 0.0, 0.0)
    setMat_VColor(mesh, meshVColor)
    
    
    # bt.setMat_singleColor(mesh, bt.colorObj(tuple(meshc + [1.0]), 0.5, 1.0, 1.0, 0.0, 2.0), 0.0)
    
    # vertex_scalars = np.load(logenfile)
    # mesh = bt.vertexScalarToUV(mesh, vertex_scalars)
    # useless = (0,0,0,1)
    # meshColor = bt.colorObj(useless, 0.5, 1.3, 1.0, 0.0, 0.4)
    # texturePath = '../meshes/YlGnBu_r_isolinemap.png' 
    # bt.setMat_texture(mesh, texturePath, meshColor)
    
    # bt.setMat_plastic(mesh, meshVColor, 0.0)
    
    # edgeThickness = 0.001
    # edgeColor = bt.colorObj((0,0,0,0),0.5, 1.0, 1.0, 0.0, 0.0)
    # meshRGBA = tuple(meshc + [1.0])
    # AOStrength = 1.0
    # setMat_edge(mesh, edgeThickness, edgeColor, meshRGBA, AOStrength)
    
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
	tree.nodes["Principled BSDF"].inputs['Roughness'].default_value = 0.25
	tree.nodes["Principled BSDF"].inputs['Sheen Tint'].default_value = [0, 0, 0, 1]
	tree.links.new(HSVNode.outputs['Color'], BCNode.inputs['Color'])
	tree.links.new(BCNode.outputs['Color'], tree.nodes['Principled BSDF'].inputs['Base Color'])


# PARAMETERS WE CAN SET
hesstype = "vfef"

setup = "horseshoe2"

initstyle = "lapl2" # liepa then biharmonic smooth

randrot = False
remeshiter = 8
remesh_every_nth_iteration = False

solvestyle = "meshflow"; iterations = 80; fidelity_weight = 100

# THE ACTUAL PROGRAM

V, F = gp.read_mesh("../models/horseshoe2.obj")
c = np.array([[-0.57, 0.45, 0.67]]).T
r = 0.43
si, sv = cut_ball_indices(V, F, c, r)

or_si, or_sv = si, sv # keep original si and sv
Vouter, Fouter = mesh_restrict(V, F, or_si)
boundary_loops = find_boundary_loops(Fouter)
bselect = 0
patch_faces = fill_hole_liepa(Vouter, Fouter, boundary_loops[bselect], method="angle")
Vhf, Fhf = gp.remesh_botsch(Vouter,
                            np.vstack([Fouter, patch_faces]),
                            i=20,
                            h=np.mean(0.5*gp.doublearea(Vouter[boundary_loops[bselect]], None)),
                            feature=np.setdiff1d(np.arange(Vouter.shape[0]), boundary_loops[bselect]),
                            project=True)

# redefine the mesh si, sv, etc according to liepa holefill
si = np.arange(np.setdiff1d(np.arange(Vouter.shape[0]), boundary_loops[bselect]).shape[0])
sv = Vhf[si, :]
nsi = not_si(Vhf, si)
nsab = not_si_and_b(Vhf, build_neighbs(Vhf, Fhf), si)
# outside of the hole
Vouter, Fouter = mesh_restrict(Vhf, Fhf, si)


# initialise the hole in various different ways
if initstyle == "original":
    # ignore all the stuff from above, initialize using original mesh
    Vhf, Fhf = V, F
    si, sv = or_si, or_sv
    nsi = not_si(Vhf, si)
    nsab = not_si_and_b(Vhf, build_neighbs(Vhf, Fhf), si)
    # outside of the hole
    Vouter, Fouter = mesh_restrict(Vhf, Fhf, si)
elif initstyle == "flat":
    # do nothing.  keep as-is, using the liepa fill
    1
elif initstyle == "lapl2":
    # smooth the liepa fill a bit with bilaplacian
    Vhf = gp.min_quad_with_fixed(gp.biharmonic_energy(Vhf, Fhf), k=si, y=sv)

Vhf0, Fhf0 = Vhf, Fhf

# indicator for faces, for visualisation.  assumes no remeshing every iteration
sif = np.zeros(Fhf.shape[0])
si_ind = np.ones(Vhf.shape[0])
si_ind[si] = 0
sif = np.any(si_ind[Fhf], axis=1)

initholefillfilename = "../results/objs/initholefill_" + setup + ".obj"
initlogendensfilename = "../results/npys/initholefill_logendens" + setup + ".npy"
endholefillfilename = "../results/objs/endholefill_" + setup + ".obj"
endlogendensfilename = "../results/npys/endholefill_logendens" + setup + ".npy"

if regen:
    # write initial hole filled energy density
    hel = gp.halfedge_lengths(Vhf0, Fhf0)
    hel = hel/np.amax(hel)
    u0x, u0y, u0z = Vhf0[:, 0], Vhf0[:, 1], Vhf0[:, 2]
    H = vfef(hel, Fhf0)
    Mfinv = 2.0/gp.doublearea_intrinsic(hel**2, Fhf0)[:, None] # inverse mass matrix for visualisation, assumes M diagonal 
    energy_density = np.linalg.norm(np.reshape(H @ u0x, (-1, 4)), axis=1, keepdims=True) + \
        np.linalg.norm(np.reshape(H @ u0y, (-1, 4)), axis=1, keepdims=True) + \
        np.linalg.norm(np.reshape(H @ u0z, (-1, 4)), axis=1, keepdims=True)
    energy_density = (Mfinv * energy_density)[:, 0]
    log_energy_density = np.log(energy_density)
    np.save(initlogendensfilename, np.array([log_energy_density, sif], dtype=object), allow_pickle=True)
    
    # write initial hole filled mesh
    gp.write_mesh(initholefillfilename, Vhf0, Fhf0)

render_horseshoe(initholefillfilename, initlogendensfilename, "../results/pngs/initholefill_" + setup + ".png")

# get initial hel
# get hel of current iteration; rescale to be within (0,1] range
hel = gp.halfedge_lengths(Vhf, Fhf)
hel = hel/np.amax(hel)

# use Vhf, Fhf, si, sv to solve
if regen:
    if solvestyle == "meshflow":
        for i in range(iterations):
            print("iteration " + str(i))
            
            H = vfef(hel, Fhf)
            Mfinv = 2.0/gp.doublearea_intrinsic(hel**2, Fhf)[:, None] # inverse mass matrix for visualisation, assumes M diagonal 
            u0x, u0y, u0z = Vhf[:, 0], Vhf[:, 1], Vhf[:, 2]
            energy_density = np.linalg.norm(np.reshape(H @ u0x, (-1, 4)), axis=1, keepdims=True) + \
                np.linalg.norm(np.reshape(H @ u0y, (-1, 4)), axis=1, keepdims=True) + \
                np.linalg.norm(np.reshape(H @ u0z, (-1, 4)), axis=1, keepdims=True)
            energy_density = (Mfinv * energy_density)[:, 0]
            log_energy_density = np.log(energy_density) 
            
            # now, iterate on the current geometry
            
            if remesh_every_nth_iteration and i % remeshiter == 0:
                Vhf, Fhf = gp.remesh_botsch(Vhf, Fhf,
                                            i=10,
                                            feature=si,
                                            project=True)
                # set si, sv to new values
                si = np.arange(si.shape[0])
                sv = Vhf[si, :]
            
            # apply a random rotation
            if randrot:
                R = Rotation.random().as_matrix()
                Vhf = (R @ Vhf.T).T
                sv = (R @ sv.T).T
            
            # get hel of current iteration; rescale to be within (0,1] range
            hel = gp.halfedge_lengths(Vhf, Fhf)
            hel = hel/np.amax(hel)
            
            # get the coordinates as vertex functions
            u0x, u0y, u0z = Vhf[:, 0], Vhf[:, 1], Vhf[:, 2]
            
            # solve for the new coordinates as vertex functions
            u1x = hessian_l1_solve(hel, Fhf, u0=u0x, mode=hesstype, alpha=fidelity_weight, y=si, k=sv[:, 0])
            u1y = hessian_l1_solve(hel, Fhf, u0=u0y, mode=hesstype, alpha=fidelity_weight, y=si, k=sv[:, 1])
            u1z = hessian_l1_solve(hel, Fhf, u0=u0z, mode=hesstype, alpha=fidelity_weight, y=si, k=sv[:, 2])
            
            # stack them into new vertex positions
            Vhf = np.hstack([u1x[:, None], u1y[:, None], u1z[:, None]])
            
            # rotate back
            if randrot:
                Rinv = np.linalg.inv(R)
                Vhf = (Rinv @ Vhf.T).T
                sv = (Rinv @ sv.T).T
            
    # save final iteration log energy density and final hole filled mesh
    hel = gp.halfedge_lengths(Vhf, Fhf)
    hel = hel/np.amax(hel)
    u0x, u0y, u0z = Vhf[:, 0], Vhf[:, 1], Vhf[:, 2]
    H = vfef(hel, Fhf)
    Mfinv = 2.0/gp.doublearea_intrinsic(hel**2, Fhf)[:, None] # inverse mass matrix for visualisation, assumes M diagonal 
    energy_density = np.linalg.norm(np.reshape(H @ u0x, (-1, 4)), axis=1, keepdims=True) + \
        np.linalg.norm(np.reshape(H @ u0y, (-1, 4)), axis=1, keepdims=True) + \
        np.linalg.norm(np.reshape(H @ u0z, (-1, 4)), axis=1, keepdims=True)
    energy_density = (Mfinv * energy_density)[:, 0]
    log_energy_density = np.log(energy_density) 
    np.save(endlogendensfilename, np.array([log_energy_density, sif], dtype=object), allow_pickle=True)
    
    gp.write_mesh(endholefillfilename, Vhf, Fhf)
    
render_horseshoe(endholefillfilename, endlogendensfilename, "../results/pngs/endholefill_" + setup + ".png")