import sys
sys.path.insert(0, '../src/')

from l1hessian import hessian_l1_solve
from l1hessian import vfef
import gpytoolbox as gp
import numpy as np
import os
import blendertoolbox as bt
import bpy
from matplotlib import colormaps
from scipy.spatial.transform import Rotation

regen = True if int(sys.argv[1]) == 1 else False
quick = True if int(sys.argv[2]) == 1 else False

def render_cube(modelfile, logenfile, outpath):
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

# mesh choice
modelname = "cube"
modelfilename = "../models/cube_remesh.obj"

V0, F0 = gp.read_mesh(modelfilename)

hesstype = "vfef"

# noise the mesh
rng = np.random.default_rng(0)
V_noise = V0 + 0.01*rng.uniform(-1, 1, size=V0.shape)

# set V, F to V_noise, F
V, F = V_noise, F0

initmeshfilename = "../results/objs/initnoise_" + modelname + ".obj"
initlogendensfilename = "../results/npys/initnoise_logendens" + modelname + ".npy"
endmeshfilename = "../results/objs/endnoise_" + modelname + ".obj"
endlogendensfilename = "../results/npys/endnoise_logendens" + modelname + ".npy"

if regen:
    hel = gp.halfedge_lengths(V_noise, F0)
    hel = hel/np.amax(hel)
    u0x, u0y, u0z = V_noise[:, 0], V_noise[:, 1], V_noise[:, 2]
    H = vfef(hel, F0)
    Mfinv = 2.0/gp.doublearea_intrinsic(hel**2, F0)[:, None] # inverse mass matrix for visualisation, assumes M diagonal 
    energy_density = np.linalg.norm(np.reshape(H @ u0x, (-1, 4)), axis=1, keepdims=True) + \
        np.linalg.norm(np.reshape(H @ u0y, (-1, 4)), axis=1, keepdims=True) + \
        np.linalg.norm(np.reshape(H @ u0z, (-1, 4)), axis=1, keepdims=True)
    energy_density = (Mfinv * energy_density)[:, 0]
    log_energy_density = np.log(energy_density) 
    np.save(initlogendensfilename, log_energy_density)
    
    gp.write_mesh(initmeshfilename, V_noise, F0)

render_cube(initmeshfilename, initlogendensfilename, "../results/pngs/initnoise_" + modelname + ".png")

hesstype = "vfef"
fidelity_weight = 500 # weight on fidelity
iterations = 10 # frame count

R = np.array([[1, 0, 0],
                      [0, 0.7071068, -0.7071068],
                      [0, 0.7071068, 0.7071068]])
R = np.array([[0.7071068, 0, 0.7071068],
                [0, 1, 0],
                [-0.7071068, 0, 0.7071068]]) @ R
V = (R @ V.T).T

if regen:
    for i in range(iterations):
        print("iteration " + str(i))
        
        # now, iterate on the current geometry
        
        # get hel of current iteration; rescale to be within (0,1] range
        hel = gp.halfedge_lengths(V, F)
        hel = hel/np.amax(hel)
        
        # get the coordinates as vertex functions
        u0x, u0y, u0z = V[:, 0], V[:, 1], V[:, 2]
        
        # solve for the new coordinates as vertex functions
        u1x = hessian_l1_solve(hel, F, u0=u0x, mode=hesstype, alpha=fidelity_weight)
        u1y = hessian_l1_solve(hel, F, u0=u0y, mode=hesstype, alpha=fidelity_weight)
        u1z = hessian_l1_solve(hel, F, u0=u0z, mode=hesstype, alpha=fidelity_weight)
        
        # stack them into new vertex positions
        V = np.hstack([u1x[:, None], u1y[:, None], u1z[:, None]])
    
    V = (R.T @ V.T).T
    
    # save denoised result to file
    hel = gp.halfedge_lengths(V, F)
    hel = hel/np.amax(hel)
    u0x, u0y, u0z = V[:, 0], V[:, 1], V[:, 2]
    H = vfef(hel, F)
    Mfinv = 2.0/gp.doublearea_intrinsic(hel**2, F)[:, None] # inverse mass matrix for visualisation, assumes M diagonal 
    energy_density = np.linalg.norm(np.reshape(H @ u0x, (-1, 4)), axis=1, keepdims=True) + \
        np.linalg.norm(np.reshape(H @ u0y, (-1, 4)), axis=1, keepdims=True) + \
        np.linalg.norm(np.reshape(H @ u0z, (-1, 4)), axis=1, keepdims=True)
    energy_density = (Mfinv * energy_density)[:, 0]
    log_energy_density = np.log(energy_density) 
    np.save(endlogendensfilename, log_energy_density)
    
    gp.write_mesh(endmeshfilename, V, F)

render_cube(endmeshfilename, endlogendensfilename, "../results/pngs/endnoise_" + modelname + ".png")