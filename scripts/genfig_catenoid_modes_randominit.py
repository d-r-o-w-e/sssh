import sys
sys.path.insert(0, '../src/')

from l1hessian import hessian_l1_modes, hessian_L2_modes
import gpytoolbox as gp
import numpy as np
import os
import blendertoolbox as bt
import bpy

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
        self.B = B # brightness
        self.C = C # contrast

def render_catenoid(modelfile, outpath, numpy_scalar, tx):
    res = [720, 720] if quick else [1080, 1080] # recommend >1080 for paper figures
    num_samples = 200 # recommend >200 for paper figures
    meshpos = (0.3, 0, 0.6) # UI: click mesh > Transform > Location
    meshrot = (0, 10.0, 0) # UI: click mesh > Transform > Rotation
    meshscale = (1.2,1.2,1.2) # UI: click mesh > Transform > Scale
    lightangle = (6, -30, -155) # UI: click Sun > Transform > Rotation

    bt.blenderInit(res[0], res[1], num_samples, 1.5)

    bt.invisibleGround(shadowBrightness=0.9)

    mesh = bt.readMesh(modelfile, meshpos, meshrot, meshscale)
    
    # mapping of the thing
    if tx == "../results/pngs/YlGnBu_r_isolinemap.png" and i == 0:
        print("zeroed out")
        numpy_scalar = np.zeros_like(numpy_scalar)
    
    mesh = bt.vertexScalarToUV(mesh, numpy_scalar)
    
    meshColor = bt.colorObj((0,0,0,1), 0.5, 1.3, 1.0, 0.0, 0.4)
    bt.setMat_texture(mesh, tx, meshColor)
    
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

# mesh choices

hesstype = "vfef"
mu = 1000000
mu_stein = 100
k = 8
eps = 5e-10
seed = 2

modelname = "catenoidlike"
modelfilename = "../models/catenoidlike.obj"
outnpy = "../results/npys/catenoidlike_modes_randominit_seed" + str(seed) + ".npy"

cmaptx = "../results/pngs/YlGnBu_r_isolinemap.png"
l2cmaptx = "../results/pngs/YlOrRd_r_isolinemap.png"

V, F = gp.read_mesh(modelfilename)

# divide by max hel
hel = gp.halfedge_lengths(V, F)
hel /= np.amax(gp.massmatrix(V, F).data)

if regen:
    modes, eigs = hessian_l1_modes(hel, F, k=k, mu=mu, eps=eps, hesstype=hesstype, randinit=True, seed=seed, verbose=True)
    l2modes, l2eigs = hessian_L2_modes(hel, F, k=k, mu=1.0, eps=eps, randinit=True, seed=seed, verbose=True)
    with open(outnpy, 'wb') as f:
        np.save(f, modes)
        np.save(f, eigs)
        np.save(f, l2modes)
        np.save(f, l2eigs)

with open(outnpy, 'rb') as f:
    print(outnpy)
    modes, eigs = np.load(f), np.load(f)
    l2modes, l2eigs = np.load(f), np.load(f)

print("l1 eigs")
print(eigs)
print("l2 eigs")
print(l2eigs)

render_catenoid(modelfilename, "../results/pngs/randominitcompressed_L2_" + modelname + "_mode" + str(0) + "_seed" + str(seed) +".png", np.zeros_like(l2modes[:, 0]), l2cmaptx)

for i in range(k):
    modei = modes[:, i]
    l2modei = l2modes[:, i]
    
    render_catenoid(modelfilename, "../results/pngs/randominitcompressed_" + modelname + "_mode" + str(i) + "_seed" + str(seed) +".png", modei, cmaptx)
    render_catenoid(modelfilename, "../results/pngs/randominitcompressed_L2_" + modelname + "_mode" + str(i+1) + "_seed" + str(seed) +".png", l2modei, l2cmaptx)