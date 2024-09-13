import sys
sys.path.insert(0, '../src/')

from l1hessian import at_hess1_segment_color, thresh_segmentation_from_v_diffusion
import gpytoolbox as gp
import numpy as np
import polyscope as ps
import os
import cv2
import blendertoolbox as bt
import bpy

regen = True if int(sys.argv[1]) == 1 else False
quick = True if int(sys.argv[2]) == 1 else False

meshpos = (0.64, 0.02, 0.01)
meshrot = (0, 0, 64)
meshscale = (0.7, 0.7, 0.7)

def image_sampler(u0):
    # rearrange so we look at it as x, y coords
    xy0 = np.transpose(u0[::-1, :, :], (1, 0, 2))
    
    def sample(xy):
        # takes in xy coordinate array (n, 2) and returns the color values (n, 3) at those points using bilinear interpolation
        xys = xy*np.array([[u0.shape[0], u0.shape[1]]])-0.5 # do -1 then + 0.5
        
        x0, x1 = np.int32(np.floor(xys[:, 0])), np.int32(np.ceil(xys[:, 0]))
        y0, y1 = np.int32(np.floor(xys[:, 1])), np.int32(np.ceil(xys[:, 1]))
        
        x0 = np.clip(x0, 0, u0.shape[0]-1)
        x1 = np.clip(x1, 0, u0.shape[0]-1)
        y0 = np.clip(y0, 0, u0.shape[0]-1)
        y1 = np.clip(y1, 0, u0.shape[0]-1)
        
        w00, w01, w10, w11 = ((x1-xys[:, 0])*(y1-xys[:, 1]), 
                              (x1-xys[:, 0])*(xys[:, 1]-y0),
                              (xys[:, 0]-x0)*(y1-xys[:, 1]),
                              (xys[:, 0]-x0)*(xys[:, 1]-y0))
        
        return w00[:, None]*xy0[x0, y0, :] + w01[:, None]*xy0[x0, y1, :] + w10[:, None]*xy0[x1, y0, :] + w11[:, None]*xy0[x1, y1, :]
    
    return sample

def render_diffusion(Vis, Fis, sis, svs, outpath):
    res = [720, 720] if quick else [1080, 1080] # recommend >1080 for paper figures
    num_samples = 200 # recommend >200 for paper figures
    
    lightangle = (6, -30, -155) # UI: click Sun > Transform > Rotation

    bt.blenderInit(res[0], res[1], num_samples, 1.5)

    bt.invisibleGround(shadowBrightness=0.9)

    allB = []
    for i in range(len(Vis)):
        Vi, Fi, si, sv = Vis[i], Fis[i], sis[i], svs[i]
        
        allB += [Vi[gp.boundary_vertices(Fi), :]]
        
        L = gp.cotangent_laplacian(Vi, Fi)
        
        newcolor = gp.min_quad_with_fixed(L, k=si, y=sv)
        
        mesh = bt.readNumpyMesh(Vi, Fi, meshpos, meshrot, meshscale)
        
        for f in mesh.data.polygons:
            f.use_smooth=True
            
        color_type = 'vertex'
        mesh = bt.setMeshColors(mesh, newcolor, color_type)
        meshVColor = bt.colorObj([], 0.5, 1.0, 1.0, 0.0, 0.0)
        bt.setMat_VColor(mesh, meshVColor)
    
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

def render_strav(modelfile, tx, outpath):
    res = [720, 720] if quick else [1080, 1080] # recommend >1080 for paper figures
    num_samples = 200 # recommend >200 for paper figures
    lightangle = (6, -30, -155) # UI: click Sun > Transform > Rotation

    bt.blenderInit(res[0], res[1], num_samples, 1.5)

    bt.invisibleGround(shadowBrightness=0.9)

    mesh = bt.readMesh(modelfile, meshpos, meshrot, (meshscale[0]/Vmult, meshscale[1]/Vmult, meshscale[2]/Vmult))
    
    meshColor = bt.colorObj((0,0,0,1), 0.5, 1.0, 1.0, 0.0, 0.0)
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

def render_diffusion_curves(Vis, Fis, sis, svs, outpath):
    res = [720, 720] if quick else [1080, 1080] # recommend >1080 for paper figures
    num_samples = 200 # recommend >200 for paper figures
    
    lightangle = (6, -30, -155) # UI: click Sun > Transform > Rotation

    bt.blenderInit(res[0], res[1], num_samples, 1.5)

    bt.invisibleGround(shadowBrightness=0.9)

    allB = []
    for i in range(len(Vis)):
        Vi, Fi, si, sv = Vis[i], Fis[i], sis[i], svs[i]
        
        allB += [Vi[gp.boundary_vertices(Fi), :]]
        
        newcolor = 0.4*np.ones((Vi.shape[0], 3))
        newcolor[si, :] = sv
        
        mesh = bt.readNumpyMesh(Vi, Fi, meshpos, meshrot, meshscale)
        
        for f in mesh.data.polygons:
            f.use_smooth=True
            
        color_type = 'vertex'
        mesh = bt.setMeshColors(mesh, newcolor, color_type)
        meshVColor = bt.colorObj([], 0.5, 1.0, 1.0, 0.0, 0.0)
        bt.setMat_VColor(mesh, meshVColor)
    
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

def render_vtx_colors(modelfile, vcolors, outpath):
    res = [720, 720] if quick else [1080, 1080] # recommend >1080 for paper figures
    num_samples = 200 # recommend >200 for paper figures
    lightangle = (6, -30, -155) # UI: click Sun > Transform > Rotation

    bt.blenderInit(res[0], res[1], num_samples, 1.5)

    bt.invisibleGround(shadowBrightness=0.9)

    mesh = bt.readMesh(modelfile, meshpos, meshrot, (meshscale[0]/Vmult, meshscale[1]/Vmult, meshscale[2]/Vmult))
    
    color_type = 'vertex'
    mesh = bt.setMeshColors(mesh, vcolors, color_type)
    meshVColor = bt.colorObj([], 0.5, 1.0, 1.0, 0.0, 0.0)
    bt.setMat_VColor(mesh, meshVColor)
    
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
filename_npy = "../results/npys/stravinsky_segdiffuse.npy"
filename_original = "../results/pngs/stravinsky_original.png"
filename_vertcolors = "../results/pngs/stravinsky_sampled2verts.png"
filename_diffused = "../results/pngs/stravinsky_diffused.png"
filename_diffuse_curves = "../results/pngs/stravinsky_diffuse_curves.png"
filename_vertcolors = "../results/pngs/stravinsky_vtx_colors.png"

meshname = "stravinsky"

modelfilename = "../models/stravinsky1.obj"
texturefilename = "../models/stravinsky1_tex.png"
resize_to = 1024
roworder = 1

V, F, UV, Ft = gp.read_mesh(modelfilename, return_UV=True)


Vmult = np.mean(np.linalg.norm(V, axis=1))
V /= Vmult

# set offset
xrange = np.amax(V[:, 0]) - np.amin(V[:, 0])
yrange = np.amax(V[:, 1]) - np.amin(V[:, 1])
zrange = np.amax(V[:, 2]) - np.amin(V[:, 2])

# load and sample texture at vertices
tx = cv2.resize(cv2.imread(texturefilename)[::roworder, :, ::-1], (resize_to, resize_to)).astype(float) / 255.0 
txsamp = image_sampler(tx)
vert_lookup = np.zeros((V.shape[0], 2), dtype=np.int32)
vert_lookup[F[:, 0], 0] = np.arange(F.shape[0]); vert_lookup[F[:, 0], 1] = 0
vert_lookup[F[:, 1], 0] = np.arange(F.shape[0]); vert_lookup[F[:, 1], 1] = 1
vert_lookup[F[:, 2], 0] = np.arange(F.shape[0]); vert_lookup[F[:, 2], 1] = 2
vertpos = UV[Ft[vert_lookup[:, 0], vert_lookup[:, 1]]]
vert_colors = txsamp(vertpos)

hesstype = "vfef"
lam = 2.0
alpha = 10000.0
eps1 = 0.1 # starting epsilon in A-T formula
eps2 = 0.0001 # ending epsilon in A-T formula
eps3 = 1e-4
print("using eps3 = " + str(eps3))
n = 30
cut_thresh = 0.95
merge_limit = 4

if regen:
    u, v = at_hess1_segment_color(V, F, vert_colors, lam=lam, alpha=alpha, eps1=eps1, eps2=eps2, eps3=eps3, n=n, verbose=True)
    print("CUTTING MESH")
    Vis, Fis, sis, svs = thresh_segmentation_from_v_diffusion(V, 
                                                              F, 
                                                              v, 
                                                              UV, 
                                                              Ft, 
                                                              txsamp, 
                                                              vertcolors=vert_colors, 
                                                              cut_thresh=cut_thresh, 
                                                              merge_limit=merge_limit)

    np.save(filename_npy, np.array([Vis, Fis, sis, svs], dtype=object), allow_pickle=True)

VisFis = np.load(filename_npy, allow_pickle=True)
Vis, Fis, sis, svs = VisFis[0], VisFis[1], VisFis[2], VisFis[3]

render_strav(modelfilename, texturefilename, filename_original)
render_diffusion(Vis, Fis, sis, svs, filename_diffused)
render_diffusion_curves(Vis, Fis, sis, svs, filename_diffuse_curves)
render_vtx_colors(modelfilename, vert_colors, filename_vertcolors)