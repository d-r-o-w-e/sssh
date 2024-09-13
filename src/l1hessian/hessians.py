import numpy as np
import gpytoolbox as gp
import scipy.sparse as sp

# ~~~ TENSOR PRODUCT HE CN TO FACE HESSIAN INTRINSIC VFEF (vertex-face-edge-face) HESSIAN

def face_d_intrinsic(hel, F):
    # u is n x 1
    # assume faces are oriented CCW; then compute the gradient of a function u in the gauge given by the vectors 0->1
    # u0, u1, u2 = u[F[:, 0], :], u[F[:, 1], :], u[F[:, 2], :]
    vshape = np.amax(F)+1
    
    lij, ljk, lki = hel[:, 2], hel[:, 0], hel[:, 1]
    coses = (lij**2 + lki**2 - ljk**2)/(2*lij*lki)
    sins = np.sqrt(1-coses**2)
    data = np.concatenate([-1.0/lij, 
                           1.0/lij, 
                           (lki*coses-lij)/(lij*lki*sins), 
                           -coses/(lij*sins),
                           1.0/(lki*sins)])
    row = np.concatenate([2*np.arange(F.shape[0]), 2*np.arange(F.shape[0]), 2*np.arange(F.shape[0])+1, 2*np.arange(F.shape[0])+1, 2*np.arange(F.shape[0])+1])
    col = np.concatenate([F[:, 0], F[:, 1], F[:, 0], F[:, 1], F[:, 2]])
    
    return sp.csr_array((data, (row, col)), shape=(2*F.shape[0], vshape))

def face_connection_intrinsic(hel, F):
    # Rotation matrix on face fj, minus identity matrix on face fi
    m = F.shape[0]
    
    # triangle validity / tip angles
    tt, tti = gp.triangle_triangle_adjacency(F)
    ta = gp.tip_angles_intrinsic(hel**2, F)
    
    # getting rotation angles between bases
    rot_to_he = np.hstack([(np.pi - ta[:, 1:2]), (2*np.pi - ta[:, 1:2] - ta[:, 2:3]), np.zeros((m, 1))]) # (i, j) = angle to rotate ith face's basis to be parallel to halfedge j
    opp_rot_to_he = rot_to_he[tt, tti]
    a = np.mod((rot_to_he - opp_rot_to_he + np.pi), (2.0*np.pi))
    th = np.where(a < np.pi, a, -(2*np.pi - a))
    
    # triplet list
    ttm = (tt.flatten() != -1)
    c, s = np.cos(th.flatten()), np.sin(th.flatten())
    
    data = np.concatenate([c, -s, 
                           s,  c, -np.ones_like(c), -np.ones_like(c)]) * np.tile(ttm, 6)
    row = np.concatenate([2*np.arange(3*m), 2*np.arange(3*m), 
                          2*np.arange(3*m)+1, 2*np.arange(3*m)+1, 2*np.arange(3*m), 2*np.arange(3*m)+1])
    col = np.concatenate([(2*tt.flatten())*ttm, (2*tt.flatten()+1)*ttm, 
                          (2*tt.flatten())*ttm, (2*tt.flatten()+1)*ttm, 2*np.repeat(np.arange(m), 3), 2*np.repeat(np.arange(m), 3)+1])
    
    return sp.csr_array((data, (row, col)), shape=(3*F.shape[0]*2, F.shape[0]*2)), th.flatten()

def tprod_he_cn_to_face_hessian_intrinsic(hel, F):
    # takes the (centroid) tensor product hessian on each face
    # see supplementary Mathematica file for some of the formulas used here
    # operates on a triangle flap region that looks like this:
    #    p----k----n
    #    \   / \  /
    #     \ /   \/
    #      i----j
    #       \  /
    #        \/
    #        m
    
    m = F.shape[0]
    
    lij, ljk, lki = hel[:, 2], hel[:, 0], hel[:, 1]
    
    ci = (lij*lij + lki*lki - ljk*ljk)/(2*lij*lki)
    si = np.sqrt(1 - ci**2)
    cj = (lij*lij + ljk*ljk - lki*lki)/(2*lij*ljk)
    sj = np.sqrt(1 - cj**2)
    
    Om = np.zeros((m,))
    lm = np.ones((m,))
    
    Sqrt = lambda x: np.sqrt(np.clip(x, 0, None))
    
    tt, tti = gp.triangle_triangle_adjacency(F)
    ttmask = tt >= 0
    ttmij, ttmjk, ttmki = ttmask[:, 2], ttmask[:, 0], ttmask[:, 1]
    
    # start by getting all the adjacent halfedge lengths
    ljn, lkp, lim = [hel[tt*ttmask, (tti+1) % 3][:, i] for i in range(3)]
    lnk, lpi, lmj = [hel[tt*ttmask, (tti+2) % 3][:, i] for i in range(3)]
    
    eij0 = (3*lij*(-lim**2 - ljk**2 + lki**2 + lmj**2))/ \
            (lij**4 - (ljk - lki)*(ljk + lki)*(lim - lmj)*(lim + lmj) - Sqrt(-((lij - ljk - lki)*(lij + ljk - lki)*(lij - ljk + lki)*(lij + ljk + lki)))* \
            Sqrt(-((lij - lim - lmj)*(lij + lim - lmj)*(lij - lim + lmj)*(lij + lim + lmj))) - lij**2*(lim**2 + ljk**2 + lki**2 + lmj**2))
    eij1 = -0.16666666666666666*(2*lki*Sqrt(1 - (lij**2 - ljk**2 + lki**2)**2/(4.*lij**2*lki**2)) + 2*lim*Sqrt(1 - (lij**2 + lim**2 - lmj**2)**2/(4.*lij**2*lim**2)))/ \
            ((lim**2 + ljk**2 - lki**2 - lmj**2)**2/(36.*lij**2) + ((lki*Sqrt(1 - (lij**2 - ljk**2 + lki**2)**2/(4.*lij**2*lki**2)))/3. + (lim*Sqrt(1 - (lij**2 + lim**2 - lmj**2)**2/(4.*lij**2*lim**2)))/3.)**2)
    
    ejk0 = (3*ljn*Sqrt(-((lij - ljk - lki)*(lij + ljk - lki)*(lij - ljk + lki)*(lij + ljk + lki)))*Sqrt(-(((-ljk + ljn - lnk)*(ljk + ljn - lnk)*(-ljk + ljn + lnk)*(ljk + ljn + lnk))/ljn**2)) - \
            3*(ljk - lki)*(ljk + lki)*(ljk**2 + ljn**2 - lnk**2) + 3*lij**2*(3*ljk**2 - ljn**2 + lnk**2))/ \
            (2.*lij*(-ljk**4 + (-lij**2 + lki**2)*(ljn - lnk)*(ljn + lnk) + ljn*Sqrt(-((lij - ljk - lki)*(lij + ljk - lki)*(lij - ljk + lki)*(lij + ljk + lki)))* \
            Sqrt(-(((-ljk + ljn - lnk)*(ljk + ljn - lnk)*(-ljk + ljn + lnk)*(ljk + ljn + lnk))/ljn**2)) + ljk**2*(lij**2 + ljn**2 + lki**2 + lnk**2)))
    ejk1 = (3*(lij**2*ljn*Sqrt(-(((-ljk + ljn - lnk)*(ljk + ljn - lnk)*(-ljk + ljn + lnk)*(ljk + ljn + lnk))/ljn**2)) + \
            ljn*(ljk - lki)*(ljk + lki)*Sqrt(-(((-ljk + ljn - lnk)*(ljk + ljn - lnk)*(-ljk + ljn + lnk)*(ljk + ljn + lnk))/ljn**2)) + \
            Sqrt(-((lij - ljk - lki)*(lij + ljk - lki)*(lij - ljk + lki)*(lij + ljk + lki)))*(ljk**2 + ljn**2 - lnk**2)))/ \
            (2.*lij*(-ljk**4 + (-lij**2 + lki**2)*(ljn - lnk)*(ljn + lnk) + ljn*Sqrt(-((lij - ljk - lki)*(lij + ljk - lki)*(lij - ljk + lki)*(lij + ljk + lki)))* \
            Sqrt(-(((-ljk + ljn - lnk)*(ljk + ljn - lnk)*(-ljk + ljn + lnk)*(ljk + ljn + lnk))/ljn**2)) + ljk**2*(lij**2 + ljn**2 + lki**2 + lnk**2)))
            
    eki0 = (-3*(Sqrt(-((lij - ljk - lki)*(lij + ljk - lki)*(lij - ljk + lki)*(lij + ljk + lki)))*lpi*Sqrt(-(((lki - lkp - lpi)*(lki + lkp - lpi)*(lki - lkp + lpi)*(lki + lkp + lpi))/lpi**2)) + \
            lij**2*(3*lki**2 + lkp**2 - lpi**2) + (ljk - lki)*(ljk + lki)*(lki**2 - lkp**2 + lpi**2)))/ \
            (2.*lij*(-lki**4 + (lij - ljk)*(lij + ljk)*(lkp - lpi)*(lkp + lpi) + Sqrt(-((lij - ljk - lki)*(lij + ljk - lki)*(lij - ljk + lki)*(lij + ljk + lki)))*lpi* \
            Sqrt(-(((lki - lkp - lpi)*(lki + lkp - lpi)*(lki - lkp + lpi)*(lki + lkp + lpi))/lpi**2)) + lki**2*(lij**2 + ljk**2 + lkp**2 + lpi**2)))
    eki1 = (3*(lij**2*lpi*Sqrt(-(((lki - lkp - lpi)*(lki + lkp - lpi)*(lki - lkp + lpi)*(lki + lkp + lpi))/lpi**2)) + (-ljk**2 + lki**2)*lpi*Sqrt(-(((lki - lkp - lpi)*(lki + lkp - lpi)*(lki - lkp + lpi)*(lki + lkp + lpi))/lpi**2)) + \
            Sqrt(-((lij - ljk - lki)*(lij + ljk - lki)*(lij - ljk + lki)*(lij + ljk + lki)))*(lki**2 - lkp**2 + lpi**2)))/ \
            (2.*lij*(-lki**4 + (lij - ljk)*(lij + ljk)*(lkp - lpi)*(lkp + lpi) + Sqrt(-((lij - ljk - lki)*(lij + ljk - lki)*(lij - ljk + lki)*(lij + ljk + lki)))*lpi* \
            Sqrt(-(((lki - lkp - lpi)*(lki + lkp - lpi)*(lki - lkp + lpi)*(lki + lkp + lpi))/lpi**2)) + lki**2*(lij**2 + ljk**2 + lkp**2 + lpi**2)))
    
    # DO THE TENSOR PRODUCT STUFF
    
    # dual edge lengths (the formulas for eij etc are of length 1 / dual edge length)
    dlij = 1.0/np.sqrt(eij0**2 + eij1**2)
    dljk = 1.0/np.sqrt(ejk0**2 + ejk1**2)
    dlki = 1.0/np.sqrt(eki0**2 + eki1**2)
    
    # unit length vectors in the dual edge directions
    tijx, tijy = dlij*eij0, dlij*eij1
    tjkx, tjky = dljk*ejk0, dljk*ejk1
    tkix, tkiy = dlki*eki0, dlki*eki1
    
    data = np.concatenate([(tjkx**3)/dljk,      (tjkx**2)*tjky/dljk, tkix**3/dlki,        (tkix**2)*tkiy/dlki, tijx**3/dlij,        (tijx**2)*tijy/dlij,
                           (tjkx**2)*tjky/dljk, tjkx*(tjky**2)/dljk, (tkix**2)*tkiy/dlki, tkix*(tkiy**2)/dlki, (tijx**2)*tijy/dlij, tijx*(tijy**2)/dlij,
                           (tjkx**2)*tjky/dljk, tjkx*(tjky**2)/dljk, (tkix**2)*tkiy/dlki, tkix*(tkiy**2)/dlki, (tijx**2)*tijy/dlij, tijx*(tijy**2)/dlij,
                           tjkx*(tjky**2)/dljk, (tjky**3)/dljk,      tkix*(tkiy**2)/dlki, (tkiy**3)/dlki,      tijx*(tijy**2)/dlij, (tijy**3)/dlij])
    all_ttms = np.concatenate([ttmjk, ttmjk, ttmki, ttmki, ttmij, ttmij, # multiply by masks
                                ttmjk, ttmjk, ttmki, ttmki, ttmij, ttmij,
                                ttmjk, ttmjk, ttmki, ttmki, ttmij, ttmij,
                                ttmjk, ttmjk, ttmki, ttmki, ttmij, ttmij])
    
    data = data * all_ttms # multiply by mask
    data = np.where(np.logical_and(~all_ttms, np.isnan(data)), 0, data) # if entry is nan and on boundary, set to 0
        
    row = np.concatenate([np.tile(4*np.arange(m), 6), 
                          np.tile(4*np.arange(m)+1, 6), 
                          np.tile(4*np.arange(m)+2, 6), 
                          np.tile(4*np.arange(m)+3, 6)])
    col = np.tile(np.concatenate([6*np.arange(m), 
                                  6*np.arange(m)+1, 
                                  6*np.arange(m)+2, 
                                  6*np.arange(m)+3, 
                                  6*np.arange(m)+4, 
                                  6*np.arange(m)+5]), 4)
    
    return sp.csr_array((data, (row, col)), shape=(4*F.shape[0], 3*F.shape[0]*2))

def circumcenter_tprod_he_cn_to_face_hessian_intrinsic(hel, F):
    # takes the (circumcenter) tensor product hessian on each face
    # see supplementary Mathematica file for some of the formulas used here
    # operates on a triangle flap region that looks like this:
    #    p----k----n
    #    \   / \  /
    #     \ /   \/
    #      i----j
    #       \  /
    #        \/
    #        m
    
    m = F.shape[0]
    
    lij, ljk, lki = hel[:, 2], hel[:, 0], hel[:, 1]
    
    ci = (lij*lij + lki*lki - ljk*ljk)/(2*lij*lki)
    si = np.sqrt(1 - ci**2)
    cj = (lij*lij + ljk*ljk - lki*lki)/(2*lij*ljk)
    sj = np.sqrt(1 - cj**2)
    
    Om = np.zeros((m,))
    lm = np.ones((m,))
    
    Sqrt = lambda x: np.sqrt(np.clip(x, 0, None))
    
    tt, tti = gp.triangle_triangle_adjacency(F)
    ttmask = tt >= 0
    ttmij, ttmjk, ttmki = ttmask[:, 2], ttmask[:, 0], ttmask[:, 1]
    
    # start by getting all the adjacent halfedge lengths
    ljn, lkp, lim = [hel[tt*ttmask, (tti+1) % 3][:, i] for i in range(3)]
    lnk, lpi, lmj = [hel[tt*ttmask, (tti+2) % 3][:, i] for i in range(3)]
    
    # unit vector entries crossing each edge
    tjkx = sj
    tjky = cj
    tkix = -si
    tkiy = ci
    tijx = 0
    tijy = -1
    
    # cotangent weights: "primal edge length over dual edge length"
    cots = gp.cotangent_weights_intrinsic(hel**2, F)
    
    # cot weights for this face
    hef1_dljk = cots[:, 0]
    hef1_dlki = cots[:, 1]
    hef1_dlij = cots[:, 2]
    
    # cot weights for the other face
    hef2_dljk = cots[ttmask[:, 0]*tt[:, 0], ttmask[:, 0]*tti[:, 0]]
    hef2_dlki = cots[ttmask[:, 1]*tt[:, 1], ttmask[:, 1]*tti[:, 1]]
    hef2_dlij = cots[ttmask[:, 2]*tt[:, 2], ttmask[:, 2]*tti[:, 2]]
    
    # add them up and mult by edge length to get the actual dual edge length (cot weight = dual edge length / edge length)
    dljk = np.abs(hef1_dljk + hef2_dljk) * ljk
    dlki = np.abs(hef1_dlki + hef2_dlki) * lki
    dlij = np.abs(hef1_dlij + hef2_dlij) * lij
    
    # DO THE TENSOR PRODUCT STUFF (same as in centroid method)
    
    data = np.concatenate([(tjkx**3)/dljk,      (tjkx**2)*tjky/dljk, tkix**3/dlki,        (tkix**2)*tkiy/dlki, tijx**3/dlij,        (tijx**2)*tijy/dlij,
                           (tjkx**2)*tjky/dljk, tjkx*(tjky**2)/dljk, (tkix**2)*tkiy/dlki, tkix*(tkiy**2)/dlki, (tijx**2)*tijy/dlij, tijx*(tijy**2)/dlij,
                           (tjkx**2)*tjky/dljk, tjkx*(tjky**2)/dljk, (tkix**2)*tkiy/dlki, tkix*(tkiy**2)/dlki, (tijx**2)*tijy/dlij, tijx*(tijy**2)/dlij,
                           tjkx*(tjky**2)/dljk, (tjky**3)/dljk,      tkix*(tkiy**2)/dlki, (tkiy**3)/dlki,      tijx*(tijy**2)/dlij, (tijy**3)/dlij])
    all_ttms = np.concatenate([ttmjk, ttmjk, ttmki, ttmki, ttmij, ttmij, # multiply by masks
                               ttmjk, ttmjk, ttmki, ttmki, ttmij, ttmij,
                               ttmjk, ttmjk, ttmki, ttmki, ttmij, ttmij,
                               ttmjk, ttmjk, ttmki, ttmki, ttmij, ttmij])
    
    data = data * all_ttms # multiply by mask
    data = np.where(np.logical_and(~all_ttms, np.isnan(data)), 0, data) # if entry is nan and on boundary, set to 0
        
    row = np.concatenate([np.tile(4*np.arange(m), 6), 
                          np.tile(4*np.arange(m)+1, 6), 
                          np.tile(4*np.arange(m)+2, 6), 
                          np.tile(4*np.arange(m)+3, 6)])
    col = np.tile(np.concatenate([6*np.arange(m), 
                                  6*np.arange(m)+1, 
                                  6*np.arange(m)+2, 
                                  6*np.arange(m)+3, 
                                  6*np.arange(m)+4, 
                                  6*np.arange(m)+5]), 4)
    
    return sp.csr_array((data, (row, col)), shape=(4*F.shape[0], 3*F.shape[0]*2))

def vfef(hel, F, center='centroid'):
    """Computes a 4|F| x |V| matrix taking vertex functions to 4 values which are squared, summed, and square-rooted per-face to integrate the l1 Hessian over the mesh.
    
    The output entries corresponding to face f are at indices 4f, 4f+1, 4f+2, 4f+3.

    Parameters
    ----------
    hel : numpy double array
          |F| x 3 array of halfedge lengths (NOT squared); halfedges are ordered as in gpytoolbox
    F   : numpy int array
          |F| x 3 face array of indices into a vertex list.
    center : string
        'centroid' or 'circumcenter' for the type of 

    Returns
    -------
    H : 4|F| x |V| scipy csr_matrix
        vfef hessian
    """

    fd = face_d_intrinsic(hel, F)
    fc, _ = face_connection_intrinsic(hel, F)
    if center == 'centroid':
        cn2f = tprod_he_cn_to_face_hessian_intrinsic(hel, F)
    elif center == 'circumcenter':
        cn2f = circumcenter_tprod_he_cn_to_face_hessian_intrinsic(hel, F)
    else:
        raise ValueError("Center type " + str(center) + " not supported.")
    Mf = sp.diags(np.repeat(0.5*gp.doublearea_intrinsic(hel**2, F), 4))
    return Mf @ cn2f @ fc @ fd

# ~~~ STEIN ET AL 2018 HESSIAN ~~~

def stein_et_al_2018(V, F):
    # from Stein et al 2018
    # 9 entries per interior vertex; non-intrinsic
    
    # get counts
    n = V.shape[0]
    m = F.shape[0]
    int_vtx = np.setdiff1d(np.arange(n), gp.boundary_vertices(F))
    i = int_vtx.shape[0]

    # construct matrices according to the 2018 paper
    G = gp.grad(V, F)
    A = sp.diags(np.tile(0.5*gp.doublearea(V, F), 3))
    Dsub = sp.hstack([G[:m, int_vtx], G[m:2*m, int_vtx], G[2*m:3*m, int_vtx]]) # Gx, Gy, Gz on interior vertices
    D = sp.block_diag([Dsub, Dsub, Dsub])
    H = D.T@A@G # v -> face x, face y, face z -> area scaled -> divergence 
    
    # reshape so that 9 adjacent entries are the hessian of that vertex
    s = np.reshape(np.reshape(np.arange(9*i), (9, -1)), (-1,), order='F')
    H = H[s, :]

    return H

# ~~~ HE ET AL 2013 EDGE LAPLACIAN ~~~

def edge_laplacian(V, F):
    # D(p) operator from He et al 2013 (see paper for full citation)
    
    he, E, he2E, E2he = gp.halfedge_edge_map(F)
    A = 0.5*gp.doublearea(V, F)
    
    # edge mask
    emask = np.logical_not(np.any(E2he[:, :, :] == -1, axis=(1, 2)))
    
    i1, i3, i2, i4 = E[:, 0], E[:, 1], F[E2he[np.arange(E.shape[0]), 0, 0], E2he[np.arange(E.shape[0]), 0, 1]], F[E2he[np.arange(E.shape[0]), 1, 0], E2he[np.arange(E.shape[0]), 1, 1]]
    
    Del123, Del134 = A[E2he[np.arange(E.shape[0]), 0, 0]], A[E2he[np.arange(E.shape[0]), 1, 0]]
    
    p1w = (Del123*np.sum((V[i4, :]-V[i3, :])*(V[i3, :]-V[i1, :]), axis=1) 
        + Del134*np.sum((V[i1, :]-V[i3, :])*(V[i3, :]-V[i2, :]), axis=1)) / (np.linalg.norm(V[i3, :]-V[i1, :], axis=1)**2 * (Del123+Del134))
    p2w = Del134/(Del123+Del134)
    p3w = (Del123*np.sum((V[i3, :]-V[i1, :])*(V[i1, :]-V[i4, :]), axis=1) 
        + Del134*np.sum((V[i2, :]-V[i1, :])*(V[i1, :]-V[i3, :]), axis=1))/(np.linalg.norm(V[i3, :]-V[i1, :], axis=1)**2 * (Del123+Del134))
    p4w = Del123/(Del123+Del134)
    
    p1w = p1w * emask
    p2w = p2w * emask
    p3w = p3w * emask
    p4w = p4w * emask
    
    data = np.concatenate([p1w, p2w, p3w, p4w])
    row = np.concatenate([np.arange(E.shape[0]),
                          np.arange(E.shape[0]),
                          np.arange(E.shape[0]),
                          np.arange(E.shape[0])])
    col = np.concatenate([i1, i2, i3, i4])
    
    return sp.csr_array((data, (row, col)), shape=(E.shape[0], V.shape[0]))
