import numpy as np
import scipy.sparse as sp
import gpytoolbox as gp
from scipy.stats import mode

from l1hessian.hessians import vfef, stein_et_al_2018, edge_laplacian
from l1hessian.l1_opt import min_fauxl1_d_with_lin, min_quadfauxl1_d_with_lin, min_quadl1_with_lin, min_l1_with_lin

def hessian_l1_solve(hel, F, u0=None, mode="vfef", center="centroid", verbose=False, tol=None, alpha=0, beta=0, y=None, k=None, A=None, b=None, Q=None, g=None, V=None, M=None):
    """ Solve a problem of the form 
        argmin_u  |g|.T @ (l1 hessian of u) + alpha*||u-u0||^2 + beta*(0.5 u^T Q u)
            s.t.  u[y] == k
                  A u  <= b
        
        g is a multiplier on the hessian when mode="vfef" (default)
        for mode="stein et al 2018", V is required because their energy isn't intrinsic.
        can also provide a mass matrix M to use instead of the one created here (e.g. for testing a "conformal l1 hessian flow")
    """

    Vshape = np.amax(F)+1

    if not (tol is None):
        raise ValueError("tolerance has no effect right now")

    # construct a matrix forcing outside-of-the-hole to be correct values
    if not (y is None) and not (k is None):
        data, row, col = [1]*(y.shape[0]) + [-1]*(y.shape[0]), list(range(2*y.shape[0])), list(np.tile(y, 2))
        A0 = sp.csr_matrix((data, (row, col)), shape=(2*k.shape[0], Vshape))
        b0 = np.vstack([k[:, None], -k[:, None]])
        
        # if A and b are provided, stack them with the sample index and value matrices
        if not (A is None) and not (b is None):
            A = sp.vstack([A0, A])
            b = sp.vstack([b0, b])
        else:
            A = A0
            b = b0
    
    # choose hessian based on selection
    if mode == "vfef":
        H = vfef(hel, F, center)
        d = 4
    elif mode == "stein et al 2018":
        if V is None:
            raise ValueError("Stein et al 2018 is ~not~ intrinsic, and requires vertex positions V to be provided !")
        H = stein_et_al_2018(V, F)
        d = 1
    elif mode == "l1_laplacian":
        H = gp.cotangent_laplacian_intrinsic(hel**2, F)
        d=1
    elif mode == "l1_edgelaplacian":
        if V is None:
            raise ValueError("Edge laplacian is not defined intrinsically, and requires vertex positions V to be provided !")
        H = edge_laplacian(V, F)
        d=1
    else:
        raise ValueError("chosen hessian has not been implemented")
    
    # construct a Q matrix to send to the solver
    M = gp.massmatrix_intrinsic(hel**2, F) if (M is None) else M
    Q_solve = 2*alpha*M if (Q is None) else 2*alpha*M + beta*Q

    # send to l1 (no d) solver if its stein or a laplacian
    if mode == "stein et al 2018" or mode == "l1_laplacian" or mode == "l1_edgelaplacian":
        if alpha != 0 or beta != 0:
            if alpha == 0:
                return min_quadl1_with_lin(Q=Q_solve, G=H, A=A, b=b)[:, 0]
            else:
                return min_quadl1_with_lin(Q=Q_solve, G=H, A=A, b=b, c=-2*alpha*M@u0[:, None], verbose=verbose)[:, 0]
        else:
            return min_l1_with_lin(G=H, A=A, b=b, si=y, sv=k, d=g, verbose=verbose)[:, 0]
            
    # otherwise, send to mosek solver and return
    if alpha != 0 or beta != 0:
        if alpha == 0:
            return min_quadfauxl1_d_with_lin(Q=Q_solve, G=H, A=A, b=b, d=d, g=g, verbose=verbose)[:, 0]
        else:
            return min_quadfauxl1_d_with_lin(Q=Q_solve, G=H, A=A, b=b, c=-2*alpha*M@u0[:, None], d=d, g=g, verbose=verbose)[:, 0]
    else:
        return min_fauxl1_d_with_lin(G=H, A=A, b=b, d=d, g=g, verbose=verbose)[:, 0]

def hessian_l1_modes(hel, F, k=5, mu=100, eps=1e-10, hesstype="vfef", randinit=False, seed=-1, verbose=False):
    # input:
    # k = number of modes to output
    # mu = weight on the l1 hessian term
    # eps = tolerance for the solver
    #
    # output:
    # l1_modes: |V| x k array, where column i is the ith "eigenmode"
    # l1_eigs: (|V|,) array, entry i is the "eigenvalue" corresponding to the ith "eigenmode"
    #
    #
    # based on:
    #   Brandt and Hildebrandt 2017 "Compressed vibration modes of elastic bodies"
    #   Sellan et al 2022 "Breaking Good: Fracture Modes for Realtime Destruction"
    
    # construct matrices
    M = gp.massmatrix_intrinsic(hel**2, F)
    L = gp.cotangent_laplacian_intrinsic(hel**2, F)
    if hesstype == "vfef":
        H = vfef(hel, F)
    else:
        raise ValueError("hesstype %s not supported" % hesstype)
    
    if randinit:
        if seed >= 0:
            rng = np.random.default_rng(seed)
        elif seed == -1:
            rng = np.random.default_rng()
        U_init = rng.random((L.shape[0], k))*2 - 1.0
    else:
        # initialize U with the eigenmodes of L
        eigs = sp.linalg.eigsh(L, k=k, M=M, sigma=0, which="LM")[1]
        U_init = eigs
    
    l1_eigs = np.zeros(k)
    
    if verbose: print("calculating modes for omega = " + str(mu) + "; seed = " + str(seed))
    for i in range(k):
        if verbose: print("calculating mode " + str(i))
        
        # initialize with the ith eigenmode of L
        ci = U_init[:, i:i+1]
        
        while 1:
            # equality constraint matrices
            if i != 0:
                A = np.vstack([modes, ci.T]) @ M
                b = np.concatenate([np.zeros(modes.shape[0]), np.ones(1)])[:, None]
            else:
                A = ci.T @ M
                b = np.ones((1, 1))
            
            A_eq, b_eq = np.vstack([A, -A]), np.vstack([b, -b])
            
            Ui = min_quadfauxl1_d_with_lin(Q=L, G=mu*H, A=A_eq, b=b_eq, d=4)
            
            if verbose: print("  loss: " + str(0.5*Ui.T @ L @ Ui + mu*np.sum(np.linalg.norm(np.reshape(H @ Ui, (-1, 4)), axis=1))))
            
            ci = Ui / np.sqrt((Ui.T @ M @ Ui)[0, 0])
            
            # check termination criteria
            diff = np.linalg.norm(Ui-ci)
            if verbose: print("    |Ui-ci|: " + str(diff))
            if diff < eps:
                break
        
        # append the newly-found Ui
        if i != 0:
            modes = np.vstack([modes, Ui.T])
        else:
            modes = Ui.T
            
        l1_eigs[i] = (0.5*Ui.T @ L @ Ui) + mu*np.sum(np.linalg.norm(np.reshape(H @ Ui, (-1, 4)), axis=1))
        
    return modes.T, l1_eigs

def hessian_L2_modes(hel, F, k=5, mu=1.0, eps=1e-10, randinit=False, seed=-1, verbose=False):
    # input:
    # k = number of modes to output
    # mu = weight on the l1 hessian term
    # eps = tolerance for the solver
    #
    # output:
    # l2_modes: |V| x k array, where column i is the ith "eigenmode"
    # l2_eigs: (|V|,) array, entry i is the "eigenvalue" corresponding to the ith "eigenmode"
    
    # construct matrices
    M = gp.massmatrix_intrinsic(hel**2, F)
    Q = gp.biharmonic_energy_intrinsic(hel**2, F, bc="curved_hessian")
    
    L2_eigs, L2_modes = sp.linalg.eigsh(Q, k=k, M=M, sigma=0, which="LM")
    
    return L2_modes, L2_eigs
    
def at_hess1_segment(V, F, lam=0.5, alpha=1000.0, eps1=0.1, eps2=0.001, eps3=1e-5, n=30, hesstype="vfef", verbose=False):
    # perform ambrosio-tortorelli using the hess1 energy
    # returns the stylised vertex positions as (n, 3) array, and v from A-T as an (n, 1) array
    
    hel = gp.halfedge_lengths(V, F)
    Mv = gp.massmatrix(V, F)
    Mf = sp.diags(0.5*gp.doublearea(V, F))
    
    def face_lapl(V, F):
        tt, _ = gp.triangle_triangle_adjacency(F)
        ttm = tt.flatten()>=0
        
        cis = (V[F[:, 0], :] + V[F[:, 1], :] + V[F[:, 2], :])/3.0
        hel = gp.halfedge_lengths(V, F).flatten()
        duallens = np.linalg.norm(cis[tt.flatten(), :] - cis[np.repeat(np.arange(F.shape[0]), 3), :], axis=1)
        inv_cots = hel / duallens
        
        data = np.concatenate([-inv_cots.flatten(), inv_cots.flatten()])*np.tile(ttm, 2)
        row = np.concatenate([np.repeat(np.arange(F.shape[0]), 3), np.repeat(np.arange(F.shape[0]), 3)])
        col = np.concatenate([tt.flatten(), np.repeat(np.arange(F.shape[0]), 3)])*np.tile(ttm, 2)
        return sp.csr_matrix((data, (row, col)))
    
    if verbose: print("building H...")
    if hesstype == "vfef":
        H = vfef(hel, F)
    else:
        raise ValueError("hesstype " + hesstype + "not supported")
    if verbose: print("building L...")
    L = face_lapl(V, F)
    
    # initialise all variables with reasonable values
    u0x, u0y, u0z = V[:, 0], V[:, 1], V[:, 2]
    ux, uy, uz = u0x, u0y, u0z
    v = 0.5*np.ones((F.shape[0], 1))
    eps3 = eps3*np.sum(0.5*gp.doublearea(V, F))
    
    eps = eps1
    j = 1
    while eps > eps2:
        if verbose: print("eps: " + str(eps))
        
        for i in range(n):
            if verbose: print("  iteration " + str(i))
            
            all_H = np.linalg.norm(np.reshape(H @ ux[:, None], (-1, 4)), axis=1) + \
                    np.linalg.norm(np.reshape(H @ uy[:, None], (-1, 4)), axis=1) + \
                    np.linalg.norm(np.reshape(H @ uz[:, None], (-1, 4)), axis=1)
            
            # solve quad program for v
            if verbose: print("    building Q")
            Q = 2*eps*lam*L + (lam/(2*eps))*Mf + 2*sp.diags(all_H)
            
            if verbose: print("    solving H")
            v = gp.min_quad_with_fixed(Q=Q, c=(-2*(lam/(4*eps))*(np.ones((1, F.shape[0])) @ Mf)[0, :])[:, None])
            
            # solve l1 opt problem for u
            uxprev = ux
            uyprev = uy
            uzprev = uz
            
            if verbose: print("    l1 solve x")
            ux = hessian_l1_solve(hel, F, u0=u0x, mode=hesstype, alpha=alpha, g=v*v)
            if verbose: print("    l1 solve y")
            uy = hessian_l1_solve(hel, F, u0=u0y, mode=hesstype, alpha=alpha, g=v*v)
            if verbose: print("    l1 solve z")
            uz = hessian_l1_solve(hel, F, u0=u0z, mode=hesstype, alpha=alpha, g=v*v)
            
            # if the inner loop has converged, break into the next epsilon
            diff = np.linalg.norm(np.hstack([(ux-uxprev)[:, None], (uy-uyprev)[:, None], (uz-uzprev)[:, None]]), axis=1)
            diffnorm = diff[None, :] @ Mv @ diff[:, None]
            if verbose: print("    current difference: " + str(diffnorm))
            if diffnorm[0, 0] < eps3:
                # move to next epsilon if converged
                break
        
        # update epsilon
        eps = eps / 2.0
        j += 1
    
    return np.hstack([ux[:, None], uy[:, None], uz[:, None]]), v

def at_hess1_segment_scalar(V, F, u0, lam=0.5, alpha=1000.0, eps1=0.1, eps2=0.001, eps3=1e-5, n=30, hesstype="vfef", verbose=False):
    # perform ambrosio-tortorelli using our hess1 energy
    # returns the stylised vertex positions as (n, 3) array, and v from A-T as an (n, 1) array
    
    hel = gp.halfedge_lengths(V, F)
    Mv = gp.massmatrix(V, F)
    Mf = sp.diags(0.5*gp.doublearea(V, F))
    
    def face_lapl(V, F):
        tt, _ = gp.triangle_triangle_adjacency(F)
        ttm = tt.flatten()>=0
        
        cis = (V[F[:, 0], :] + V[F[:, 1], :] + V[F[:, 2], :])/3.0
        hel = gp.halfedge_lengths(V, F).flatten()
        duallens = np.linalg.norm(cis[tt.flatten(), :] - cis[np.repeat(np.arange(F.shape[0]), 3), :], axis=1)
        inv_cots = hel / duallens
        
        data = np.concatenate([-inv_cots.flatten(), inv_cots.flatten()])*np.tile(ttm, 2)
        row = np.concatenate([np.repeat(np.arange(F.shape[0]), 3), np.repeat(np.arange(F.shape[0]), 3)])
        col = np.concatenate([tt.flatten(), np.repeat(np.arange(F.shape[0]), 3)])*np.tile(ttm, 2)
        return sp.csr_matrix((data, (row, col)))
    
    if verbose: print("building H...")
    if hesstype == "vfef":
        H = vfef(hel, F)
    else:
        raise ValueError("hesstype " + hesstype + "not supported")
    if verbose: print("building L...")
    L = face_lapl(V, F)
    
    # initialise all variables with reasonable values
    u = u0
    v = 0.5*np.ones((F.shape[0], 1))
    eps3 = eps3*np.sum(0.5*gp.doublearea(V, F))
    
    eps = eps1
    j = 1
    while eps > eps2:
        if verbose: print("eps: " + str(eps))
        
        for i in range(n):
            if verbose: print("  iteration " + str(i))
            
            all_H = np.linalg.norm(np.reshape(H @ u[:, None], (-1, 4)), axis=1)
            
            # solve quad program for v
            if verbose: print("    building Q")
            Q = 2*eps*lam*L + (lam/(2*eps))*Mf + 2*sp.diags(all_H)
            
            if verbose: print("    solving H")
            v = gp.min_quad_with_fixed(Q=Q, c=(-2*(lam/(4*eps))*(np.ones((1, F.shape[0])) @ Mf)[0, :])[:, None])
            
            # solve l1 opt problem for u
            uprev = u
            
            if verbose: print("    l1 solve")
            u = hessian_l1_solve(hel, F, u0=u0, mode=hesstype, alpha=alpha, g=v*v)
            
            # if the inner loop has converged, break into the next epsilon
            diff = np.abs(u-uprev)
            diffnorm = diff[None, :] @ Mv @ diff[:, None]
            if verbose: print("    current difference: " + str(diffnorm))
            if diffnorm[0, 0] < eps3:
                # move to next epsilon if converged
                break
        
        # update epsilon
        eps = eps / 2.0
        j += 1
    
    return u, v

def at_hess1_segment_color(V, F, u0, lam=0.5, alpha=1000.0, eps1=0.1, eps2=0.001, eps3=1e-5, n=30, hesstype="vfef", verbose=False):
    # perform ambrosio-tortorelli using our hess1 energy
    # returns the stylised colors as (n, 3) array, and v from A-T as an (n, 1) array
    
    hel = gp.halfedge_lengths(V, F)
    Mv = gp.massmatrix(V, F)
    Mf = sp.diags(0.5*gp.doublearea(V, F))
    
    def face_lapl(V, F):
        tt, _ = gp.triangle_triangle_adjacency(F)
        ttm = tt.flatten()>=0
        
        cis = (V[F[:, 0], :] + V[F[:, 1], :] + V[F[:, 2], :])/3.0
        hel = gp.halfedge_lengths(V, F).flatten()
        duallens = np.linalg.norm(cis[tt.flatten(), :] - cis[np.repeat(np.arange(F.shape[0]), 3), :], axis=1)
        inv_cots = hel / duallens
        
        data = np.concatenate([-inv_cots.flatten(), inv_cots.flatten()])*np.tile(ttm, 2)
        row = np.concatenate([np.repeat(np.arange(F.shape[0]), 3), np.repeat(np.arange(F.shape[0]), 3)])
        col = np.concatenate([tt.flatten(), np.repeat(np.arange(F.shape[0]), 3)])*np.tile(ttm, 2)
        return sp.csr_matrix((data, (row, col)))
    
    if verbose: print("building H...")
    if hesstype == "vfef":
        H = vfef(hel, F)
    else:
        raise ValueError("hesstype " + hesstype + "not supported")
    if verbose: print("building L...")
    L = face_lapl(V, F)
    
    # initialise all variables with reasonable values
    u0r, u0g, u0b = u0[:, 0], u0[:, 1], u0[:, 2]
    ur, ug, ub = u0r, u0g, u0b
    v = 0.5*np.ones((F.shape[0], 1))
    eps3 = eps3*np.sum(0.5*gp.doublearea(V, F))
    
    eps = eps1
    j = 1
    while eps > eps2:
        if verbose: print("eps: " + str(eps))
        
        for i in range(n):
            if verbose: print("  iteration " + str(i))
            
            all_H = np.linalg.norm(np.reshape(H @ ur[:, None], (-1, 4)), axis=1) + \
                    np.linalg.norm(np.reshape(H @ ug[:, None], (-1, 4)), axis=1) + \
                    np.linalg.norm(np.reshape(H @ ub[:, None], (-1, 4)), axis=1)
            
            # solve quad program for v
            if verbose: print("    building Q")
            Q = 2*eps*lam*L + (lam/(2*eps))*Mf + 2*sp.diags(all_H)
            
            if verbose: print("    solving H")
            v = gp.min_quad_with_fixed(Q=Q, c=(-2*(lam/(4*eps))*(np.ones((1, F.shape[0])) @ Mf)[0, :])[:, None])
            
            # solve l1 opt problem for u
            urprev = ur
            ugprev = ug
            ubprev = ub
            
            if verbose: print("    l1 solve x")
            ur = hessian_l1_solve(hel, F, u0=u0r, mode=hesstype, alpha=alpha, g=v*v)
            if verbose: print("    l1 solve y")
            ug = hessian_l1_solve(hel, F, u0=u0g, mode=hesstype, alpha=alpha, g=v*v)
            if verbose: print("    l1 solve z")
            ub = hessian_l1_solve(hel, F, u0=u0b, mode=hesstype, alpha=alpha, g=v*v)
            
            # if the inner loop has converged, break into the next epsilon
            diff = np.linalg.norm(np.hstack([(ur-urprev)[:, None], (ug-ugprev)[:, None], (ub-ubprev)[:, None]]), axis=1)
            diffnorm = diff[None, :] @ Mv @ diff[:, None]
            if verbose: print("    current difference: " + str(diffnorm))
            if diffnorm[0, 0] < eps3:
                # move to next epsilon if converged
                break
        
        # update epsilon
        eps = eps / 2.0
        j += 1
    
    return np.hstack([ur[:, None], ug[:, None], ub[:, None]]), v

def thresh_segmentation_from_v(V, F, v, cut_thresh=0.8, merge_limit=4):
    # given v on faces, clean up the segmentation and cut the mesh into pieces
    
    # spread to mean
    tt, tti = gp.triangle_triangle_adjacency(F)
    he_v = np.repeat(v, 3, axis=1)
    he_v = (he_v + he_v[tt, tti])/2.0
    # threshold
    halfedges_to_cut = he_v < cut_thresh
    
    # ~~~ INITIAL CUT BASED ON THRESHOLD ~~~
    
    # mesh with cut edges
    edges_to_cut = gp.halfedge_edge_map(F)[2][halfedges_to_cut]
    F1, I = gp.cut_edges(F, gp.edges(F)[edges_to_cut, :])
    
    # separate into components
    c, cf = gp.connected_components(F1, return_face_indices=True)
    topcomp = np.amax(c)
    
    cf2 = cf.copy()
    
    # treat cf as correct on the original mesh.  then for each component that has < 3, change it by voting
    merged_count = 0
    nonzero_cts = np.array([np.count_nonzero(cf2==i) for i in range(topcomp)])
    while np.any(np.logical_and(nonzero_cts <= merge_limit, nonzero_cts > 0)):
        print(np.count_nonzero(np.logical_and(nonzero_cts <= merge_limit, nonzero_cts > 0)))
        for i in range(topcomp+1):
            comp_size = np.count_nonzero(cf2==i)
            if comp_size <= merge_limit and comp_size != 0:
                merged_count += 1
                nzs = np.nonzero(cf2==i)
                # mode of the boundary
                adj_comps_mode = mode(cf2[np.setdiff1d(tt[nzs, :].flatten(), 
                                                       np.append(nzs, -1))])[0]
                
                cf2[np.nonzero(cf2==i)] = adj_comps_mode # set equal to mode of the adjacency
        nonzero_cts = np.array([np.count_nonzero(cf2==i) for i in range(topcomp)])
    
    print("merged " + str(merged_count) + " total original components to larger components")
    
    # ~~~ SECOND CUT BASED ON VOTING ~~~
    
    # recut based on cf2
    halfedges_to_cut = cf2[np.reshape(np.repeat(np.arange(F.shape[0]), 3), (-1, 3))] != cf2[np.where(tt!=-1, tt, np.reshape(np.repeat(np.arange(F.shape[0]), 3), (-1, 3)))]
    
    # mesh with cut edges
    edges_to_cut = gp.halfedge_edge_map(F)[2][halfedges_to_cut]
    F2, I = gp.cut_edges(F, gp.edges(F)[edges_to_cut, :])
    V2 = V[I]
    
    # separate into components
    c3, cf3 = gp.connected_components(F2, return_face_indices=True)
    topcomp = np.amax(c3)
    
    print("ended up with " + str(topcomp+1) + " components!")
    
    Vis, Fis = [], []
    for i in range(topcomp+1):
        Fi = F2[cf3==i, :]
        Vi, Fi = gp.remove_unreferenced(V2, Fi)
        
        Vis += [Vi]
        Fis += [Fi]
        
    return Vis, Fis, cf3

def thresh_segmentation_from_v_diffusion(V, F, v, uv, ft, txsamp, vertcolors, cut_thresh=0.8, merge_limit=4):
    # given v on faces, clean up the segmentation and cut the mesh into pieces
    # return V, F for each of these pieces
    
    # spread to mean
    tt, tti = gp.triangle_triangle_adjacency(F)
    he_v = np.repeat(v, 3, axis=1)
    he_v = (he_v + he_v[tt, tti])/2.0
    # threshold
    halfedges_to_cut = he_v < cut_thresh
    
    # ~~~ INITIAL CUT BASED ON THRESHOLD ~~~
    
    # mesh with cut edges
    edges_to_cut = gp.halfedge_edge_map(F)[2][halfedges_to_cut]
    F1, I = gp.cut_edges(F, gp.edges(F)[edges_to_cut, :])
    
    # separate into components
    c, cf = gp.connected_components(F1, return_face_indices=True)
    topcomp = np.amax(c)
    
    cf2 = cf.copy()
    
    # treat cf as correct on the original mesh.  then for each component that has <= merge_limit, change it by voting
    merged_count = 0
    nonzero_cts = np.array([np.count_nonzero(cf2==i) for i in range(topcomp)])
    while np.any(np.logical_and(nonzero_cts <= merge_limit, nonzero_cts > 0)):
        print(np.count_nonzero(np.logical_and(nonzero_cts <= merge_limit, nonzero_cts > 0)))
        for i in range(topcomp+1):
            comp_size = np.count_nonzero(cf2==i)
            if comp_size <= merge_limit and comp_size != 0:
                merged_count += 1
                nzs = np.nonzero(cf2==i)
                # mode of the boundary
                adj_comps_mode = mode(cf2[np.setdiff1d(tt[nzs, :].flatten(), 
                                                       nzs)])[0]
                
                cf2[np.nonzero(cf2==i)] = adj_comps_mode # set equal to mode of the adjacency
        nonzero_cts = np.array([np.count_nonzero(cf2==i) for i in range(topcomp)])
    
    print("merged " + str(merged_count) + " original components to larger components")
    
    # ~~~ SECOND CUT BASED ON VOTING FOR SMALL COMPONENTS ~~~
    
    # recut based on cf2
    halfedges_to_cut = cf2[np.reshape(np.repeat(np.arange(F.shape[0]), 3), (-1, 3))] != cf2[np.where(tt!=-1, tt, np.reshape(np.repeat(np.arange(F.shape[0]), 3), (-1, 3)))]
    
    # mesh with cut edges
    edges_to_cut = gp.halfedge_edge_map(F)[2][halfedges_to_cut]
    F2, I = gp.cut_edges(F, gp.edges(F)[edges_to_cut, :])
    V2 = V[I, :]
    
    # separate into components
    c3, cf3 = gp.connected_components(F2, return_face_indices=True)
    topcomp = np.amax(c3)
    
    print("ended up with " + str(topcomp+1) + " components!")
    
    Vis, Fis, sis, svs = [], [], [], []
    for i in range(topcomp+1):
        Fi = F2[cf3==i, :]
        Vi, Fi, Iru, Jru = gp.remove_unreferenced(V2, Fi, return_maps=True)
        
        # for each face in Fi, get the corresponding face in F2
        si = gp.boundary_vertices(Fi)
        V2si_idx = Jru[si] # original vertices in V2
        Vsi_idx = I[V2si_idx] # original vertices in V
        
        # look at all containing faces until we find one with cf3 == i.  set the color equal to whatever's at that corner
        sv = np.zeros((si.shape[0], 3))
        for j in range(si.shape[0]):
            F_cont = np.logical_and(cf3 == i, np.any(F == Vsi_idx[j], axis=1)) # faces containing our vertex with cf3 == i
            # choose a single face
            first_F_cont = np.nonzero(F_cont)[0][0]
            idx_of_Vsi_idx = np.nonzero(F[first_F_cont, :] == Vsi_idx[j])[0]
            sv[j, :] = txsamp(uv[ft[first_F_cont, idx_of_Vsi_idx], :])[0, :]
        
        # solve an inverse problem to get the colors on the boundary
        # get the original colors
        colors_internal = vertcolors[I[Jru[np.arange(Vi.shape[0])]], :]
        colors_internal[si] = sv
        
        # solve inverse problem to get best harmonic function
        Mi = gp.massmatrix(Vi, Fi)
        Li = gp.cotangent_laplacian(Vi, Fi)
        
        # approximate
        alpha2 = 1.0
        harmonic_colors = gp.min_quad_with_fixed(Q=2*(alpha2*Mi + Li), c=-2*alpha2*Mi@(colors_internal))
        sv = harmonic_colors[si, :]
        
        Vis += [Vi]
        Fis += [Fi]
        sis += [si]
        svs += [sv]
        
    return Vis, Fis, sis, svs
