import numpy as np
import scipy.sparse as sp
import sys, mosek
from sksparse.cholmod import cholesky

def streamprinter(text):
    sys.stdout.write(text)
    sys.stdout.flush()

def min_quadfauxl1_d_with_lin(Q, G, c=None, A=None, b=None, g=None, verbose=False, d=2):
    # construct a MOSEK conic program which represents the optimisation problem:
    
    # min 0.5 u^T Q u + c^T u + g.T @ |G u|_1
    # s.t. A u <= b
    
    # this time, G takes u to d*m; the entries are like [...,x0i, x1i, x2i,...].T
    # the l1 is instead a sum over the norms of each subvector
    
    # the cone program becomes
    # min r + c^T u + g^T z
    # s.t. [A 0 0] [u | z | r]^T <= b
    #      (z_i, Gu_d*i+0, Gu_{d*i+1}, ...) \in Q^d for all i, where Q^d is the quadratic cone x0 >= sqrt(x1^2 + x2^2 ...)
    #      (1, r, L.T P u) \in Qr for all i, where Qr is the rotated k+2 quadratic cone, and L/P are from cholesky
    
    # Q is n x n
    # c is n x 1
    # G is d*m x n
    # A is l x n
    # b is l x 1
    
    # do sparse CHOLESKY to Q.  get the Lt which is ACTUALLY a factor of Q
    ch = cholesky(Q.tocsc())
    Lt = ch.L().T[:, np.argsort(ch.P())]
    
    k = Lt.shape[0]
    n = Q.shape[0]
    m = G.shape[0]//d
    if A is None and b is None:
        A = np.zeros((1, n))
        b = np.zeros((1, 1))
    if c is None:
        c = np.zeros((n, 1))
    if g is None:
        g = np.ones((m, 1))
    l = A.shape[0]
    
    # assert shapes
    assert Q.shape == (n, n)
    assert c.shape == (n, 1)
    assert G.shape == (d*m, n)
    assert A.shape == (l, n)
    assert b.shape == (l, 1)
    assert g.shape == (m, 1)
    assert Lt.shape == (k, n)
    
    # put all into csr format
    G = sp.csr_array(G)
    A = sp.csr_array(A)
    Lt = sp.csr_array(Lt)
    # various 0 and 1 matrices
    Olm1 = sp.csr_matrix((l, m+1))
    Im = sp.identity(m)
    
    moA = sp.bmat([[A, Olm1]])
    moF = sp.bmat([[G, None, None],
                   [None, Im, None],
                   [Lt, None, None],
                   [None, None, sp.coo_array([[1]])],
                   [None, None, sp.coo_array([[0]])]])
    
    moAi, moAj, moAv = sp.find(moA)
    moFi, moFj, moFv = sp.find(moF)
    
    # actually set the mosek task
    with mosek.Task() as task:
        # print to the stream printer if verbose
        if verbose:
            task.set_Stream(mosek.streamtype.log, streamprinter)
        
        task.appendvars(n + m + 1)
        task.appendcons(l)
        
        for j in range(n):
            task.putcj(j, c[j, 0])
        for j in range(n, n+m):
            task.putcj(j, g[j-n, 0])
        for j in range(n+m, n+m+1):
            task.putcj(j, 1)
            
        # n+m free variables; r is lower bounded by 0 by PSD-ness of Q
        for j in range(n + m):
            task.putvarbound(j, mosek.boundkey.fr, 0, 0)
        task.putvarbound(n+m, mosek.boundkey.lo, 0, 0) # r is bounded below
        
        # linear constraints
        task.putaijlist(moAi, moAj, moAv)
        buc = list(b[:, 0])
        for j in range(l):
            task.putconbound(j, mosek.boundkey.up, 0, buc[j])
        
        # afe storage filling
        task.appendafes((d+1)*m + k + 2)
        task.putafefentrylist(moFi, moFj, moFv)
        task.putafeg((d+1)*m + k + 1, 1) # a single 1 in the final entry, so that 1 can be in the rotated quadcone
        
        # append all affine conic constraints (quadratic cones)
        for i in range(m):
            quadcone = task.appendquadraticconedomain(d+1)
            task.appendacc(quadcone,
                           [d*m+i] + [d*i+j for j in range(d)], # rows from F: zi, Gu_2*i, Gu_2*i+1
                           None) # unused
        
        # rotated cone constraint for quadratic optimisation
        rquadcone = task.appendrquadraticconedomain(k+2)
        task.appendacc(rquadcone,
                       [(d+1)*m+k+1, (d+1)*m+k] + list(range((d+1)*m, (d+1)*m+k)),
                       None) # unused
        
        task.putobjsense(mosek.objsense.minimize)
        
        task.optimize()
        
        if verbose:
            task.solutionsummary(mosek.streamtype.msg)
        
        solsta = task.getsolsta(mosek.soltype.itr)
        
        if solsta == mosek.solsta.optimal:
            if verbose:
                print("Nice")
            return np.array([task.getxx(mosek.soltype.itr)]).T[:n, :] # only send back the first n.  the rest are dummies
        elif (solsta == mosek.solsta.dual_infeas_cer or
              solsta == mosek.solsta.prim_infeas_cer):
            print("Primal or dual infeasibility certificate found.\n")
            raise ValueError("Primal or dual infeasibility certificate found.\n")
        elif solsta == mosek.solsta.unknown:
            raise ValueError("Unknown solution status")
        else:
            raise ValueError("Other solution status")

def min_fauxl1_d_with_lin(G, c=None, A=None, b=None, g=None, verbose=False, d=2):
    # construct a MOSEK conic program which represents the optimisation problem:
    # (WITHOUT Q)
    
    # min c^T u + g.T @ |G u|_1
    # s.t. A u <= b
    
    # this time, G takes u to d*m; the entries are like [...,x0i, x1i x2i,...].T
    # the l1 is instead a sum over the |xji|_2 of the whole array or something
    
    # the cone program becomes
    # min c^T u + g^T z
    # s.t. [A 0] [u | z]^T <= b
    #      (z_i, Gu_{d*i+0}, Gu_{d*i+1}, ...) \in Q for all i, where Q is the quadratic cone x0 >= sqrt(x1^2 + x2^2 ...)
    
    # c is n x 1
    # G is d*m x n
    # A is l x n
    # b is l x 1
    
    n = G.shape[1]
    m = G.shape[0]//d
    if A is None and b is None:
        A = np.zeros((1, n))
        b = np.zeros((1, 1))
    if c is None:
        c = np.zeros((n, 1))
    if g is None:
        g = np.ones((m, 1))
    l = A.shape[0]
    
    # assert shapes
    assert c.shape == (n, 1)
    assert G.shape == (d*m, n)
    assert A.shape == (l, n)
    assert b.shape == (l, 1)
    assert g.shape == (m, 1)
    
    # put all into csr format
    G = sp.csr_array(G)
    A = sp.csr_array(A)
    # various 0 and 1 matrices
    Olm1 = sp.csr_matrix((l, m))
    Im = sp.identity(m)
    
    moA = sp.bmat([[A, Olm1]])
    moF = sp.bmat([[G, None],
                   [None, Im]])
    
    moAi, moAj, moAv = sp.find(moA)
    moFi, moFj, moFv = sp.find(moF)
    
    # actually set the mosek task
    with mosek.Task() as task:
        # print to the stream printer if verbose
        if verbose:
            task.set_Stream(mosek.streamtype.log, streamprinter)
        
        task.appendvars(n + m)
        task.appendcons(l)
        
        for j in range(n):
            task.putcj(j, c[j, 0])
        for j in range(n, n+m):
            task.putcj(j, g[j-n, 0])
            
        # n+m free variables
        for j in range(n + m):
            task.putvarbound(j, mosek.boundkey.fr, 0, 0)
        
        # linear constraints
        task.putaijlist(moAi, moAj, moAv)
        buc = list(b[:, 0])
        for j in range(l):
            task.putconbound(j, mosek.boundkey.up, 0, buc[j])
        
        # afe storage filling
        task.appendafes((d+1)*m)
        task.putafefentrylist(moFi, moFj, moFv)
        
        # append all affine conic constraints (quadratic cones)
        for i in range(m):
            quadcone = task.appendquadraticconedomain(d+1)
            task.appendacc(quadcone,
                           [d*m+i] + [d*i+j for j in range(d)], # rows from F: zi, Gu_2*i, Gu_2*i+1
                           None) # unused
        
        task.putobjsense(mosek.objsense.minimize)
        
        task.optimize()
        
        if verbose:
            task.solutionsummary(mosek.streamtype.msg)
        
        solsta = task.getsolsta(mosek.soltype.itr)
        
        if solsta == mosek.solsta.optimal:
            if verbose:
                print("Nice")
            return np.array([task.getxx(mosek.soltype.itr)]).T[:n, :] # only send back the first n.  the rest are dummies
        elif (solsta == mosek.solsta.dual_infeas_cer or
              solsta == mosek.solsta.prim_infeas_cer):
            raise ValueError("Primal or dual infeasibility certificate found.\n")
        elif solsta == mosek.solsta.unknown:
            raise ValueError("Unknown solution status")
        else:
            raise ValueError("Other solution status")

def min_quadl1_with_lin(Q, G, c=None, A=None, b=None, d=None, verbose=False):
    # construct a MOSEK conic program which represents the optimisation problem:
    
    # min 0.5 u^T Q u + c^T u + d^T|G u|
    # s.t. A u <= b
    
    # by a reduction to the mosek-friendly problem
    
    # min 0.5 * u^T Q u + c^T u + d^T y
    # s.t. A u <= b
    #     -G u - y <= 0
    #      G u - y <= 0
    #       y >= 0
    
    # Q is n x n
    # c is n x 1
    # G is m x n
    # A is l x n
    # b is l x 1
    
    n = Q.shape[0]
    m = G.shape[0]
    if A is None and b is None:
        A = np.zeros((1, n))
        b = np.zeros((1, 1))
    if c is None:
        c = np.zeros((n, 1))
    if d is None:
        d = np.ones((m, 1))
    l = A.shape[0]
    
    # assert shapes
    assert Q.shape == (n, n)
    assert c.shape == (n, 1)
    assert G.shape == (m, n)
    assert A.shape == (l, n)
    assert b.shape == (l, 1)
    
    # put all into csr format
    Q = sp.csr_array(Q)
    G = sp.csr_array(G)
    A = sp.csr_array(A)
    Im = sp.identity(m)
    Om = sp.csr_matrix((l, m))
    
    moA = sp.bmat([[ A,  Om],
                   [-G, -Im],
                   [ G, -Im]])
    
    Qi, Qj, Qv = sp.find(Q)
    moAi, moAj, moAv = sp.find(moA)
    
    # mosek only takes lower triangular matrices for objective Q.  so only use the Q indices where col >= row
    li = np.nonzero(Qi >= Qj)
    Qi, Qj, Qv = Qi[li], Qj[li], Qv[li]
    
    # actually set the mosek task
    with mosek.Task() as task:
        # print to the stream printer if verbose
        if verbose:
            task.set_Stream(mosek.streamtype.log, streamprinter)
        
        task.appendvars(n + m)
        task.appendcons(l + 2*m)
        
        # insert Q to the task
        task.putqobj(Qi, Qj, Qv)
        
        for j in range(n):
            task.putcj(j, c[j, 0])
        for j in range(n, n+m):
            task.putcj(j, d[j-n, 0])
            
        # variable constraints
        bkx =  n*[mosek.boundkey.fr] + m*[mosek.boundkey.lo]
        for j in range(n+m):
            task.putvarbound(j, bkx[j], 0, 0)
        
        # linear constraints
        task.putaijlist(moAi, moAj, moAv)
        buc = list(b[:, 0]) + 2*m*[0]
        for j in range(l + 2*m):
            task.putconbound(j, mosek.boundkey.up, 0, buc[j])
        
        task.putobjsense(mosek.objsense.minimize)
        
        task.optimize()
        
        if verbose:
            task.solutionsummary(mosek.streamtype.msg)
        
        solsta = task.getsolsta(mosek.soltype.itr)
        
        if solsta == mosek.solsta.optimal:
            if verbose:
                print("Nice")
            return np.array([task.getxx(mosek.soltype.itr)]).T[:n, :]
        elif (solsta == mosek.solsta.dual_infeas_cer or
              solsta == mosek.solsta.prim_infeas_cer):
            raise ValueError("Primal or dual infeasibility certificate found.\n")
        elif solsta == mosek.solsta.unknown:
            raise ValueError("Unknown solution status")
        else:
            raise ValueError("Other solution status")

def min_l1_with_lin(G, c=None, A=None, b=None, d=None, si=None, sv=None, verbose=False):
    # construct a MOSEK conic program which represents the optimisation problem:
    
    # min c^T u + d^T|G u|
    # s.t. A u <= b
    
    # by a reduction to the mosek-friendly problem
    
    # min c^T u + d^T y
    # s.t. A u <= b
    #     -G u - y <= 0
    #      G u - y <= 0
    #       y >= 0
    
    # Q is n x n
    # c is n x 1
    # G is m x n
    # A is l x n
    # b is l x 1
    
    n = G.shape[1]
    m = G.shape[0]
    if A is None and b is None:
        A = np.zeros((1, n))
        b = np.zeros((1, 1))
    if c is None:
        c = np.zeros((n, 1))
    if d is None:
        d = np.ones((m, 1))
    l = A.shape[0]
    
    # assert shapes
    # assert Q.shape == (n, n)
    assert c.shape == (n, 1)
    assert G.shape == (m, n)
    assert A.shape == (l, n)
    assert b.shape == (l, 1)
    
    # put all into csr format
    # Q = sp.csr_array(Q)
    G = sp.csr_array(G)
    A = sp.csr_array(A)
    Im = sp.identity(m)
    Om = sp.csr_array((l, m))
    
    moA = sp.bmat([[ A,  Om],
                   [-G, -Im],
                   [ G, -Im]])
    
    moAi, moAj, moAv = sp.find(moA)
    
    # actually set the mosek task
    with mosek.Task() as task:
        # print to the stream printer if verbose
        if verbose:
            task.set_Stream(mosek.streamtype.log, streamprinter)
        
        task.appendvars(n + m)
        task.appendcons(l + 2*m)
        
        for j in range(n):
            task.putcj(j, c[j, 0])
        for j in range(n, n+m):
            task.putcj(j, d[j-n, 0])
            
        # variable constraints
        bkx =  n*[mosek.boundkey.fr] + m*[mosek.boundkey.lo]
        if not (si is None):
            for j in range(len(si)):
                bkx[si[j]] = mosek.boundkey.fx
        for j in range(n+m):
            task.putvarbound(j, bkx[j], 0, 0)
        if not (si is None):
            for j in range(len(si)):
                task.putvarbound(si[j], bkx[si[j]], sv[j], sv[j])
        
        # linear constraints
        task.putaijlist(moAi, moAj, moAv)
        buc = list(b[:, 0]) + 2*m*[0]
        for j in range(l + 2*m):
            task.putconbound(j, mosek.boundkey.up, 0, buc[j])
        
        task.putobjsense(mosek.objsense.minimize)
        
        task.optimize()
        
        if verbose:
            task.solutionsummary(mosek.streamtype.msg)
        
        solsta = task.getsolsta(mosek.soltype.itr)
        
        if solsta == mosek.solsta.optimal:
            if verbose:
                print("Nice")
            return np.array([task.getxx(mosek.soltype.itr)]).T[:n, :]
        elif (solsta == mosek.solsta.dual_infeas_cer or
              solsta == mosek.solsta.prim_infeas_cer):
            raise ValueError("Primal or dual infeasibility certificate found.\n")
        elif solsta == mosek.solsta.unknown:
            raise ValueError("Unknown solution status")
        else:
            raise ValueError("Other solution status")
