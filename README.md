
<h1>sssh</h1>

Code for the paper "Sharpening and Sparsifying with Surface Hessians", accepted at SIGGRAPH Asia 2024.

<h2>Installation</h2>
Written and tested using Python version 3.10.12.
Scripts are assumed to be run from the scripts folder.

The following should work, in the top directory.  The user must install SuiteSparse on their system, and have MOSEK installed and licensed.
```
conda env create -f environment.yaml
conda activate sssh
```

and to test the installation, do

```
cd scripts
python3 sphere_rand_interp.py
```

A Polyscope window should appear that allows you to play with a sphere sparse interpolation example.

To rerender all figures from the paper, go to the scripts directory and type:

```
python3 gen_all_figs.py 1 0
```

The first argument determines whether (1) or not (0) to generate the geometry again (instead of just rendering from previous geometry).
The second argument determines whether to render at low (1) or high (0) quality.

To just render one figure, do

```
python3 genfig_....py 1 0
```

where the arguments have the same meaning.

<h2>Relevant Functions</h2>

The most important functions are in the folder /src/l1hessian/:

- **L1 Hessian solver:** ```hessian_l1_solve(hel, F, u0=None, mode="vfef", center="centroid", verbose=False, tol=None, alpha=0, beta=0, y=None, k=None, A=None, b=None, Q=None, g=None, V=None, M=None)```
  
    Solves (using MOSEK) a problem of the form
  
  $`\text{argmin}_u \int_\Omega g \cdot \|\text{Hess}\, u\|_F \,d\sigma + \alpha \int_\Omega\|u-u_0\|_2^2 \,d\sigma + \frac{\beta}{2}u^\intercal Q u`$
  
  $`\,\,\,\,\,\text{subj. to}\,\,\, u(y_i) = k_i `$

  $`\,\,\,\,\,Au \leq b`$

  The inputs ```hel``` and ```F``` are the (unsquared) halfedge lengths and face list of the input mesh, respectively (using the same ordering conventions as gpytoolbox).

  The Hessian used is determined by ```mode```; the default is ```'vfef'```, and the possible options are:
  - ```'vfef'``` for our intrinsic discretization of the L1 Hessian
  - ```'stein et al 2018'``` for an "axis-aligned" extrinsic version of the L1 Hessian as defined in "Natural Boundary Conditions for Smoothing in Geometry Processing" (Stein et al. 2018).  This is the one detailed in their Appendix C.

  If using ```'stein et al 2018'```, also provide the vertex positions of the mesh in the argument ```V=...```, since that Hessian is extrinsic.

  There is also an option to provide a mass matrix (```M=...```) which differs from the one defined by ```hel``` and ```F```.  This mass matrix is only used for the fidelity term ($`\int_\Omega \|u-u_0\|_2^2`$).

  ```u```, ```u0```, ```g```, ```b``` are treated as column (d, 1) arrays; ```y```, ```k``` are treated as flat (d,) arrays.

<br />

- **L1-Hessian-Ambrosio-Tortorelli Solver:** ```at_hess1_segment(V, F, lam=0.5, alpha=1000.0, eps1=0.1, eps2=0.001, eps3=1e-5, n=30, hesstype="vfef", verbose=False)```
  
    Solves an Ambrosio-Tortorelli like functional defined as
  
  $`\text{argmin}_u \sum_{i\in{x, y, z}}\int_\Omega v^2 \|\text{Hess}\, u_i\|_F \,d\sigma + \alpha \sum_{i\in{x, y, z}}\int_\Omega\|u_i-u_{0i}\|_2^2 \,d\sigma + \lambda \int_\Omega\epsilon\|\nabla v\|^2 +\frac{1}{4\epsilon}(1-v)^2 \,d\sigma`$

  ...where the epsilon decreases until it is smaller than eps2, starting from eps1, and the $u$ variables represent the coordinates of the input and output vertex positions.

  The output is of the form ```(U_final, v)```, where ```U_final``` is an (n, 3) array containing the final (stylised) positions of the vertices, and ```v``` is an ```(m, 1)``` array containing the final facewise values of the indicator-like variable in the Ambrosio-Tortorelli functional.

  The input variable ```eps3``` gives a convergence criterion tolerance for the inner loop of the optimisation; it is rescaled to match the surface area of the mesh internally (since the convergence criterion is based on an integration over the mesh).  The inner loop will terminate after ```n``` iterations if it does not reach ```eps3``` first.

  Informally, ```lambda``` is a penalisation on the length of discontinuity curves in the solution, and ```alpha``` determines how similar the final solution is to the input mesh.
  
<br />

- **L1-Hessian Eigenmodes Solver:** ```hessian_l1_modes(hel, F, k=5, mu=100, eps=1e-10, hesstype="vfef", randinit=False, seed=-1, verbose=False)```

    Solves the iterative compressed eigenmode problem corresponding to the L1 Hessian energy, for each $i\in[k]$:

  $`\text{argmin}_u \int_\Omega \|\nabla u_i\|^2 + \mu \|\text{Hess}\, u_i\|_F \,d\sigma `$
  
  $`\,\,\,\,\,\text{subj. to}\,\,\, \int_\Omega u_i \, d\sigma = 1`$

  $`\,\,\,\,\,\,\,\,\, \langle u_i, u_j \rangle = 0 \,\, \forall j < i`$


    High $\mu$ (e.g. ```100```) gives modes more representative of our energy; low $\mu$ (e.g. ```1e-6```) gives modes more representative of the standard eigenmodes of our mesh.
<br />

- **L1 Hessian Matrix:** ```vfef(hel, F, center="centroid")```
  
    Takes in ```hel``` as would be output by ```gp.halfedge_lengths(V, F)```, and face list ```F```. 
    Returns the 4|F| x |V| matrix representing the integrated Hessian operator.
    The result of multiplying this matrix by a vector is "to-be-normed" across every 4 entries; that is, to get the integrated L1 Hessian per-face for an (n, 1) vector ```u```, one would do something like
    ```np.linalg.norm(np.reshape(vfef(hel, F) @ u, (-1, 4)), axis=1, keepdims=True)```.

<br />

<h2> Known issues </h2>

The figure generation scripts are known to have some minor issues:
- On Linux machines, the conda installation sometimes fails to generate figures using ```genfig_stravinsky_segdiffuse.py```.
- On Apple machines, the files requiring gpytoolbox's ```curved_hessian``` or ```remesh_botsch``` sometimes fail to find the relevant C++ bindings, and cannot generate the relevant figures.
- On some machines, the colors for segmentation results may differ slightly, and small details in some segmentation results may differ (the segmentation procedure does not employ randomness, so this is likely due to architectural differences).
Additionally, the file ```genfig_evora_holefill.py``` will not function on any architecture, as the evora mesh is too large to upload to Github.

<h2>Issues / Questions</h2>

Email ```dylanr@usc.edu``` if there are issues or if you have other questions.

