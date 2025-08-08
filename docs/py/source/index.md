# Signed Heat Method (Python)

`signed_heat_method` is a Python library implementing the [Signed Heat Method](https://nzfeng.github.io/research/SignedHeatMethod/index.html) for computing **robust signed distance fields (SDFs)** to triangle meshes, polygon meshes, and point clouds in 3D. 

![teaser image](../../../media/teaser.png)

This library consists of Python bindings to [C++ code](https://github.com/nzfeng/signed-heat-3d). Toggle the "C++/Python" switch at the top of this page to see documentation for the C++ version.

Install `signed_heat_method` via PyPI:
```
pip install signed_heat_method
```

**Sample:**

```python
import signed_heat_method as shm

V, F = # your mesh, specified as a vertex and face array

# Initalize tet mesh solver
tet_solver = shm.SignedHeatTetSolver(verbose=True) # print all messages

# Solve!
sdf_tets = tet_solver.compute_distance_to_mesh(V, F)

# Solve on a regular grid instead.
grid_solver = shm.SignedHeatGridSolver(verbose=True)
sdf_grid = grid_solver.compute_distance_to_mesh(V, F)
```

For a demo, see the [sample project](https://github.com/nzfeng/signed-heat-python/blob/main/test/demo.py), which uses [Polyscope](https://polyscope.run) for visualization.

More info about the method, including a blog-style summary, an introductory 10-minute talk, and the corresponding academic paper are all located at the project page [here](https://www.youtube.com/watch?v=mw5Xz9CFZ7A).

**Related libraries**

The Signed Heat Method has been implemented in both C++ and Python, in 2D and in 3D.

* If you're interested in using the Signed Heat Method *on* 2D surface domains, rather than in 3D space, the method has been implemented in [geometry-central](https://geometry-central.net) for [triangle mesh domains](https://geometry-central.net/surface/algorithms/signed_heat_method/), [polygon mesh domains](https://geometry-central.net/surface/algorithms/polygon_heat_solver/#signed-geodesic-distance), and [point cloud domains](https://geometry-central.net/pointcloud/algorithms/heat_solver/#signed-geodesic-distance). A demo project exists at [signed-heat-demo](https://github.com/nzfeng/signed-heat-demo).
* Likewise, Python bindings to the geometry-central C++ code has been implemented in the Python package [potpourri3d](https://github.com/nmwsharp/potpourri3d).

**Credits**

`signed_heat_method` is developed by [Nicole Feng](https://nzfeng.github.io), with advice from [Alex Kaszynski](https://github.com/akaszynski) and [Nicholas Sharp](https://nmwsharp.com).

If this code contributes to an academic publication, cite it as:
```bib
@article{Feng:2024:SHM,
	author = {Feng, Nicole and Crane, Keenan},
	title = {A Heat Method for Generalized Signed Distance},
	year = {2024},
	issue_date = {August 2024},
	publisher = {Association for Computing Machinery},
	address = {New York, NY, USA},
	volume = {43},
	number = {4},
	issn = {0730-0301},
	url = {https://doi.org/10.1145/3658220},
	doi = {10.1145/3658220},
	journal = {ACM Trans. Graph.},
	month = {jul},
	articleno = {92},
	numpages = {16}
}
```
