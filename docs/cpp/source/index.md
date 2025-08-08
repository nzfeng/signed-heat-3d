# Signed Heat Method

`signed-heat-3d` is a C++ library implementing the [Signed Heat Method](https://nzfeng.github.io/research/SignedHeatMethod/index.html) for computing **robust signed distance fields (SDFs)** to triangle meshes, polygon meshes, and point clouds in 3D. 

![teaser image](../../../media/teaser.png)

Python bindings to this C++ code also exist via the `signed_heat_method` package on PyPI: toggle the "C++/Python" switch at the top of this page to see documentation for the Python version.

**Sample:**

The library uses [geometry-central](https://geometry-central.net/) to manage mesh and point cloud structures.
```cpp
#include "signedheat3d/signed_heat_grid_solver.h"
#include "signedheat3d/signed_heat_tet_solver.h"

#include "geometrycentral/surface/vertex_position_geometry.h"
#include "geometrycentral/pointcloud/point_position_normal_geometry.h"

using namespace geometrycentral;
using namespace geometrycentral::surface;
using namespace geometrycentral::pointcloud;

// Assume we have some input geometry -- these are geometry-central objects.
SurfaceMesh mesh;
VertexPositionGeometry geometry;
PointPositionNormalGeometry pointGeom;

// Initalize tet mesh solver
SignedHeatTetSolver tetSolver = SignedHeatTetSolver();

// Configure some options
SignedHeat3DOptions solveOptions; // all the default options should be pretty good

// Solve!
Vector<double> sdf = tetSolver.computeDistance(geometry, solveOptions); // get distance to the mesh!
Vector<double> sdf = tetSolver.computeDistance(pointGeom, solveOptions); // get distance to the point cloud!

// Solve on a grid instead.
SignedHeatGridSolver gridSolver = SignedHeatGridSolver();
solveOptions.resolution = {64, 64, 64}; // change the resolution of the grid
Vector<double> sdf = gridSolver.computeDistance(geometry, solveOptions); // get distance to the mesh!
Vector<double> sdf = gridSolver.computeDistance(pointGeom, solveOptions); // get distance to the point cloud!
```

See the [sample project](https://github.com/nzfeng/signed-heat-demo-3d) to get started with a GUI and build system.

More info about the method, including a blog-style summary, an introductory 10-minute talk, and the corresponding academic paper are all located at the project page [here](https://www.youtube.com/watch?v=mw5Xz9CFZ7A).

**Related libraries**

The Signed Heat Method has been implemented in both C++ and Python, in 2D and in 3D.

* If you're interested in using the Signed Heat Method *on* 2D surface domains, rather than in 3D space, the method has been implemented in [geometry-central](https://geometry-central.net) for [triangle mesh domains](https://geometry-central.net/surface/algorithms/signed_heat_method/), [polygon mesh domains](https://geometry-central.net/surface/algorithms/polygon_heat_solver/#signed-geodesic-distance), and [point cloud domains](https://geometry-central.net/pointcloud/algorithms/heat_solver/#signed-geodesic-distance). A demo project exists at [signed-heat-demo](https://github.com/nzfeng/signed-heat-demo).
* Likewise, Python bindings to the geometry-central C++ code has been implemented in the Python package [potpourri3d](https://github.com/nmwsharp/potpourri3d).

**Credits**

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
