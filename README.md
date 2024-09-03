# signed-heat-3d (3D volumetric domains)

C++ demo for "[A Heat Method for Generalized Signed Distance](https://nzfeng.github.io/research/SignedHeatMethod/index.html)" by [Nicole Feng](https://nzfeng.github.io/index.html) and [Keenan Crane](https://www.cs.cmu.edu/~kmcrane/), presented at SIGGRAPH 2024.

Project page with links to paper, pseudocode, supplementals, & videos: [link](https://nzfeng.github.io/research/SignedHeatMethod/index.html)

This Github repository demonstrates the _Signed Heat Method (SHM)_ on **3D volumetric domains**, solving for (generalized) signed distance to triangle meshes, polygon meshes, and point clouds. 

For visualization, the solution is solved on a background tet mesh or grid, which you do _not_ need to supply yourself -- the program discretizes the domain for you, and you just need to supply the geometry to which you'd like the signed distance. 

If you're interested in using the Signed Heat Method in 2D surface domains, go to [this Github repository](https://github.com/nzfeng/signed-heat-demo) which demonstrates the [geometry-central implementation on (2D) surface meshes and point clouds](https://geometry-central.net/surface/algorithms/signed_heat_method/) .

![teaser image](media/teaser.pdf)

If this code contributes to academic work, please cite as:
```bibtex
@article{Feng:2024:SHM,
    author = {Feng, Nicole and Crane, Keenan},
    title = {A Heat Method for Generalized Signed Distance},
    year = {2024},
    issue_date = {July 2024},
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
    numpages = {19}
}
```

## Getting started

This project uses [geometry-central](https://geometry-central.net) for mesh computations, [Tetgen](https://www.wias-berlin.de/software/tetgen/1.5/index.html) for tet mesh generation, [Polyscope](http://polyscope.run/) for visualization, and [libigl](https://libigl.github.io/) for a marching tets implementation. These dependencies are added as git submodules, so copies will be downloaded locally when you clone this project as below.

```
git clone --recursive https://github.com/nzfeng/signed-heat-3d.git
cd SHM
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release .. # use `Debug` mode to enable checks
make -j8 # or however many cores you want to use
bin/main /path/to/mesh
```
A Polyscope GUI will open.

If you do not clone recursively, you may need to initialize/update submodules by running `git submodule update --init` or `git submodule update`.

# Mesh & point cloud input

## File formats
An input mesh may be an `obj`, `ply`, `off`, or `stl`. See [the geometry-central website](https://geometry-central.net/surface/utilities/io/#reading-meshes) for up-to-date information on supported file types.

An input point cloud should be specified using point positions and normals. Some example files, with extension `.pc`, are in the `data` directory.

The algorithm is robust to self-intersections, holes, and noise in the input geometry, and to a certain amount of inconsistent normal orientations.

# Usage

In addition to the mesh file, you can pass several arguments to the command line, including flags which are also shown as options in the GUI.

|flag | purpose|
| ------------- |-------------|
|`--g`, `--grid`| Solve on a background grid. By default, the domain will be discretized as a tet mesh. |
|`--f`, `--fast`| Solve using a less accurate, but significantly faster, method of integration. |
|`--V`, `--verbose`| Verbose output. On by default|
|`--h`| Controls the tet/grid spacing proportional to $2^{-h}$, with larger values indicating more refinement. Default value is 0.|
|`--help`| Display help. |

To improve performance, operators and spatial discretizations are only built as necessary, and re-used in future computations if the underlying discretization hasn't changed. This means future computations can be significantly faster than the initial solve (which includes, for example, tet mesh construction and matrix factorization.)

# Performance

There is no acceleration is applied in this program, even though there are several obvious areas of performance improvement.

In 3D domains, Step 1 of the Signed Heat Method (vector diffusion) can be done by convolution; the integral is evaluted simply by direct summation, even though Step 1 is trivially parallelizable. 

One could also optimize loop order when iterating over source/domain elements (whichever is smaller) for better cache behavior.

More performance-critical implementations could also implement hierarchical summation.

# Output

Polyscope lets you inspect your solution on the interior of the domain by adding [_slice planes_](https://polyscope.run/features/slice_planes/). In this program, you can also contour your solution, and export isosurfaces as OBJ files.
