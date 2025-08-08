# signed-heat-3d (3D volumetric domains)

## Documentation is at [nzfeng.github.io/signed-heat-3d](https://nzfeng.github.io/signed-heat-3d)

C++ demo for "[A Heat Method for Generalized Signed Distance](https://nzfeng.github.io/research/SignedHeatMethod/index.html)" by [Nicole Feng](https://nzfeng.github.io/index.html) and [Keenan Crane](https://www.cs.cmu.edu/~kmcrane/), presented at SIGGRAPH 2024.

This library implements the _Signed Heat Method (SHM)_ on **3D volumetric domains**, solving for (generalized) signed distance to triangle meshes, polygon meshes, and point clouds. No assumptions are placed on the input, besides that it be consistently oriented.

![teaser image](media/teaser.png)

Check out the [sample project](https://github.com/nzfeng/signed-heat-demo-3d) to get started with a build system and a GUI.

Check out the docs at [nzfeng.github.io/signed-heat-3d](nzfeng.github.io/signed-heat-3d).

More resources:
* Python bindings have been released as the `signed-heat-method` package on PyPI. Checkout the [documentation](nzfeng.github.io/signed-heat-3d) for install instructions.
* If you're interested in using the Signed Heat Method in 2D surface domains, go to [this Github repository](https://github.com/nzfeng/signed-heat-demo) which demonstrates the [geometry-central implementation on (2D) surface meshes and point clouds](https://geometry-central.net/surface/algorithms/signed_heat_method/).
* Project page with links to paper, pseudocode, supplementals, & videos: [link](https://nzfeng.github.io/research/SignedHeatMethod/index.html)

## Performance

1. To improve performance, operators and spatial discretizations are only built as necessary, and re-used in future computations if the underlying discretization hasn't changed. This means future computations can be significantly faster than the initial solve (which includes, for example, tet mesh construction and matrix factorization.)

2. Linear solves are (optionally) accelerated using the algebraic multigrid library [AMGCL](https://amgcl.readthedocs.io/en/latest/), which (unfortunately) requires Boost. If you do not want to use Boost, use `cmake -DSHM_NO_AMGCL=On` to compile to a program without AMGCL but with solve times \~5x slower (more or less for larger/smaller problems). Force use of AMGCL via `cmake -DSHM_NO_AMGCL=Off`. Boost can be installed on macOS using `brew install boost`, and the necessary modules on Ubuntu using
```
sudo apt-get -y update
sudo apt-get -y install libboost-dev libboost-test-dev libboost-program-options-dev libboost-serialization-dev
```
Windows users should probably follow the instructions on the [Boost website](https://www.boost.org/releases/latest/).

3. There are still several further obvious areas of performance improvement, which haven't been implemented yet:
* In 3D domains, Step 1 of the Signed Heat Method (vector diffusion) can be done by convolution; the integral is evaluted simply by direct summation, even though this summation is trivially parallelizable. 
* One could optimize loop order when iterating over source/domain elements (whichever is smaller) for better cache behavior.
* More performance-critical implementations could also implement hierarchical summation.

## Citation

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
