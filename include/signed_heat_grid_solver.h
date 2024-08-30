#pragma once

#include "geometrycentral/numerical/linear_solvers.h"
#include "polyscope/volume_grid.h"

#include "signed_heat_3d.h"

using namespace geometrycentral;
using namespace geometrycentral::surface;

class SignedHeatGridSolver {

  public:
    SignedHeatGridSolver();

    Vector<double> computeDistance(VertexPositionGeometry& geometry,
                                   const SignedHeat3DOptions& options = SignedHeat3DOptions());

    Vector<double> computeDistance(pointcloud::PointPositionNormalGeometry& pointGeom,
                                   const SignedHeat3DOptions& options = SignedHeat3DOptions());

    bool VERBOSE = true;

  private:
    size_t nx, ny, nz; // number of vertices on x/y/z side of grid
    Vector3 bboxMin, bboxMax;

    double shortTime, cellSize;

    SparseMatrix<double> laplaceMat;
    std::unique_ptr<PositiveDefiniteSolver<double>> poissonSolver;

    SparseMatrix<double> laplacian() const;
    SparseMatrix<double> gradient() const;
    double evaluateFunction(const Vector<double>& u, const Vector3& q) const;
    double evaluateAverageAlongSourceGeometry(VertexPositionGeometry& geometry, const Vector<double>& u) const;
    Vector3 barycenter(VertexPositionGeometry& geometry, const Face& f) const;

    size_t indicesToNodeIndex(const size_t& i, const size_t& j, const size_t& k) const;
    Vector3 indicesToNodePosition(const size_t& i, const size_t& j, const size_t& k) const;
};