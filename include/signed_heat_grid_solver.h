#pragma once

#include "geometrycentral/numerical/linear_solvers.h"

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

    // Expose parameters for visualizing the grid
    std::vector<size_t> getGridResolution() const;
    std::tuple<Eigen::Vector3d, Eigen::Vector3d> getBBox() const;

  private:
    double shortTime, cellSize;
    size_t nx = 0;
    size_t ny, nz; // number of vertices on x/y/z side of grid
    Vector3 bboxMin, bboxMax;

    Eigen::SparseMatrix<double, Eigen::RowMajor> laplaceMat;
    FaceData<double> faceAreas;    // of the source geometry
    FaceData<Vector3> faceNormals; // of the source geometry

    Eigen::SparseMatrix<double, Eigen::RowMajor> laplacian() const;
    Eigen::SparseMatrix<double, Eigen::RowMajor> gradient() const;
    Vector<double> integrateGreedily(const Eigen::VectorXd& Yt);
    double evaluateFunction(const Vector<double>& u, const Vector3& q) const;
    void trilinearCoefficients(const Vector3& q, std::vector<size_t>& nodeIndices, std::vector<double>& coeffs) const;
    double evaluateAverageAlongSourceGeometry(VertexPositionGeometry& geometry, const Vector<double>& u) const;
    double evaluateAverageAlongSourceGeometry(pointcloud::PointPositionGeometry& pointGeom,
                                              const Vector<double>& u) const;
    Vector3 barycenter(VertexPositionGeometry& geometry, const Face& f) const;
    size_t indicesToNodeIndex(const size_t& i, const size_t& j, const size_t& k) const;
    Vector3 indicesToNodePosition(const size_t& i, const size_t& j, const size_t& k) const;
    void exportData(const Vector<double>& phi, const SignedHeat3DOptions& options) const;
};