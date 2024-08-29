#pragma once

#include "geometrycentral/numerical/linear_solvers.h"
#include "geometrycentral/pointcloud/point_position_normal_geometry.h"
#include "geometrycentral/surface/surface_mesh.h"
#include "geometrycentral/surface/vertex_position_geometry.h"

#include "polyscope/volume_mesh.h"

#include "signed_heat_3d.h"
#include <igl/marching_tets.h>

#define TETLIBRARY
#include "tetgen.h"

#include <queue>

using namespace geometrycentral;
using namespace geometrycentral::surface;

class SignedHeatTetSolver {

  public:
    SignedHeatTetSolver();

    Vector<double> computeDistance(VertexPositionGeometry& geometry,
                                   const SignedHeat3DOptions& options = SignedHeat3DOptions());

    Vector<double> computeDistance(PointPositionGeometry& pointGeom,
                                   const SignedHeat3DOptions& options = SignedHeat3DOptions());

    void extractIsosurface(std::unique_ptr<SurfaceMesh>& isoMesh, std::unique_ptr<VertexPositionGeometry>& isoGeom,
                           const Vector<double>& phi, double isoval = 0.) const;

  private:
    // == mesh encoding input surface
    std::vector<int> surfaceFaces; // indexes into faces of tetmesh; sign indicates relative orientation

    std::unique_ptr<pointcloud::PointCloud> cloud;
    std::unique_ptr<pointcloud::PointPositionNormalGeometry> pointGeom;
    polyscope::PointCloud* psCloud;

    // == tetmesh quantities
    Eigen::MatrixXd vertices; // vertex positions
    Eigen::MatrixXi tets;     // tetrahedra -- each row is vertex indices
    Eigen::MatrixXi faces;    // faces -- each row is vertex indices
    Eigen::MatrixXi edges;    // edges -- each row is vertex indices
    size_t nVertices, nTets, nFaces, nEdges;

    Eigen::VectorXd faceAreas, tetVolumes, vertexDualVolumes;
    Eigen::MatrixXi tetFace;    // (nTets x 4) tet-face (signed) adjacency
    Eigen::MatrixXi faceTet;    // (nFaces x 2) tet-face adjacency (interior tet 1st, -1 if no neighboring tet)
    Eigen::MatrixXi edgeVertex; // (nEdges x 2) edge-vertex (signed) adjacency

    double meanNodeSpacing;
    double shortTime;
    bool VERBOSE = true;

    // == solvers
    SparseMatrix<double> laplaceMat, massMat, avgMat;
    std::unique_ptr<PositiveDefiniteSolver<double>> poissonSolver;
    std::unique_ptr<SquareSolver<double>> projectionSolver;

    // == algorithm
    Vector<double> computeDistance(VertexPositionGeometry& geometry, const SignedHeat3DOptions& options);
    Vector<double> computeDistance(PointPositionNormalGeometry& pointGeom, const SignedHeat3DOptions& options);
    SparseMatrix<double> buildCrouzeixRaviartLaplacian() const;
    SparseMatrix<double> buildCrouzeixRaviartMassMatrix() const;
    Vector<double> faceDivergence(const Eigen::MatrixXd& X) const;
    Vector<double> projectOntoVertices(const Vector<double>& u) const;
    SparseMatrix<double> buildAveragingMatrix() const;
    double computeAverageValueOnSource(const Vector<double>& phi) const;

    //== tet mesh utilities
    Eigen::VectorXd computeTetVolumes() const;
    double computeTetVolume(size_t tIdx) const;
    double tetVolume(const Eigen::Vector3d& a, const Eigen::Vector3d& b, const Eigen::Vector3d& c,
                     const Eigen::Vector3d& d) const;
    Eigen::Vector3d areaWeightedNormalVector(int fIdx) const;
    Eigen::Vector3d faceBarycenter(size_t fIdx) const;

    //== tet-meshing
    std::string TET_PREFIX = "pq1.414zfenna"; // need -f, -e to output all faces, edges in tetmesh; -nn for adjacency
    std::string TETFLAGS, TETFLAGS_PRESERVE;
    void buildTetMesh(const SignedHeat3DOptions& options, bool pointCloud);
    void tetmeshDomain(VertexPositionGeometry& geometry);
    void tetmeshPointCloud(PointPositionGeometry& pointGeom);
    void triangulateCube(tetgenio& cubeSurface, const Vector3& centroid, const double& radius, double scale = 2) const;
    std::vector<Vector3> buildCubeAroundSurface(const Vector3& centroid, const double& radius, double scale) const;
    void getTetmeshData(tetgenio& out);
    double computeMeanNodeSpacing() const;
    Vector3 centroid(VertexPositionGeometry& geometry) const;
    Vector3 centroid(PointPositionGeometry& pointGeom) const;
    double radius(VertexPositionGeometry& geometry, const Vector3& centroid) const;
    double radius(PointPositionGeometry& pointGeom, const Vector3& c) const;
};