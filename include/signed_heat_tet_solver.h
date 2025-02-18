#pragma once

#include "geometrycentral/numerical/linear_solvers.h"
#include "geometrycentral/surface/surface_mesh_factories.h"

#include "polyscope/volume_mesh.h"

#include "signed_heat_3d.h"
#include <igl/marching_tets.h>

#include <set>

#define TETLIBRARY
#include "tetgen.h"

using namespace geometrycentral;
using namespace geometrycentral::surface;

class SignedHeatTetSolver {

  public:
    SignedHeatTetSolver();

    Vector<double> computeDistance(VertexPositionGeometry& geometry,
                                   const SignedHeat3DOptions& options = SignedHeat3DOptions());

    Vector<double> computeDistance(pointcloud::PointPositionNormalGeometry& pointGeom,
                                   const SignedHeat3DOptions& options = SignedHeat3DOptions());

    void isosurface(std::unique_ptr<SurfaceMesh>& isoMesh, std::unique_ptr<VertexPositionGeometry>& isoGeom,
                    const Vector<double>& phi, double isoval = 0.) const;

    bool VERBOSE = true;

  private:
    // == mesh encoding input surface
    std::vector<int> surfaceFaces; // indexes into faces of tetmesh; sign indicates relative orientation

    // == tetmesh quantities
    Eigen::MatrixXd vertices; // vertex positions
    Eigen::MatrixXi tets;     // tetrahedra -- each row is vertex indices
    Eigen::MatrixXi faces;    // faces -- each row is vertex indices
    size_t nVertices, nTets, nFaces, nEdges;

    Eigen::VectorXd faceAreas, tetVolumes;
    Eigen::MatrixXi tetFace; // (nTets x 4) tet-face (signed) adjacency
    std::vector<std::set<size_t>> vertexTet;

    double meanNodeSpacing;
    double shortTime;

    FaceData<double> surfaceFaceAreas;    // of the source geometry
    FaceData<Vector3> surfaceFaceNormals; // of the source geometry
    std::unique_ptr<pointcloud::PointCloud> cloud;
    std::unique_ptr<pointcloud::PointPositionGeometry> pointPolyGeom; // for polygon mesh

    // == solvers
    SparseMatrix<double> laplaceMat, laplaceCR, massMat, avgMat;
    std::unique_ptr<PositiveDefiniteSolver<double>> poissonSolver, poissonSolverCR;
    std::unique_ptr<SquareSolver<double>> projectionSolver;

    // == algorithm
    Vector<double> integrateVectorField(VertexPositionGeometry& geometry, const Eigen::MatrixXd& Yt,
                                        const SignedHeat3DOptions& options);
    Vector<double> integrateVectorField(pointcloud::PointPositionGeometry& pointGeom, const Eigen::MatrixXd& Yt,
                                        const SignedHeat3DOptions& options);
    Vector<double> integrateVectorFieldToFaces(VertexPositionGeometry& geometry, const Eigen::MatrixXd& Yt,
                                               const SignedHeat3DOptions& options);
    Vector<double> integrateVectorFieldGreedily(VertexPositionGeometry& geometry, const Eigen::MatrixXd& Yt,
                                                const SignedHeat3DOptions& options);
    Vector<double> integrateVectorFieldGreedily(pointcloud::PointPositionGeometry& pointGeom, const Eigen::MatrixXd& Yt,
                                                const SignedHeat3DOptions& options);
    void integrateGreedily(const Eigen::MatrixXd& Yt, Vector<bool>& visited, Vector<double>& phi) const;
    Vector<double> integrateGreedilyMultipleLevelSets(IntrinsicGeometryInterface& geometry,
                                                      const Eigen::MatrixXd& Yt) const;
    SparseMatrix<double> buildCrouzeixRaviartLaplacian() const;
    SparseMatrix<double> buildCrouzeixRaviartMassMatrix() const;
    SparseMatrix<double> dualLaplacian() const;
    Vector<double> faceDivergence(const Eigen::MatrixXd& X) const;
    Vector<double> vertexDivergence(const Eigen::MatrixXd& X) const;
    Vector<double> projectOntoVertices(const Vector<double>& u) const;
    SparseMatrix<double> buildAveragingMatrix() const;
    double averageFaceDataOnSource(VertexPositionGeometry& geometry, const Vector<double>& phi) const;
    double averageVertexDataOnSource(VertexPositionGeometry& geometry, const Vector<double>& phi) const;
    double averageVertexDataOnSource(pointcloud::PointPositionGeometry& pointGeom, const Vector<double>& phi) const;

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
    bool tetmeshDomain(VertexPositionGeometry& geometry);
    void tetmeshPointCloud(pointcloud::PointPositionGeometry& pointGeom);
    void triangulateCube(tetgenio& cubeSurface, const Vector3& centroid, const double& radius, double scale = 2.) const;
    void tetmeshCube(tetgenio& in, tetgenio& out, const Vector3& centroid, const double& radius,
                     double scale = 2) const;
    std::vector<Vector3> buildCubeAroundSurface(const Vector3& centroid, const double& radius, double scale) const;
    void getTetmeshData(tetgenio& out);
    double computeMeanNodeSpacing() const;
};