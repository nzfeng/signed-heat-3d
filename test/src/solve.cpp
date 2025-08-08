#include "geometrycentral/surface/meshio.h"

#include "gtest/gtest.h"

#include "signedheat3d/signed_heat_grid_solver.h"
#include "signedheat3d/signed_heat_tet_solver.h"

#include <chrono>
using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;
std::chrono::time_point<high_resolution_clock> t1, t2;
std::chrono::duration<double, std::milli> ms_fp;

using namespace geometrycentral;
using namespace geometrycentral::surface;

namespace {

class SignedDistanceSolversTest : public ::testing::Test {

    // class SignedDistanceSolversTest {

  public:
    // Constructor
    SignedDistanceSolversTest() {

        // Load in mesh
        std::tie(mesh, geometry) = readSurfaceMesh("../assets/bunny_small.obj");
        // Just initialize point cloud from mesh
        size_t nPts = mesh->nVertices();
        cloud = std::unique_ptr<pointcloud::PointCloud>(new pointcloud::PointCloud(nPts));
        pointcloud::PointData<Vector3> pointPositions = pointcloud::PointData<Vector3>(*cloud);
        pointcloud::PointData<Vector3> pointNormals = pointcloud::PointData<Vector3>(*cloud);
        geometry->requireVertexNormals();
        for (size_t i = 0; i < nPts; i++) {
            pointPositions[i] = geometry->vertexPositions[i];
            pointNormals[i] = geometry->vertexNormals[i];
        }
        pointGeom = std::unique_ptr<pointcloud::PointPositionNormalGeometry>(
            new pointcloud::PointPositionNormalGeometry(*cloud, pointPositions, pointNormals));

        // Initialize solvers
        tetSolver = std::unique_ptr<SignedHeatTetSolver>(new SignedHeatTetSolver());
        gridSolver = std::unique_ptr<SignedHeatGridSolver>(new SignedHeatGridSolver());
        tetSolver->VERBOSE = true;
        gridSolver->VERBOSE = true;

        // Set some "standard" options
        options.levelSetConstraint = LevelSetConstraint::ZeroSet;
        options.tCoef = 1.0;
        options.resolution = {16, 16, 16};
    }

    // For all these tests, naively double-loop without acceleration, and simply compute pseudonormal distance.
    Vector<double> tetDistanceToMesh() {
        Eigen::MatrixXd nodes = tetSolver->getVertices();
        geometry->requireFaceNormals();
        size_t nNodes = nodes.rows();
        Vector<double> sdf(nNodes);
        for (size_t i = 0; i < nNodes; i++) {
            Vector3 q = {nodes(i, 0), nodes(i, 1), nodes(i, 2)};
            // Assume tet mesh is more refined than mesh
            double min_dist = std::numeric_limits<double>::infinity();
            size_t min_idx = 0;
            for (size_t fIdx = 0; fIdx < mesh->nFaces(); fIdx++) {
                Halfedge he = mesh->face(fIdx).halfedge();
                Vector3 a = geometry->vertexPositions[he.vertex()];
                Vector3 b = geometry->vertexPositions[he.next().vertex()];
                Vector3 c = geometry->vertexPositions[he.next().next().vertex()];
                double dist = distanceToTriangle(q, a, b, c);
                if (dist < min_dist) {
                    min_dist = dist;
                    min_idx = fIdx;
                }
            }
            Vector3 n = geometry->faceNormals[min_idx];
            Vector3 c = {0, 0, 0};
            for (Vertex v : mesh->face(min_idx).adjacentVertices()) {
                c += geometry->vertexPositions[v];
            }
            c /= mesh->face(min_idx).degree();
            double s = dot(q - c, n) > 0. ? 1. : -1.;
            sdf[i] = s * min_dist;
        }
        return sdf;
    }

    // Vector<double> tetApproximateDistanceToPointCloud() {
    //     Eigen::MatrixXd nodes = tetSolver->getVertices();
    //     pointGeom->requireNormals();
    //     size_t nNodes = nodes.rows();
    //     size_t nPoints = cloud->nPoints();
    //     Vector<double> sdf(nNodes);
    //     for (size_t i = 0; i < nNodes; i++) {
    //         Vector3 q = {nodes(i, 0), nodes(i, 1), nodes(i, 2)};
    //         // Assume tet mesh is more refined than the point cloud, so loop over the point cloud in the inner loop
    //         double min_dist = std::numeric_limits<double>::infinity();
    //         size_t min_idx = 0;
    //         for (size_t ptIdx = 0; ptIdx < nPoints; ptIdx++) {
    //             double dist = (q - pointGeom->positions[ptIdx]).norm();
    //             if (dist < min_dist) {
    //                 min_dist = dist;
    //                 min_idx = ptIdx;
    //             }
    //         }
    //         Vector3 n = pointGeom->normals[min_idx];
    //         Vector3 c = pointGeom->positions[min_idx];
    //         double s = dot(q - c, n) > 0. ? 1. : -1.;
    //         sdf[i] = s * min_dist;
    //     }
    //     return sdf;
    // }

    Vector<double> gridDistanceToMesh() {
        std::array<size_t, 3> resolution = gridSolver->getGridResolution();
        geometry->requireFaceNormals();
        size_t nNodes = resolution[0] * resolution[1] * resolution[2];
        Vector<double> sdf(nNodes);
        for (size_t i = 0; i < resolution[0]; i++) {
            for (size_t j = 0; j < resolution[1]; j++) {
                for (size_t k = 0; k < resolution[2]; k++) {
                    size_t idx = gridSolver->indicesToNodeIndex(i, j, k);
                    Vector3 q = gridSolver->indicesToNodePosition(i, j, k);
                    // Assume grid is more refined than mesh
                    double min_dist = std::numeric_limits<double>::infinity();
                    size_t min_idx = 0;
                    for (size_t fIdx = 0; fIdx < mesh->nFaces(); fIdx++) {
                        Halfedge he = mesh->face(fIdx).halfedge();
                        Vector3 a = geometry->vertexPositions[he.vertex()];
                        Vector3 b = geometry->vertexPositions[he.next().vertex()];
                        Vector3 c = geometry->vertexPositions[he.next().next().vertex()];
                        double dist = distanceToTriangle(q, a, b, c);
                        if (dist < min_dist) {
                            min_dist = dist;
                            min_idx = fIdx;
                        }
                    }
                    Vector3 n = geometry->faceNormals[min_idx];
                    Vector3 c = {0, 0, 0};
                    for (Vertex v : mesh->face(min_idx).adjacentVertices()) {
                        c += geometry->vertexPositions[v];
                    }
                    c /= mesh->face(min_idx).degree();
                    double s = dot(q - c, n) > 0. ? 1. : -1.;
                    sdf[idx] = s * min_dist;
                }
            }
        }
        return sdf;
    }

    std::unique_ptr<SignedHeatTetSolver> tetSolver;
    std::unique_ptr<SignedHeatGridSolver> gridSolver;
    SignedHeat3DOptions options;

    std::unique_ptr<SurfaceMesh> mesh;
    std::unique_ptr<VertexPositionGeometry> geometry;
    std::unique_ptr<pointcloud::PointCloud> cloud;
    std::unique_ptr<pointcloud::PointPositionNormalGeometry> pointGeom;

  protected:
    // quick-copied off https://stackoverflow.com/a/74395029
    Vector3 closestPointOnTriangle(const Vector3& p, const Vector3& a, const Vector3& b, const Vector3& c) {
        const Vector3 ab = b - a;
        const Vector3 ac = c - a;
        const Vector3 ap = p - a;

        const float d1 = dot(ab, ap);
        const float d2 = dot(ac, ap);
        if (d1 <= 0.f && d2 <= 0.f) return a; // #1

        const Vector3 bp = p - b;
        const float d3 = dot(ab, bp);
        const float d4 = dot(ac, bp);
        if (d3 >= 0.f && d4 <= d3) return b; // #2

        const Vector3 cp = p - c;
        const float d5 = dot(ab, cp);
        const float d6 = dot(ac, cp);
        if (d6 >= 0.f && d5 <= d6) return c; // #3

        const float vc = d1 * d4 - d3 * d2;
        if (vc <= 0.f && d1 >= 0.f && d3 <= 0.f) {
            const float v = d1 / (d1 - d3);
            return a + v * ab; // #4
        }

        const float vb = d5 * d2 - d1 * d6;
        if (vb <= 0.f && d2 >= 0.f && d6 <= 0.f) {
            const float v = d2 / (d2 - d6);
            return a + v * ac; // #5
        }

        const float va = d3 * d6 - d5 * d4;
        if (va <= 0.f && (d4 - d3) >= 0.f && (d5 - d6) >= 0.f) {
            const float v = (d4 - d3) / ((d4 - d3) + (d5 - d6));
            return b + v * (c - b); // #6
        }

        const float denom = 1.f / (va + vb + vc);
        const float v = vb * denom;
        const float w = vc * denom;
        return a + v * ab + w * ac; // #0
    }

    double distanceToTriangle(const Vector3& p, const Vector3& a, const Vector3& b, const Vector3& c) {
        Vector3 cp = closestPointOnTriangle(p, a, b, c);
        return (p - cp).norm();
    }
};

double range(const Vector<double>& u) {
    double minVal = std::numeric_limits<double>::infinity();
    double maxVal = 0.;
    for (size_t i = 0; i < u.size(); i++) {
        minVal = std::min(minVal, u[i]);
        maxVal = std::max(maxVal, u[i]);
    }
    return maxVal - minVal;
}

TEST_F(SignedDistanceSolversTest, tetComputeDistanceToMesh) {

    t1 = high_resolution_clock::now();
    Vector<double> phi = tetSolver->computeDistance(*geometry, options);
    t2 = high_resolution_clock::now();
    ms_fp = t2 - t1;
    std::cerr << "Solve time (s): " << ms_fp.count() / 1000. << std::endl;

    phi = tetSolver->computeDistance(*geometry, options); // solve twice to test "rebuild" functionality

    Vector<double> sdf = tetDistanceToMesh();

    double error = (phi.cwiseAbs() - sdf.cwiseAbs()).mean() / range(sdf);

    EXPECT_TRUE(std::abs(error) < 2e-2)
        << "[LevelSetConstraint::ZeroSet] SDF not close to approximate ground-truth, residual = " << error;

    options.levelSetConstraint = LevelSetConstraint::None;
    phi = tetSolver->computeDistance(*geometry, options);
    error = (phi.cwiseAbs() - sdf.cwiseAbs()).mean() / range(sdf);
    EXPECT_TRUE(std::abs(error) < 2e-2)
        << "[LevelSetConstraint::None] SDF not close to approximate ground-truth, residual = " << error;

    options.levelSetConstraint = LevelSetConstraint::Multiple;
    phi = tetSolver->computeDistance(*geometry, options);
    error = (phi.cwiseAbs() - sdf.cwiseAbs()).mean() / range(sdf);
    EXPECT_TRUE(std::abs(error) < 1e-1)
        << "[LevelSetConstraint::Multiple] SDF not close to approximate ground-truth, residual = " << error;
}

TEST_F(SignedDistanceSolversTest, tetDistanceToPointCloud) {
    t1 = high_resolution_clock::now();
    Vector<double> phi = tetSolver->computeDistance(*pointGeom, options);
    t2 = high_resolution_clock::now();
    ms_fp = t2 - t1;
    std::cerr << "Solve time (s): " << ms_fp.count() / 1000. << std::endl;

    phi = tetSolver->computeDistance(*pointGeom, options); // solve twice to test "rebuild" functionality

    Vector<double> sdf = tetDistanceToMesh();

    double error = (phi.cwiseAbs() - sdf.cwiseAbs()).mean() / range(sdf);

    EXPECT_TRUE(std::abs(error) < 2e-2)
        << "[LevelSetConstraint::ZeroSet] SDF not close to approximate ground-truth, residual = " << error;

    options.levelSetConstraint = LevelSetConstraint::None;
    phi = tetSolver->computeDistance(*pointGeom, options);
    error = (phi.cwiseAbs() - sdf.cwiseAbs()).mean() / range(sdf);
    EXPECT_TRUE(std::abs(error) < 2e-0)
        << "[LevelSetConstraint::None] SDF not close to approximate ground-truth, residual = " << error;

    options.levelSetConstraint = LevelSetConstraint::Multiple;
    phi = tetSolver->computeDistance(*pointGeom, options);
    error = (phi.cwiseAbs() - sdf.cwiseAbs()).mean() / range(sdf);
    EXPECT_TRUE(std::abs(error) < 2e-2)
        << "[LevelSetConstraint::Multiple] SDF not close to approximate ground-truth, residual = " << error;
}

TEST_F(SignedDistanceSolversTest, gridDistanceToMesh) {
    t1 = high_resolution_clock::now();
    Vector<double> phi = gridSolver->computeDistance(*geometry, options);
    t2 = high_resolution_clock::now();
    ms_fp = t2 - t1;
    std::cerr << "Solve time (s): " << ms_fp.count() / 1000. << std::endl;

    phi = gridSolver->computeDistance(*geometry, options); // solve twice to test "rebuild" functionality

    Vector<double> sdf = gridDistanceToMesh();

    double error = (phi.cwiseAbs() - sdf.cwiseAbs()).mean() / range(sdf);

    EXPECT_TRUE(std::abs(error) < 2e-2) << "SDF not close to approximate ground-truth, residual = " << error;
}

TEST_F(SignedDistanceSolversTest, gridDistanceToPointCloud) {
    t1 = high_resolution_clock::now();
    Vector<double> phi = gridSolver->computeDistance(*pointGeom, options);
    t2 = high_resolution_clock::now();
    ms_fp = t2 - t1;
    std::cerr << "Solve time (s): " << ms_fp.count() / 1000. << std::endl;

    phi = gridSolver->computeDistance(*pointGeom, options); // solve twice to test "rebuild" functionality

    Vector<double> sdf = gridDistanceToMesh();

    double error = (phi.cwiseAbs() - sdf.cwiseAbs()).mean() / range(sdf);

    EXPECT_TRUE(std::abs(error) < 2e-2) << "SDF not close to approximate ground-truth, residual = " << error;
}

} // namespace

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}