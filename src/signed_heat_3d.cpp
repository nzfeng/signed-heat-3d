#include "signed_heat_3d.h"

Vector3 centroid(VertexPositionGeometry& geometry) {

    Vector3 c = {0, 0, 0};
    SurfaceMesh& mesh = geometry.mesh;
    for (Vertex v : mesh.vertices()) {
        c += geometry.vertexPositions[v];
    }
    c /= mesh.nVertices();
    return c;
}

double radius(VertexPositionGeometry& geometry, const Vector3& c) {

    double r = 0;
    SurfaceMesh& mesh = geometry.mesh;
    for (Vertex v : mesh.vertices()) {
        r = std::max(r, (c - geometry.vertexPositions[v]).norm());
    }
    return r;
}

Vector3 centroid(pointcloud::PointPositionGeometry& pointGeom) {

    Vector3 c = {0, 0, 0};
    size_t nPoints = pointGeom.cloud.nPoints();
    for (size_t i = 0; i < nPoints; i++) {
        c += pointGeom.positions[i];
    }
    c /= nPoints;
    return c;
}

double radius(pointcloud::PointPositionGeometry& pointGeom, const Vector3& c) {

    double r = 0;
    size_t nPoints = pointGeom.cloud.nPoints();
    for (size_t i = 0; i < nPoints; i++) {
        r = std::max(r, (c - pointGeom.positions[i]).norm());
    }
    return r;
}

bool isBoundingBoxValid(const Vector3& bboxMin, const Vector3& bboxMax) {
    return (bboxMax[0] > bboxMin[0]) && (bboxMax[1] > bboxMin[1]) && (bboxMax[2] > bboxMin[2]);
}

bool isResolutionValid(const std::array<size_t, 3>& resolution) {
    return (resolution[0] > 0) && (resolution[1] > 0) && (resolution[2] > 0);
}

std::pair<Vector3, Vector3> computeBBox(VertexPositionGeometry& geometry) {

    Vector3 c = centroid(geometry);
    double r = radius(geometry, c);
    double s = 2. * r;
    // clang-format off
    Vector3 bboxMin = {-s, -s, -s}; Vector3 bboxMax = {s, s, s};
    bboxMin += c; bboxMax += c;
    // clang-format on
    return std::make_pair(bboxMin, bboxMax);
}

std::pair<Vector3, Vector3> computeBBox(pointcloud::PointPositionNormalGeometry& pointGeom) {
    Vector3 c = centroid(pointGeom);
    double r = radius(pointGeom, c);
    double s = 2. * r;
    // clang-format off
    Vector3 bboxMin = {-s, -s, -s}; Vector3 bboxMax = {s, s, s};
    bboxMin += c; bboxMax += c;
    // clang-format on
    return std::make_pair(bboxMin, bboxMax);
}

double yukawaPotential(const Vector3& x, const Vector3& y, const double& lambda) {

    double r = (x - y).norm();
    return std::exp(-lambda * r) / r;
}

double meanEdgeLength(IntrinsicGeometryInterface& geom) {

    double h = 0;
    SurfaceMesh& mesh = geom.mesh;
    geom.requireEdgeLengths();
    for (Edge e : mesh.edges()) h += geom.edgeLengths[e];
    h /= mesh.nEdges();
    geom.unrequireEdgeLengths();
    return h;
}

void setFaceVectorAreas(VertexPositionGeometry& geometry, FaceData<double>& areas, FaceData<Vector3>& normals) {

    SurfaceMesh& mesh = geometry.mesh;
    if (mesh.isTriangular()) {
        geometry.requireFaceAreas();
        geometry.requireFaceNormals();
        areas = geometry.faceAreas;
        normals = geometry.faceNormals;
        geometry.unrequireFaceAreas();
        geometry.unrequireFaceNormals();
    }
    // Use shoelace formula.
    areas = FaceData<double>(mesh);
    normals = FaceData<Vector3>(mesh);
    for (Face f : mesh.faces()) {
        Vector3 N = {0, 0, 0};
        for (Halfedge he : f.adjacentHalfedges()) {
            Vertex vA = he.vertex();
            Vertex vB = he.next().vertex();
            Vector3 pA = geometry.vertexPositions[vA];
            Vector3 pB = geometry.vertexPositions[vB];
            N += cross(pA, pB);
        }
        N *= 0.5;
        areas[f] = N.norm();
        normals[f] = N / areas[f];
    }
}

#ifndef SHM_NO_AMGCL
Vector<double> AMGCL_solve(const SparseMatrix<double>& L, const Vector<double>& RHS, bool verbose) {

    // AMGCL needs Eigen matrices to be in row-major order.
    Eigen::SparseMatrix<double, Eigen::RowMajor> LHS = L;
    typedef amgcl::backend::eigen<double> Backend;

    typedef amgcl::make_solver<
        // Use AMG as preconditioner:
        amgcl::amg<Backend, amgcl::coarsening::smoothed_aggregation, amgcl::relaxation::spai0>,
        // Set iterative solver:
        amgcl::solver::bicgstab<Backend>>
        Solver;
    Solver solve(LHS);

    int iters;
    double error;
    size_t n = LHS.rows();
    Vector<double> x(n);
    std::tie(iters, error) = solve(LHS, RHS, x);
    if (verbose) std::cerr << "AMGCL # iters: " << iters << "\tAMGCL residual: " << error << std::endl;
    return x;
}
#endif