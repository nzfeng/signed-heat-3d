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