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

Vector<double> AMGCL_solve(const Eigen::SparseMatrix<double, Eigen::RowMajor>& LHS, const Vector<double>& RHS,
                           bool verbose) {

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

Vector<double> AMGCL_blockSolve(const Eigen::SparseMatrix<double, Eigen::RowMajor>& L,
                                const Eigen::SparseMatrix<double, Eigen::RowMajor>& A,
                                const Eigen::SparseMatrix<double, Eigen::RowMajor>& Z, const Vector<double>& rhs,
                                bool verbose) {

    Eigen::SparseMatrix<double, Eigen::RowMajor> LHS1 = horizontalStack<double>({Z, A});             // (m x m) x (m, n)
    Eigen::SparseMatrix<double, Eigen::RowMajor> LHS2 = horizontalStack<double>({A.transpose(), L}); // (n x m) x (n, n)
    Eigen::SparseMatrix<double, Eigen::RowMajor> LHS = verticalStack<double>({LHS1, LHS2});
    size_t m = A.rows();
    size_t n = A.cols();
    Vector<double> RHS = Vector<double>::Zero(m + n);
    RHS.tail(n) = rhs;

    typedef amgcl::backend::eigen<double> Backend;
    // Solver::params prm;
    // prm.solver.tol = 1e-8;
    // prm.solver.maxiter = 100;
    // prm.active_rows = m;

    boost::property_tree::ptree prm;
    // prm.put("solver.tol", 1e-3);
    // prm.put("solver.maxiter", 100);
    prm.put("solver.active_rows", m);

    typedef amgcl::amg<Backend, amgcl::runtime::coarsening::wrapper, amgcl::runtime::relaxation::wrapper> PPrecond;
    typedef amgcl::relaxation::as_preconditioner<Backend, amgcl::runtime::relaxation::wrapper> SPrecond;
    amgcl::make_solver<amgcl::preconditioner::cpr<PPrecond, SPrecond>, amgcl::runtime::solver::wrapper<Backend>> solve(
        LHS, prm);

    int iters;
    double error;
    Vector<double> x(m + n);
    std::tie(iters, error) = solve(RHS, x);
    if (verbose) std::cerr << "AMGCL # iters: " << iters << "\tAMGCL residual: " << error << std::endl;
    return x;
}