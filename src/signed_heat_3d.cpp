#include "signedheat3d/signed_heat_3d.h"

#include <exception>

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

/* A wrapper function around solveSquare, to handle occasional failures of direct solver. */
Vector<double> solveSquareSystem(SparseMatrix<double>& LHS, const Vector<double>& RHS, bool verbose) {
    bool success = true;
    Vector<double> soln;
    try {
        soln = solveSquare(LHS, RHS);
    } catch (const std::exception& e) {
        if (verbose) std::cerr << "Caught exception: " << e.what() << std::endl;
        success = false;
    }

    double solnNorm = soln.norm();
    double error = (LHS * soln - RHS).norm();
    double tol = 1.0;
    if (verbose) std::cerr << "Direct solver residual: " << error << std::endl;
    if (std::isinf(solnNorm) || std::isnan(solnNorm) || abs(error) > tol) {
        if (verbose) std::cerr << "Direct solver failed, using iterative solver" << std::endl;
        success = false;

        Eigen::BiCGSTAB<SparseMatrix<double>> solver;
        solver.compute(LHS);
        soln = solver.solve(RHS);
        if (verbose) {
            std::cout << "\t#iterations:     " << solver.iterations() << std::endl;
            std::cout << "\testimated error: " << solver.error() << std::endl;
        }
    }
    return soln;
}

Vector<double> solvePositiveDefiniteSystem(SparseMatrix<double>& LHS, const Vector<double>& RHS, bool verbose) {
    bool success = true;
    Vector<double> soln;
    try {
        soln = solvePositiveDefinite(LHS, RHS);
    } catch (const std::exception& e) {
        if (verbose) std::cerr << "Caught exception: " << e.what() << std::endl;
        success = false;
    }

    double solnNorm = soln.norm();
    double error = (LHS * soln - RHS).norm();
    double tol = 1.0;
    if (verbose) std::cerr << "Direct solver residual: " << error << std::endl;
    if (std::isinf(solnNorm) || std::isnan(solnNorm) || abs(error) > tol) {
        if (verbose) std::cerr << "Direct solver failed, using iterative solver" << std::endl;
        success = false;

        Eigen::ConjugateGradient<SparseMatrix<double>, Eigen::Lower | Eigen::Upper> solver;
        solver.compute(LHS);
        soln = solver.solve(RHS);
        if (verbose) {
            std::cout << "\t#iterations:     " << solver.iterations() << std::endl;
            std::cout << "\testimated error: " << solver.error() << std::endl;
        }
    }
    return soln;
}

Vector<double> solvePositiveDefiniteSystem(SparseMatrix<double>& LHS, const Vector<double>& RHS,
                                           std::unique_ptr<PositiveDefiniteSolver<double>>& solver, bool factorize,
                                           bool verbose) {
    bool success = true;
    Vector<double> soln;
    try {
        solver.reset(new PositiveDefiniteSolver<double>(LHS));
        soln = solver->solve(RHS);
    } catch (const std::exception& e) {
        if (verbose) std::cerr << "Caught exception: " << e.what() << std::endl;
        success = false;
    }

    double solnNorm = soln.norm();
    double error = (LHS * soln - RHS).norm();
    double tol = 1.0;
    if (verbose) std::cerr << "Direct solver residual: " << error << std::endl;
    if (std::isinf(solnNorm) || std::isnan(solnNorm) || abs(error) > tol) {
        if (verbose) std::cerr << "Direct solver failed, using iterative solver" << std::endl;
        success = false;

        Eigen::ConjugateGradient<SparseMatrix<double>, Eigen::Lower | Eigen::Upper> cg;
        cg.compute(LHS);
        soln = cg.solve(RHS);
        if (verbose) {
            std::cout << "\t#iterations:     " << cg.iterations() << std::endl;
            std::cout << "\testimated error: " << cg.error() << std::endl;
        }
    }
    return soln;
}

#ifndef SHM_NO_AMGCL
Vector<double> AMGCL_solve(SparseMatrix<double>& L, const Vector<double>& RHS, bool& success, bool verbose) {

    // AMGCL needs Eigen matrices to be in row-major order.
    Eigen::SparseMatrix<double, Eigen::RowMajor> LHS = L;
    typedef amgcl::backend::eigen<double> Backend;

    typedef amgcl::make_solver<
        // Use AMG as preconditioner:
        amgcl::amg<Backend, amgcl::coarsening::smoothed_aggregation, amgcl::relaxation::spai0>,
        // Set iterative solver:
        amgcl::solver::bicgstab<Backend>> // seems the most reliable
        // amgcl::solver::cg<Backend>>
        // amgcl::solver::gmres<Backend>>
        Solver;

    Solver::params prm;
    prm.solver.tol = 1e-2;
    prm.solver.maxiter = 1000;

    Solver solve(LHS, prm);

    int iters;
    double error;
    size_t n = LHS.rows();
    Vector<double> x(n);
    success = true;
    try {
        std::tie(iters, error) = solve(LHS, RHS, x);
    } catch (const std::exception& e) {
        if (verbose) {
            std::cerr << "Caught exception: " << e.what() << std::endl;
            std::cerr << "Use direct solver" << std::endl;
        }
        success = false;
        return x;
    }
    if (verbose) std::cerr << "AMGCL # iters: " << iters << "\tAMGCL residual: " << error << std::endl;

    if (std::isnan(error) || abs(error) > prm.solver.tol) {
        if (verbose) std::cerr << "AMGCL failed, use direct solver" << std::endl;
        success = false;
    }

    return x;
}
#endif