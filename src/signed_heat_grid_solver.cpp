#include "signed_heat_grid_solver.h"

SignedHeatGridSolver::SignedHeatGridSolver() {}

std::array<size_t, 3> SignedHeatGridSolver::getGridResolution() const {
    return resolution;
}

std::tuple<Eigen::Vector3d, Eigen::Vector3d> SignedHeatGridSolver::getBBox() const {

    Eigen::Vector3d boundMin, boundMax;
    for (int i = 0; i < 3; i++) {
        boundMin(i) = bboxMin[i];
        boundMax(i) = bboxMax[i];
    }
    return std::make_tuple(boundMin, boundMax);
}

Vector<double> SignedHeatGridSolver::computeDistance(VertexPositionGeometry& geometry,
                                                     const SignedHeat3DOptions& options) {

    if (options.rebuild) {
        if (VERBOSE) std::cerr << "Building grid..." << std::endl;
        std::chrono::time_point<high_resolution_clock> t1, t2;
        std::chrono::duration<double, std::milli> ms_fp;
        t1 = high_resolution_clock::now();
        if (!isBoundingBoxValid(options.bboxMin, options.bboxMax)) {
            std::tie(bboxMin, bboxMax) = computeBBox(geometry);
        } else {
            bboxMin = options.bboxMin;
            bboxMax = options.bboxMax;
        }
        if (isResolutionValid(options.resolution)) resolution = options.resolution;
        Vector3 diag = bboxMax - bboxMin;
        for (int i = 0; i < 3; i++) cellSizes[i] = diag[i] / (resolution[i] - 1);
        if (VERBOSE) std::cerr << "Building Laplacian..." << std::endl;
        laplaceMat = laplacian();
        t2 = high_resolution_clock::now();
        ms_fp = t2 - t1;
        if (VERBOSE) std::cerr << "Pre-compute time (s): " << ms_fp.count() / 1000. << std::endl;
    }

    if (VERBOSE) std::cerr << "Steps 1 & 2..." << std::endl;
    // With direct convolution in R^n, it's not clear what we should pick as our timestep. Use the
    // input mesh as a heuristic.
    SurfaceMesh& mesh = geometry.mesh;
    double h = meanEdgeLength(geometry);
    shortTime = options.tCoef * h * h;
    double lambda = std::sqrt(1. / shortTime);
    size_t totalNodes = resolution[0] * resolution[1] * resolution[2];
    Eigen::VectorXd Y = Eigen::VectorXd::Zero(3 * totalNodes);
    setFaceVectorAreas(geometry, faceAreas, faceNormals);
    for (size_t i = 0; i < resolution[0]; i++) {
        for (size_t j = 0; j < resolution[1]; j++) {
            for (size_t k = 0; k < resolution[2]; k++) {
                size_t idx = indicesToNodeIndex(i, j, k);
                Vector3 x = indicesToNodePosition(i, j, k);
                for (Face f : mesh.faces()) {
                    Vector3 N = faceNormals[f];
                    Vector3 y = barycenter(geometry, f);
                    double A = faceAreas[f];
                    Vector3 source = N * A * yukawaPotential(x, y, lambda);
                    for (int p = 0; p < 3; p++) Y(3 * idx + p) += source[p];
                }
                Vector3 X = {Y(3 * idx + 0), Y(3 * idx + 1), Y(3 * idx + 2)};
                X /= X.norm();
                for (int p = 0; p < 3; p++) Y(3 * idx + p) = X[p];
            }
        }
    }
    if (VERBOSE) std::cerr << "\tCompleted." << std::endl;

    // Integrate gradient to get distance.
    if (VERBOSE) std::cerr << "Step 3..." << std::endl;
    SparseMatrix<double> D = gradient(); // 3N x N
    Vector<double> divYt = D.transpose() * Y;
    for (size_t i = 0; i < divYt.size(); i++) {
        if (std::isinf(divYt[i]) || std::isnan(divYt[i])) divYt[i] = 0.;
    }
    // No level set constraints implemented for grid.
    Vector<double> phi;
    if (options.fastIntegration) {
        phi = integrateGreedily(Y);
    } else {
        SparseMatrix<double> A;
        size_t m = 0;
        std::vector<size_t> nodeIndices;
        std::vector<double> coeffs;
        std::vector<bool> hasCellBeenUsed(totalNodes, false);
        std::vector<Eigen::Triplet<double>> tripletList;
        for (Face f : mesh.faces()) {
            Vector3 b = barycenter(geometry, f);
            Vector3 d = b - bboxMin;
            size_t i = std::floor(d[0] / cellSizes[0]);
            size_t j = std::floor(d[1] / cellSizes[1]);
            size_t k = std::floor(d[2] / cellSizes[2]);
            size_t nodeIdx = indicesToNodeIndex(i, j, k);
            if (hasCellBeenUsed[nodeIdx]) continue;
            trilinearCoefficients(b, nodeIndices, coeffs);
            for (size_t i = 0; i < nodeIndices.size(); i++) tripletList.emplace_back(m, nodeIndices[i], coeffs[i]);
            hasCellBeenUsed[nodeIdx] = true;
            m++;
        }
        A.resize(m, totalNodes);
        A.setFromTriplets(tripletList.begin(), tripletList.end());
        SparseMatrix<double> Z(m, m);
        SparseMatrix<double> LHS1 = horizontalStack<double>({laplaceMat, A.transpose()});
        SparseMatrix<double> LHS2 = horizontalStack<double>({A, Z});
        SparseMatrix<double> LHS = verticalStack<double>({LHS1, LHS2});
        Vector<double> RHS = Vector<double>::Zero(totalNodes + m);
        RHS.head(totalNodes) = divYt;
        // clang-format off
        #ifndef SHM_NO_AMGCL
        Vector<double> soln = AMGCL_solve(LHS, RHS, VERBOSE);
        #else
        Vector<double> soln = solveSquare(LHS, RHS);
        #endif
        // clang-format on
        phi = -soln.head(totalNodes);
    }
    double shift = evaluateAverageAlongSourceGeometry(geometry, phi);
    phi -= shift * Vector<double>::Ones(totalNodes);
    if (VERBOSE) std::cerr << "\tCompleted." << std::endl;

    if (options.exportData) exportData(phi, options);
    return phi;
}

Vector<double> SignedHeatGridSolver::computeDistance(pointcloud::PointPositionNormalGeometry& pointGeom,
                                                     const SignedHeat3DOptions& options) {

    if (options.rebuild) {
        if (VERBOSE) std::cerr << "Building grid..." << std::endl;
        std::chrono::time_point<high_resolution_clock> t1, t2;
        std::chrono::duration<double, std::milli> ms_fp;
        t1 = high_resolution_clock::now();
        if (!isBoundingBoxValid(options.bboxMin, options.bboxMax)) {
            std::tie(bboxMin, bboxMax) = computeBBox(pointGeom);
        } else {
            bboxMin = options.bboxMin;
            bboxMax = options.bboxMax;
        }
        if (isResolutionValid(options.resolution)) resolution = options.resolution;
        Vector3 diag = bboxMax - bboxMin;
        for (int i = 0; i < 3; i++) cellSizes[i] = diag[i] / (resolution[i] - 1);
        if (VERBOSE) std::cerr << "Building Laplacian..." << std::endl;
        laplaceMat = laplacian();
        t2 = high_resolution_clock::now();
        ms_fp = t2 - t1;
        if (VERBOSE) std::cerr << "Pre-compute time (s): " << ms_fp.count() / 1000. << std::endl;
    }

    if (VERBOSE) std::cerr << "Steps 1 & 2..." << std::endl;
    // With direct convolution in R^n, it's not clear what we should pick as our timestep. Use the
    // input mesh as a heuristic.
    pointGeom.requireTuftedTriangulation();
    pointGeom.tuftedGeom->requireVertexDualAreas();
    double h = meanEdgeLength(*(pointGeom.tuftedGeom));
    shortTime = options.tCoef * h * h;
    double lambda = std::sqrt(1. / shortTime);
    size_t totalNodes = resolution[0] * resolution[1] * resolution[2];
    Eigen::VectorXd Y = Eigen::VectorXd::Zero(3 * totalNodes);
    size_t P = pointGeom.cloud.nPoints();
    for (size_t i = 0; i < resolution[0]; i++) {
        for (size_t j = 0; j < resolution[1]; j++) {
            for (size_t k = 0; k < resolution[2]; k++) {
                size_t idx = indicesToNodeIndex(i, j, k);
                Vector3 y = indicesToNodePosition(i, j, k);
                for (size_t pIdx = 0; pIdx < P; pIdx++) {
                    Vector3 x = pointGeom.positions[pIdx];
                    Vector3 n = pointGeom.normals[pIdx];
                    double A = pointGeom.tuftedGeom->vertexDualAreas[pIdx];
                    Vector3 source = n * A * yukawaPotential(x, y, lambda);
                    for (int p = 0; p < 3; p++) Y(3 * idx + p) += source[p];
                }
                Vector3 X = {Y(3 * idx + 0), Y(3 * idx + 1), Y(3 * idx + 2)};
                X /= X.norm();
                for (int p = 0; p < 3; p++) Y(3 * idx + p) = X[p];
            }
        }
    }
    if (VERBOSE) std::cerr << "\tCompleted." << std::endl;

    // Integrate gradient to get distance.
    if (VERBOSE) std::cerr << "Step 3..." << std::endl;
    SparseMatrix<double> D = gradient(); // 3N x N
    Vector<double> divYt = D.transpose() * Y;
    // No level set constraints implemented for grid.
    Vector<double> phi;
    if (options.fastIntegration) {
        phi = integrateGreedily(Y);
    } else {
        SparseMatrix<double> A;
        size_t m = 0;
        std::vector<size_t> nodeIndices;
        std::vector<double> coeffs;
        std::vector<bool> hasCellBeenUsed(totalNodes, false);
        std::vector<Eigen::Triplet<double>> tripletList;
        for (size_t pIdx = 0; pIdx < P; pIdx++) {
            Vector3 b = pointGeom.positions[pIdx];
            Vector3 d = b - bboxMin;
            size_t i = std::floor(d[0] / cellSizes[0]);
            size_t j = std::floor(d[1] / cellSizes[1]);
            size_t k = std::floor(d[2] / cellSizes[2]);
            size_t nodeIdx = indicesToNodeIndex(i, j, k);
            if (hasCellBeenUsed[nodeIdx]) continue;
            trilinearCoefficients(b, nodeIndices, coeffs);
            for (size_t i = 0; i < nodeIndices.size(); i++) tripletList.emplace_back(m, nodeIndices[i], coeffs[i]);
            hasCellBeenUsed[nodeIdx] = true;
            m++;
        }
        A.resize(m, totalNodes);
        A.setFromTriplets(tripletList.begin(), tripletList.end());
        SparseMatrix<double> Z(m, m);
        SparseMatrix<double> LHS1 = horizontalStack<double>({laplaceMat, A.transpose()});
        SparseMatrix<double> LHS2 = horizontalStack<double>({A, Z});
        SparseMatrix<double> LHS = verticalStack<double>({LHS1, LHS2});
        Vector<double> RHS = Vector<double>::Zero(totalNodes + m);
        RHS.head(totalNodes) = divYt;
        // clang-format off
        #ifndef SHM_NO_AMGCL
        Vector<double> soln = AMGCL_solve(LHS, RHS, VERBOSE);
        #else
        Vector<double> soln = solveSquare(LHS, RHS);
        #endif
        // clang-format on
        phi = -soln.head(totalNodes);
    }
    double shift = evaluateAverageAlongSourceGeometry(pointGeom, phi);
    phi -= shift * Vector<double>::Ones(totalNodes);
    if (VERBOSE) std::cerr << "\tCompleted." << std::endl;
    pointGeom.unrequireTuftedTriangulation();
    pointGeom.tuftedGeom->unrequireVertexDualAreas();
    if (options.exportData) exportData(phi, options);
    return phi;
}

Vector<double> SignedHeatGridSolver::integrateGreedily(const Eigen::VectorXd& Yt) {

    size_t nx = resolution[0];
    size_t ny = resolution[1];
    size_t nz = resolution[2];
    Vector<double> phi = Vector<double>::Zero(nx * ny * nz);
    Vector<bool> visited = Vector<bool>::Zero(nx * ny * nz);
    std::queue<std::array<size_t, 3>> queue;
    queue.push({0, 0, 0});
    visited[0] = true;
    std::array<size_t, 3> curr, next;
    while (!queue.empty()) {
        curr = queue.front();
        Vector3 p = indicesToNodePosition(curr[0], curr[1], curr[2]);
        size_t currIdx = indicesToNodeIndex(curr[0], curr[1], curr[2]);
        Eigen::Vector3d Yp = {Yt(3 * currIdx), Yt(3 * currIdx + 1), Yt(3 * currIdx + 2)};
        queue.pop();
        for (int i = 0; i < 3; i++) {
            if (curr[i] > 0) {
                next = curr;
                next[i] -= 1;
                size_t nextIdx = indicesToNodeIndex(next[0], next[1], next[2]);
                if (!visited[nextIdx]) {
                    Vector3 q = indicesToNodePosition(next[0], next[1], next[2]);
                    Vector3 edge = q - p;
                    Eigen::Vector3d Yq = {Yt(3 * nextIdx), Yt(3 * nextIdx + 1), Yt(3 * nextIdx + 2)};
                    Eigen::Vector3d Y_avg = (Yq + Yp);
                    Y_avg /= Y_avg.norm();
                    Vector3 Y = {Y_avg[0], Y_avg[1], Y_avg[2]};
                    phi[nextIdx] = phi[currIdx] + dot(Y, edge);
                    visited[nextIdx] = true;
                    queue.push(next);
                }
            }
            if (curr[i] < resolution[i] - 1) {
                next = curr;
                next[i] += 1;
                size_t nextIdx = indicesToNodeIndex(next[0], next[1], next[2]);
                if (!visited[nextIdx]) {
                    Vector3 q = indicesToNodePosition(next[0], next[1], next[2]);
                    Vector3 edge = q - p;
                    Eigen::Vector3d Yq = {Yt(3 * nextIdx), Yt(3 * nextIdx + 1), Yt(3 * nextIdx + 2)};
                    Eigen::Vector3d Y_avg = (Yq + Yp);
                    Y_avg /= Y_avg.norm();
                    Vector3 Y = {Y_avg[0], Y_avg[1], Y_avg[2]};
                    phi[nextIdx] = phi[currIdx] + dot(Y, edge);
                    visited[nextIdx] = true;
                    queue.push(next);
                }
            }
        }
    }
    return phi;
}

/* Builds negative-definite Laplace */
SparseMatrix<double> SignedHeatGridSolver::laplacian() const {


    // clang-format off
    size_t nx = resolution[0]; size_t ny = resolution[1]; size_t nz = resolution[2];
    double hx = 1. / cellSizes[0]; double hy = 1. / cellSizes[1]; double hz = 1. / cellSizes[2];
    double hx2 = hx*hx; double hy2 = hy*hy; double hz2 = hz*hz;
    // clang-format on
    size_t N = nx * ny * nz;
    // Use 5-point stencil (well, I guess 7-point in 3D)
    SparseMatrix<double> L(N, N);
    std::vector<Eigen::Triplet<double>> triplets;
    for (size_t i = 0; i < nx; i++) {
        for (size_t j = 0; j < ny; j++) {
            for (size_t k = 0; k < nz; k++) {
                size_t currIdx = indicesToNodeIndex(i, j, k);
                size_t currX = currIdx;
                size_t currY = currIdx;
                size_t currZ = currIdx;
                size_t nextX = indicesToNodeIndex(i + 1, j, k);
                size_t nextY = indicesToNodeIndex(i, j + 1, k);
                size_t nextZ = indicesToNodeIndex(i, j, k + 1);
                size_t prevX = indicesToNodeIndex(i - 1, j, k);
                size_t prevY = indicesToNodeIndex(i, j - 1, k);
                size_t prevZ = indicesToNodeIndex(i, j, k - 1);

                // Use mirroring for differences along boundary.
                if (i == nx - 1) {
                    nextX = currIdx;
                    currX = indicesToNodeIndex(i - 1, j, k);
                } else if (i == 0) {
                    prevX = currX;
                    currX = nextX;
                }
                if (j == ny - 1) {
                    nextY = currIdx;
                    currY = indicesToNodeIndex(i, j - 1, k);
                } else if (j == 0) {
                    prevY = currIdx;
                    currY = nextY;
                }
                if (k == nz - 1) {
                    nextZ = currIdx;
                    currZ = indicesToNodeIndex(i, j, k - 1);
                } else if (k == 0) {
                    prevZ = currIdx;
                    currZ = nextZ;
                }

                triplets.emplace_back(currIdx, nextX, hx2);
                triplets.emplace_back(currIdx, nextY, hy2);
                triplets.emplace_back(currIdx, nextZ, hz2);
                triplets.emplace_back(currIdx, prevX, hx2);
                triplets.emplace_back(currIdx, prevY, hy2);
                triplets.emplace_back(currIdx, prevZ, hz2);
                triplets.emplace_back(currIdx, currIdx, -2. * (hx2 + hy2 + hz2));
            }
        }
    }
    L.setFromTriplets(triplets.begin(), triplets.end());

    return L;
}

SparseMatrix<double> SignedHeatGridSolver::gradient() const {

    // clang-format off
    size_t nx = resolution[0]; size_t ny = resolution[1]; size_t nz = resolution[2];
    double hx = 1. / cellSizes[0]; double hy = 1. / cellSizes[1]; double hz = 1. / cellSizes[2];
    // clang-format on
    size_t N = nx * ny * nz;
    SparseMatrix<double> D(3 * N, N);
    std::vector<Eigen::Triplet<double>> tripletList;
    for (size_t i = 0; i < nx; i++) {
        for (size_t j = 0; j < ny; j++) {
            for (size_t k = 0; k < nz; k++) {
                size_t currIdx = indicesToNodeIndex(i, j, k);
                // forward differences
                size_t currX = currIdx;
                size_t currY = currIdx;
                size_t currZ = currIdx;
                size_t nextX = indicesToNodeIndex(i + 1, j, k);
                size_t nextY = indicesToNodeIndex(i, j + 1, k);
                size_t nextZ = indicesToNodeIndex(i, j, k + 1);
                // Use mirroring for differences along boundary.
                if (i == nx - 1) {
                    nextX = currIdx;
                    currX = indicesToNodeIndex(i - 1, j, k);
                }
                if (j == ny - 1) {
                    nextY = currIdx;
                    currY = indicesToNodeIndex(i, j - 1, k);
                }
                if (k == nz - 1) {
                    nextZ = currIdx;
                    currZ = indicesToNodeIndex(i, j, k - 1);
                }
                tripletList.emplace_back(3 * currIdx, nextX, hx);
                tripletList.emplace_back(3 * currIdx, currX, -hx);
                tripletList.emplace_back(3 * currIdx + 1, nextY, hy);
                tripletList.emplace_back(3 * currIdx + 1, currY, -hy);
                tripletList.emplace_back(3 * currIdx + 2, nextZ, hz);
                tripletList.emplace_back(3 * currIdx + 2, currZ, -hz);
            }
        }
    }
    D.setFromTriplets(tripletList.begin(), tripletList.end());

    return D;
}

/* Evaluate a function at position q, interpolating trilinearly inside grid cells. */
double SignedHeatGridSolver::evaluateFunction(const Vector<double>& u, const Vector3& q) const {

    Vector3 d = q - bboxMin;
    int i = static_cast<int>(std::floor(d[0] / cellSizes[0]));
    int j = static_cast<int>(std::floor(d[1] / cellSizes[1]));
    int k = static_cast<int>(std::floor(d[2] / cellSizes[2]));
    Vector3 p000 = indicesToNodePosition(i, j, k);
    double v000 = u[indicesToNodeIndex(i, j, k)];
    double v100 = u[indicesToNodeIndex(i + 1, j, k)];
    double v010 = u[indicesToNodeIndex(i, j + 1, k)];
    double v001 = u[indicesToNodeIndex(i, j, k + 1)];
    double v110 = u[indicesToNodeIndex(i + 1, j + 1, k)];
    double v101 = u[indicesToNodeIndex(i + 1, j, k + 1)];
    double v011 = u[indicesToNodeIndex(i, j + 1, k + 1)];
    double v111 = u[indicesToNodeIndex(i + 1, j + 1, k + 1)];
    double tx = (q[0] - p000[0]) / cellSizes[0];
    double ty = (q[1] - p000[1]) / cellSizes[1];
    double tz = (q[2] - p000[2]) / cellSizes[2];
    double v00 = v000 * (1. - tx) + v100 * tx;
    double v01 = v001 * (1. - tx) + v101 * tx;
    double v10 = v010 * (1. - tx) + v110 * tx;
    double v11 = v011 * (1. - tx) + v111 * tx;
    double v0 = v00 * (1. - ty) + v10 * ty;
    double v1 = v01 * (1. - ty) + v11 * ty;
    double v = v0 * (1. - tz) + v1 * tz;
    return v;
}

void SignedHeatGridSolver::trilinearCoefficients(const Vector3& q, std::vector<size_t>& nodeIndices,
                                                 std::vector<double>& coeffs) const {

    Vector3 d = q - bboxMin;
    size_t i = std::floor(d[0] / cellSizes[0]);
    size_t j = std::floor(d[1] / cellSizes[1]);
    size_t k = std::floor(d[2] / cellSizes[2]);
    Vector3 p000 = indicesToNodePosition(i, j, k);
    size_t i000 = indicesToNodeIndex(i, j, k);
    size_t i100 = indicesToNodeIndex(i + 1, j, k);
    size_t i010 = indicesToNodeIndex(i, j + 1, k);
    size_t i001 = indicesToNodeIndex(i, j, k + 1);
    size_t i110 = indicesToNodeIndex(i + 1, j + 1, k);
    size_t i101 = indicesToNodeIndex(i + 1, j, k + 1);
    size_t i011 = indicesToNodeIndex(i, j + 1, k + 1);
    size_t i111 = indicesToNodeIndex(i + 1, j + 1, k + 1);
    nodeIndices = {i000, i100, i010, i001, i110, i101, i011, i111};
    double tx = (q[0] - p000[0]) / cellSizes[0];
    double ty = (q[1] - p000[1]) / cellSizes[1];
    double tz = (q[2] - p000[2]) / cellSizes[2];
    coeffs = {
        (1. - tx) * (1. - ty) * (1. - tz), // 000
        tx * (1. - ty) * (1. - tz),        // 100
        (1. - tx) * ty * (1. - tz),        // 010
        (1. - tx) * (1. - ty) * tz,        // 001
        tx * ty * (1. - tz),               // 110
        tx * (1. - ty) * tz,               // 101
        (1. - tx) * ty * tz,               // 011
        tx * ty * tz                       // 111
    };
}

double SignedHeatGridSolver::evaluateAverageAlongSourceGeometry(VertexPositionGeometry& geometry,
                                                                const Vector<double>& u) const {

    // Again integrate (approximately) using 1-pt quadrature.
    SurfaceMesh& mesh = geometry.mesh;
    double shift = 0.;
    double normalization = 0.;
    for (Face f : mesh.faces()) {
        double A = faceAreas[f];
        Vector3 x = barycenter(geometry, f);
        shift += A * evaluateFunction(u, x);
        normalization += A;
    }
    shift /= normalization;
    return shift;
}

double SignedHeatGridSolver::evaluateAverageAlongSourceGeometry(pointcloud::PointPositionGeometry& pointGeom,
                                                                const Vector<double>& u) const {

    double shift = 0.;
    double normalization = 0.;
    size_t P = pointGeom.cloud.nPoints();
    for (size_t i = 0; i < P; i++) {
        double A = pointGeom.tuftedGeom->vertexDualAreas[i];
        shift += A * evaluateFunction(u, pointGeom.positions[i]);
        normalization += A;
    }
    shift /= normalization;
    return shift;
}

Vector3 SignedHeatGridSolver::barycenter(VertexPositionGeometry& geometry, const Face& f) const {
    Vector3 c = {0, 0, 0};
    for (Vertex v : f.adjacentVertices()) c += geometry.vertexPositions[v];
    c /= f.degree();
    return c;
}

size_t SignedHeatGridSolver::indicesToNodeIndex(const size_t& i, const size_t& j, const size_t& k) const {
    return i + j * resolution[0] + k * (resolution[0] * resolution[1]);
}

Vector3 SignedHeatGridSolver::indicesToNodePosition(const size_t& i, const size_t& j, const size_t& k) const {
    Vector3 pos = {i * cellSizes[0], j * cellSizes[1], k * cellSizes[2]};
    pos += bboxMin;
    return pos;
}

/*
 * Write CSV file, where each row is a node of the grid.
 * The grid positions are defined in the computeDistance() functions.
 * Columns: xCoord, yCoord, zCoord, SDF
 * The first three columns record the (x,y,z) position of the node of the grid.
 * "SDF" records the SDF value at the node.
 */
void SignedHeatGridSolver::exportData(const Vector<double>& phi, const SignedHeat3DOptions& options) const {

    std::string filename = "../export/" + options.meshname + ".csv";
    std::fstream f;
    f.open(filename, std::ios::out | std::ios::trunc);
    if (f.is_open()) {
        f << "xCoord,yCoord,zCoord,SDF" << "\n";
        for (size_t i = 0; i < resolution[0]; i++) {
            for (size_t j = 0; j < resolution[1]; j++) {
                for (size_t k = 0; k < resolution[2]; k++) {
                    Vector3 x = indicesToNodePosition(i, j, k);
                    size_t idx = indicesToNodeIndex(i, j, k);
                    f << x[0] << "," << x[1] << "," << x[2] << "," << phi[idx] << "\n";
                }
            }
        }
        f.close();
        if (VERBOSE) std::cerr << "File " << filename << " written succesfully." << std::endl;
    } else {
        if (VERBOSE) std::cerr << "Could not export '" << filename << "'!" << std::endl;
    }
}