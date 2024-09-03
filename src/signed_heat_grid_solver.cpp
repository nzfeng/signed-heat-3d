#include "signed_heat_grid_solver.h"

SignedHeatGridSolver::SignedHeatGridSolver() {}

Vector<double> SignedHeatGridSolver::computeDistance(VertexPositionGeometry& geometry,
                                                     const SignedHeat3DOptions& options) {

    if (options.rebuild || poissonSolver == nullptr) {
        if (VERBOSE) std::cerr << "Building grid..." << std::endl;
        std::chrono::time_point<high_resolution_clock> t1, t2;
        std::chrono::duration<double, std::milli> ms_fp;
        t1 = high_resolution_clock::now();
        Vector3 c = centroid(geometry);
        double r = radius(geometry, c);
        double s = r * options.scale;
        // clang-format off
        bboxMin = {-s, -s, -s}; bboxMax = {s, s, s};
        bboxMin += c; bboxMax += c;
        glm::vec3 boundMin, boundMax;
        for (int i = 0; i < 3; i++) {
            boundMin[i] = bboxMin[i];
            boundMax[i] = bboxMax[i];
        }
        nx = 2 * std::pow(2, options.hCoef + 3); ny = nx; nz = nx;
        // clang-format on
        cellSize = 2. * s / (nx - 1);
        if (VERBOSE) std::cerr << "Building Laplacian..." << std::endl;
        laplaceMat = laplacian();
        if (VERBOSE) std::cerr << "Factorizing Laplacian..." << std::endl;
        poissonSolver.reset(new PositiveDefiniteSolver<double>(laplaceMat));
        if (VERBOSE) std::cerr << "Matrices factorized." << std::endl;
        t2 = high_resolution_clock::now();
        ms_fp = t2 - t1;
        if (VERBOSE) std::cerr << "Pre-compute time (s): " << ms_fp.count() / 1000. << std::endl;
        polyscope::VolumeGrid* psGrid = polyscope::registerVolumeGrid("domain", {nx, ny, nz}, boundMin, boundMax);
    }

    if (VERBOSE) std::cerr << "Steps 1 & 2..." << std::endl;
    // With direct convolution in R^n, it's not clear what we should pick as our timestep. Use the
    // input mesh as a heuristic.
    SurfaceMesh& mesh = geometry.mesh;
    double h = meanEdgeLength(geometry);
    shortTime = options.tCoef * h * h;
    double lambda = std::sqrt(1. / shortTime);
    size_t totalNodes = nx * ny * nz;
    Eigen::VectorXd Y = Eigen::VectorXd::Zero(3 * totalNodes);
    setFaceVectorAreas(geometry, faceAreas, faceNormals);
    for (size_t i = 0; i < nx; i++) {
        for (size_t j = 0; j < ny; j++) {
            for (size_t k = 0; k < nz; k++) {
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
            size_t i = std::floor(b[0] / cellSize);
            size_t j = std::floor(b[1] / cellSize);
            size_t k = std::floor(b[2] / cellSize);
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
        Vector<double> soln = solveSquare(LHS, RHS);
        phi = -soln.head(totalNodes);
    }
    double shift = evaluateAverageAlongSourceGeometry(geometry, phi);
    phi -= shift * Vector<double>::Ones(totalNodes);
    if (VERBOSE) std::cerr << "\tCompleted." << std::endl;
    return phi;
}

Vector<double> SignedHeatGridSolver::computeDistance(pointcloud::PointPositionNormalGeometry& pointGeom,
                                                     const SignedHeat3DOptions& options) {

    if (options.rebuild || poissonSolver == nullptr) {
        if (VERBOSE) std::cerr << "Building grid..." << std::endl;
        std::chrono::time_point<high_resolution_clock> t1, t2;
        std::chrono::duration<double, std::milli> ms_fp;
        t1 = high_resolution_clock::now();
        Vector3 c = centroid(pointGeom);
        double r = radius(pointGeom, c);
        double s = r * options.scale;
        // clang-format off
        bboxMin = {-s, -s, -s}; bboxMax = {s, s, s};
        bboxMin += c; bboxMax += c;
        glm::vec3 boundMin, boundMax;
        for (int i = 0; i < 3; i++) {
            boundMin[i] = bboxMin[i];
            boundMax[i] = bboxMax[i];
        }
        nx = 2 * std::pow(2, options.hCoef + 3); ny = nx; nz = nx;
        // clang-format on
        cellSize = 2. * s / (nx - 1);
        if (VERBOSE) std::cerr << "Building Laplacian..." << std::endl;
        laplaceMat = laplacian();
        t2 = high_resolution_clock::now();
        ms_fp = t2 - t1;
        if (VERBOSE) std::cerr << "Pre-compute time (s): " << ms_fp.count() / 1000. << std::endl;
        polyscope::VolumeGrid* psGrid = polyscope::registerVolumeGrid("domain", {nx, ny, nz}, boundMin, boundMax);
    }

    if (VERBOSE) std::cerr << "Steps 1 & 2..." << std::endl;
    // With direct convolution in R^n, it's not clear what we should pick as our timestep. Use the
    // input mesh as a heuristic.
    pointGeom.requireTuftedTriangulation();
    pointGeom.tuftedGeom->requireVertexDualAreas();
    double h = meanEdgeLength(*(pointGeom.tuftedGeom));
    shortTime = options.tCoef * h * h;
    double lambda = std::sqrt(1. / shortTime);
    size_t totalNodes = nx * ny * nz;
    Eigen::VectorXd Y = Eigen::VectorXd::Zero(3 * totalNodes);
    size_t P = pointGeom.cloud.nPoints();
    for (size_t i = 0; i < nx; i++) {
        for (size_t j = 0; j < ny; j++) {
            for (size_t k = 0; k < nz; k++) {
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
            size_t i = std::floor(b[0] / cellSize);
            size_t j = std::floor(b[1] / cellSize);
            size_t k = std::floor(b[2] / cellSize);
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
        Vector<double> soln = solveSquare(LHS, RHS);
        phi = -soln.head(totalNodes);
    }
    double shift = evaluateAverageAlongSourceGeometry(pointGeom, phi);
    phi -= shift * Vector<double>::Ones(totalNodes);
    if (VERBOSE) std::cerr << "\tCompleted." << std::endl;
    pointGeom.unrequireTuftedTriangulation();
    pointGeom.tuftedGeom->unrequireVertexDualAreas();
    return phi;
}

Vector<double> SignedHeatGridSolver::integrateGreedily(const Eigen::VectorXd& Yt) {

    Vector<double> phi = Vector<double>::Zero(nx * ny * nz);
    Vector<bool> visited = Vector<bool>::Zero(nx * ny * nz);
    std::queue<std::array<size_t, 3>> queue;
    queue.push({0, 0, 0});
    visited[0] = true;
    std::array<size_t, 3> dims = {nx, ny, nz};
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
            if (curr[i] < dims[i] - 1) {
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

    // Use 5-point stencil (well, I guess 7-point in 3D)
    size_t N = nx * ny * nz;
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

                triplets.emplace_back(currIdx, nextX, 1);
                triplets.emplace_back(currIdx, nextY, 1);
                triplets.emplace_back(currIdx, nextZ, 1);
                triplets.emplace_back(currIdx, prevX, 1);
                triplets.emplace_back(currIdx, prevY, 1);
                triplets.emplace_back(currIdx, prevZ, 1);
                triplets.emplace_back(currIdx, currIdx, -6);
            }
        }
    }
    L.setFromTriplets(triplets.begin(), triplets.end());

    return L / (cellSize * cellSize);
}

SparseMatrix<double> SignedHeatGridSolver::gradient() const {

    size_t N = nx * ny * nz;
    SparseMatrix<double> D(3 * N, N);
    std::vector<Eigen::Triplet<double>> tripletList;
    for (size_t i = 0; i < nx; i++) {
        for (size_t j = 0; j < ny; j++) {
            for (size_t k = 0; k < nz; k++) {
                size_t currIdx = indicesToNodeIndex(i, j, k);
                // if (i < nx - 1) {
                //     size_t nextX = indicesToNodeIndex(i + 1, j, k);
                //     tripletList.emplace_back(3 * currIdx, nextX, 1);
                // }
                // if (i > 0) {
                //     size_t prevX = indicesToNodeIndex(i - 1, j, k);
                //     tripletList.emplace_back(3 * currIdx, prevX, -1);
                // }
                // if (j < ny - 1) {
                //     size_t nextY = indicesToNodeIndex(i, j + 1, k);
                //     tripletList.emplace_back(3 * currIdx + 1, nextY, 1);
                // }
                // if (j > 0) {
                //     size_t prevY = indicesToNodeIndex(i, j - 1, k);
                //     tripletList.emplace_back(3 * currIdx + 1, prevY, -1);
                // }
                // if (k < nz - 1) {
                //     size_t nextZ = indicesToNodeIndex(i, j, k + 1);
                //     tripletList.emplace_back(3 * currIdx + 2, nextZ, 1);
                // }
                // if (k > 0) {
                //     size_t prevZ = indicesToNodeIndex(i, j, k - 1);
                //     tripletList.emplace_back(3 * currIdx + 2, prevZ, -1);
                // }

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
                tripletList.emplace_back(3 * currIdx, nextX, 1);
                tripletList.emplace_back(3 * currIdx, currX, -1);
                tripletList.emplace_back(3 * currIdx + 1, nextY, 1);
                tripletList.emplace_back(3 * currIdx + 1, currY, -1);
                tripletList.emplace_back(3 * currIdx + 2, nextZ, 1);
                tripletList.emplace_back(3 * currIdx + 2, currZ, -1);
            }
        }
    }
    D.setFromTriplets(tripletList.begin(), tripletList.end());

    return D / cellSize;
}

/* Evaluate a function at position q, interpolating trilinearly inside grid cells. */
double SignedHeatGridSolver::evaluateFunction(const Vector<double>& u, const Vector3& q) const {

    Vector3 d = q - bboxMin;
    int i = static_cast<int>(std::floor(d[0] / cellSize));
    int j = static_cast<int>(std::floor(d[1] / cellSize));
    int k = static_cast<int>(std::floor(d[2] / cellSize));
    Vector3 p000 = indicesToNodePosition(i, j, k);
    double v000 = u[indicesToNodeIndex(i, j, k)];
    double v100 = u[indicesToNodeIndex(i + 1, j, k)];
    double v010 = u[indicesToNodeIndex(i, j + 1, k)];
    double v001 = u[indicesToNodeIndex(i, j, k + 1)];
    double v110 = u[indicesToNodeIndex(i + 1, j + 1, k)];
    double v101 = u[indicesToNodeIndex(i + 1, j, k + 1)];
    double v011 = u[indicesToNodeIndex(i, j + 1, k + 1)];
    double v111 = u[indicesToNodeIndex(i + 1, j + 1, k + 1)];
    double tx = (q[0] - p000[0]) / cellSize;
    double ty = (q[1] - p000[1]) / cellSize;
    double tz = (q[2] - p000[2]) / cellSize;
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

    double h = cellSize;
    Vector3 d = q - bboxMin;
    size_t i = std::floor(d[0] / h);
    size_t j = std::floor(d[1] / h);
    size_t k = std::floor(d[2] / h);
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
    double tx = (q[0] - p000[0]) / h;
    double ty = (q[1] - p000[1]) / h;
    double tz = (q[2] - p000[2]) / h;
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
    // return i * (ny * nz) + j * nz + k;
    return i + j * ny + k * (nx * ny);
}

Vector3 SignedHeatGridSolver::indicesToNodePosition(const size_t& i, const size_t& j, const size_t& k) const {
    Vector3 pos = {i * cellSize, j * cellSize, k * cellSize};
    pos += bboxMin;
    return pos;
}