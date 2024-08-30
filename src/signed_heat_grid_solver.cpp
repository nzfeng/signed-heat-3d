#include "signed_heat_grid_solver.h"

SignedHeatGridSolver::SignedHeatGridSolver() {}

Vector<double> SignedHeatGridSolver::computeDistance(VertexPositionGeometry& geometry,
                                                     const SignedHeat3DOptions& options) {

    if (options.rebuild) {
        Vector3 c = centroid(geometry);
        double r = radius(geometry, c);
        double s = r * options.scale;
        // clang-format off
        c = {1,0,0};
        bboxMin = {-s, -s, -s}; bboxMax = {s, s, s};
        bboxMin += c; bboxMax += c;
        glm::uvec3 boundMin, boundMax;
        for (int i = 0; i < 3; i++) {
            boundMin[i] = bboxMin[i];
            boundMax[i] = bboxMax[i];
        }
        nx = 2 * std::pow(10, options.hCoef); ny = nx; nz = nx;
        // clang-format on
        cellSize = 2. * s / nx;
        if (VERBOSE) std::cerr << "Building Laplacian..." << std::endl;
        laplaceMat = laplacian();
        poissonSolver.reset(new PositiveDefiniteSolver<double>(laplaceMat));
        if (VERBOSE) std::cerr << "Matrices factorized." << std::endl;
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
    Eigen::MatrixXd Y = Eigen::MatrixXd::Zero(totalNodes, 3);
    geometry.requireFaceNormals();
    geometry.requireFaceAreas();
    for (Face f : mesh.faces()) {
        Vector3 N = geometry.faceNormals[f];
        Vector3 y = barycenter(geometry, f);
        double A = geometry.faceAreas[f];
        for (size_t k = 0; k < nz; k++) {
            for (size_t j = 0; j < ny; j++) {
                for (size_t i = 0; i < nx; i++) {
                    size_t idx = indicesToNodeIndex(i, j, k);
                    Vector3 x = indicesToNodePosition(i, j, k);
                    Vector3 source = N * A * yukawaPotential(x, y, lambda);
                    for (int p = 0; p < 3; p++) Y(idx, p) += source[p];
                }
            }
        }
    }
    geometry.unrequireFaceNormals();
    geometry.unrequireFaceAreas();
    if (VERBOSE) std::cerr << "\tCompleted." << std::endl;

    // Integrate gradient to get distance.
    if (VERBOSE) std::cerr << "Step 3..." << std::endl;
    SparseMatrix<double> D = gradient(); // 3N x N
    Vector<double> divYt = D.transpose() * Y;
    Vector<double> phi;
    if (options.levelSetConstraint == LevelSetConstraint::None) {
        phi = poissonSolver->solve(divYt);
        double shift = evaluateAverageAlongSourceGeometry(geometry, phi);
        phi -= shift * Vector<double>::Ones(totalNodes);
    } else if (options.levelSetConstraint == LevelSetConstraint::ZeroSet) {
        // Add constraint that function is zero at barycenter of each face of the input surface is zero (when
        // trilinearly interpolated). WARNING: This assumes that faces are smaller than grid cells.
        // SparseMatrix<double> C;
        // size_t m = 0;
        // std::vector<size_t> nodeIndices;
        // std::vector<double> coeffs;
        // std::vector<Eigen::Triplet<double>> tripletList;
        // std::vector<bool> hasCellBeenUsed(totalNodes, false);
        // double h = hCoef * gridSize;
        // for (Face f : mesh.faces()) {
        //     Vector3 b = barycenter(geometry, f);
        //     // Hack: Only enforce constraint at most for 1 triangle per grid cell
        //     Vector3 d = b - bboxMin;
        //     size_t i = std::floor(b[0] / h);
        //     size_t j = std::floor(b[1] / h);
        //     size_t k = std::floor(b[2] / h);
        //     size_t nodeIdx = indicesToNodeIndex(i, j, k);
        //     if (hasCellBeenUsed[nodeIdx]) continue;

        //     trilinearCoefficients(b, nodeIndices, coeffs);
        //     for (size_t i = 0; i < nodeIndices.size(); i++) {
        //         tripletList.emplace_back(m, nodeIndices[i], coeffs[i]);
        //     }
        //     hasCellBeenUsed[nodeIdx] = true;
        //     m++;
        // }
        // C.resize(m, totalNodes);
        // C.setFromTriplets(tripletList.begin(), tripletList.end());

        // SparseMatrix<double> Z(m, m);
        // SparseMatrix<double> LHS1 = horizontalStack<double>({L, C.transpose()});
        // SparseMatrix<double> LHS2 = horizontalStack<double>({C, Z});
        // SparseMatrix<double> LHS = verticalStack<double>({LHS1, LHS2});
        // Vector<double> RHS = Vector<double>::Zero(totalNodes + m);
        // RHS.head(totalNodes) = divYt;
        // shiftDiagonal(LHS, 1e-16);
        // Vector<double> soln = solveSquare(LHS, RHS);
        // lastPhi = soln.head(totalNodes);
    } else if (options.levelSetConstraint == LevelSetConstraint::Multiple) {
    }
    if (VERBOSE) std::cerr << "\tCompleted." << std::endl;
    return phi;
}

Vector<double> SignedHeatGridSolver::computeDistance(pointcloud::PointPositionNormalGeometry& pointGeom,
                                                     const SignedHeat3DOptions& options) {
    // TODO
}

/* Builds negative-definite Laplace; should be equal to D^TD (where D = gradient operator). */
SparseMatrix<double> SignedHeatGridSolver::laplacian() const {

    // Use 5-point stencil (well, I guess 7-point in 3D)
    size_t N = nx * ny * nz;
    SparseMatrix<double> L(N, N);
    std::vector<Eigen::Triplet<double>> triplets;
    for (size_t k = 0; k < nz; k++) {
        for (size_t j = 0; j < ny; j++) {
            for (size_t i = 0; i < nx; i++) {
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

                // Use mirroring for differences along boundary. This means the Laplacian will just be zero here.
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
    // Forward differences; could also take centered differences
    for (size_t k = 0; k < nz; k++) {
        for (size_t j = 0; j < ny; j++) {
            for (size_t i = 0; i < nx; i++) {
                size_t currIdx = indicesToNodeIndex(i, j, k);
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
                // forward difference in x
                tripletList.emplace_back(3 * currIdx, nextX, 1);
                tripletList.emplace_back(3 * currIdx, currX, -1);
                // forward difference in y
                tripletList.emplace_back(3 * currIdx + 1, nextY, 1);
                tripletList.emplace_back(3 * currIdx + 1, currY, -1);
                // forward difference in z
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

double SignedHeatGridSolver::evaluateAverageAlongSourceGeometry(VertexPositionGeometry& geometry,
                                                                const Vector<double>& u) const {

    // Again integrate (approximately) using 1-pt quadrature.
    SurfaceMesh& mesh = geometry.mesh;
    geometry.requireFaceAreas();
    double shift = 0.;
    double normalization = 0.;
    for (Face f : mesh.faces()) {
        double A = geometry.faceAreas[f];
        Vector3 x = barycenter(geometry, f);
        shift += A * evaluateFunction(u, x);
        normalization += A;
    }
    geometry.unrequireFaceAreas();
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
    return i + j * nx + k * (nx * ny);
}

Vector3 SignedHeatGridSolver::indicesToNodePosition(const size_t& i, const size_t& j, const size_t& k) const {
    Vector3 pos = {i * cellSize, j * cellSize, k * cellSize};
    pos += bboxMin;
    return pos;
}