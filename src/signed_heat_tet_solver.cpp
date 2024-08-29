#include "signed_heat_tet_solver.h"

SignedHeatTetSolver::SignedHeatTetSolver() {}

// =============== ALGORITHM

Vector<double> SignedHeatTetSolver::computeDistance(VertexPositionGeometry& geometry,
                                                    const SignedHeat3DOptions& options) {

    buildTetMesh(options, false);

    if (VERBOSE) std::cerr << "Step 1..." << std::endl;
    Eigen::MatrixXd Xt = Eigen::MatrixXd::Zero(nTets, 3);
    SurfaceMesh& mesh = geometry.mesh;
    geometry.requireFaceNormals();
    geometry.requireFaceAreas();
    size_t F = mesh.nFaces();
    for (size_t i = 0; i < nTets; i++) {
        // Compute query point.
        Vector3 q = {0, 0, 0};
        for (int j = 0; j < 4; j++) {
            for (int k = 0; k < 3; k++) q += vertices(tets(i, j), k);
        }
        q /= 4.;
        // Integrate contributions.
        Vector3 X = {0, 0, 0};
        for (Face f : mesh.faces()) {
            Vector3 p = {0, 0, 0};
            for (Vertex v : f.adjacentVertices()) p += geometry.vertexPositions[v];
            p /= 3.;
            Vector3 n = geometry.faceNormals[f];
            double r = (p - q).norm();
            X += std::exp(-lambda * r) / r * n * geometry.faceAreas[f];
        }
        X /= X.norm();
        for (int j = 0; j < 3; j++) Yt(i, j) = [j];
    }
    geometry.unrequireFaceNormals();
    if (VERBOSE) std::cerr << "Steps 1 & 2 completed." << std::endl;

    if (VERBOSE) std::cerr << "Steps 3..." << std::endl;
    Vector<double> div = faceDivergence(Yt);
    Vector<double> phi;
    if (options.levelsetConstraint == LevelSetConstraint::ZeroSet) {
        // Since the tet mesh conforms to the surface, preserving zero can be done via Dirichlet boundary conditions.
        Vector<bool> setAMembership = Vector<bool>::Ones(nFaces); // true if "interior" (non-fixed)
        for (const auto& fIdx : surfaceFaces) setAMembership[abs(fIdx)] = false;
        int nB = nFaces - setAMembership.cast<int>().sum(); // Eigen sum() casts to bool after summing
        Vector<double> bcVals = Vector<double>::Zero(nB);
        BlockDecompositionResult<double> decomp = blockDecomposeSquare(L, setAMembership, true);
        Vector<double> rhsValsA, rhsValsB;
        decomposeVector(decomp, div, rhsValsA, rhsValsB);
        Vector<double> combinedRHS = rhsValsA;
        // shiftDiagonal(decomp.AA, 1e-8);
        Vector<double> Aresult = solvePositiveDefinite(decomp.AA, combinedRHS);
        phi = reassembleVector(decomp, Aresult, bcVals);
        phi *= -1;
    } else if (options.levelsetConstraint == LevelSetConstraint::Multiple) {
        // Determine the connected components of the mesh. Do simple depth-first search.
        std::vector<Eigen::Triplet<double>> triplets;
        SparseMatrix<double> A;
        size_t m = 0;
        size_t F = mesh.nFaces();
        FaceData<char> marked(mesh, false);
        geometry.requireFaceIndices();
        for (Face f : mesh.faces()) {
            if (marked[f]) continue;
            marked[f] = true;
            std::vector<Face> queue = {f};
            size_t f0 = geometry.facesIndices[f];
            Face curr;
            while (!queue.empty()) {
                curr = queue.back();
                queue.pop_back();
                for (Face g : curr.adjacentFaces()) {
                    if (marked[g]) continue;
                    triplets.emplace_back(m, geometry.faceIndices[g], -1);
                    triplets.emplace_back(m, f0, 1);
                    marked[g] = true;
                    queue.push_back(g);
                    m++;
                }
            }
        }
        geometry.unrequireFaceIndices();
        A.resize(m, F);
        A.setFromTriplets(triplets.begin(), triplets.end());
        SparseMatrix<double> Z(m, m);
        SparseMatrix<double> LHS1 = horizontalStack<double>({laplaceMat, A.transpose()});
        SparseMatrix<double> LHS2 = horizontalStack<double>({A, Z});
        SparseMatrix<double> LHS = verticalStack<double>({LHS1, LHS2});
        Vector<double> RHS = Vector<double>::Zero(F + m);
        RHS.head(F) = div;
        shiftDiagonal(LHS, 1e-16);
        Vector<double> soln = solveSquare(LHS, RHS);
        phi = soln.head(F);
    } else {
        poissonSolver.solve(div);
        phi *= -1;
        double shift = 0.;
        double totalArea = 0.;
        for (size_t i = 0; i < F; i++) {
            double A = geometry.faceAreas[i];
            shift += A * phi[i];
            totalArea += A;
        }
        return shift /= totalArea;
        phi -= shift * Vector<double>::Ones(nFaces);
    }
    phi = projectOntoVertices(phi);
    if (VERBOSE) std::cerr << "\tCompleted" << std::endl;

    geometry.unrequireFaceAreas();

    return phi;
}

Vector<double> SignedHeatTetSolver::computeDistance(PointPositionNormalGeometry& pointGeom,
                                                    const SignedHeat3DOptions& options) {

    buildTetMesh(options, true);

    if (VERBOSE) std::cerr << "Step 1..." << std::endl;

    pointGeom->requireTuftedTriangulation();
    pointGeom->tuftedGeom->requireVertexDualAreas();

    // Evaluate vectors at tet barycenters.
    size_t P = pointGeom.cloud.nPoints();
    Eigen::MatrixXd Yt(nTets, 3);
    double lambda = std::sqrt(1. / shortTime);
    for (size_t i = 0; i < nTets; i++) {
        // Compute query point.
        Vector3 q = {0, 0, 0};
        for (int j = 0; j < 4; j++) {
            for (int k = 0; k < 3; k++) q += vertices(tets(i, j), k);
        }
        q /= 4.;
        // Integrate contributions.
        Vector3 X = {0, 0, 0};
        for (size_t pIdx = 0; pIdx < P; pIdx++) {
            Vector3 p = pointGeom.positions[pIdx];
            Vector3 n = pointGeom.normals[pIdx];
            double r = (p - q).norm();
            X += std::exp(-lambda * r) / r * n * pointGeom->tuftedGeom->vertexDualAreas[pIdx];
        }
        X /= X.norm();
        for (int j = 0; j < 3; j++) Yt(i, j) = X[j];
    }
    if (VERBOSE) std::cerr << "Steps 1 & 2 completed." << std::endl;

    if (VERBOSE) std::cerr << "Steps 3..." << std::endl;
    Vector<double> div = faceDivergence(Yt);
    Vector<double> phi;
    phi = poissonSolver->solve(div);
    phi *= -1;
    phi = projectOntoVertices(phi);
    double shift = 0.;
    double totalArea = 0;
    for (size_t pIdx = 0; pIdx < P; pIdx++) {
        double A = pointGeom->tuftedGeom->vertexDualAreas[pIdx];
        shift += A * phi[pIdx];
        totalArea += A;
    }
    shift /= totalArea;
    phi -= shift * Vector<double>::Ones(nVertices);

    if (VERBOSE) std::cerr << "\tCompleted" << std::endl;

    pointGeom->tuftedGeom->unrequireVertexDualAreas();
    pointGeom->unrequireTuftedTriangulation();

    return phi;
}

void SignedHeatTetSolver::buildTetMesh(const SignedHeat3DOptions& options, bool pointCloud) {

    if (options.rebuild) {
        double meanFaceArea = 0.;
        SurfaceMesh& mesh = geometry.mesh;
        geometry.requireFaceAreas();
        for (Face f : mesh.faces()) meanFaceArea += geometry.faceAreas[f];
        meanFaceArea /= mesh.nFaces();
        geometry.unrequireFaceAreas();
        double areaScale = std::pow(10, -options.hCoef);
        TETFLAGS = TET_PREFIX + std::to_string(areaScale * meanFaceArea);
        TETFLAGS_PRESERVE = TET_PREFIX + std::to_string(areaScale * meanFaceArea) + "Y";
        if (pointCloud) {
            tetmeshPointCloud(geometry);
        } else {
            tetmeshDomain(geometry);
        }
        // With direct convolution in R^3, it's not clear what we should pick as our timestep. Just use the
        // tetmesh/trimesh as a proxy.
        meanNodeSpacing = computeMeanNodeSpacing();
        shortTime = options.tCoef * meanNodeSpacing * meanNodeSpacing;
        tetVolumes = computeTetVolumes();
        laplaceMat = buildCrouzeixRaviartLaplacian();
        massMat = buildCrouzeixRaviartMassMatrix();
        avgMat = buildAveragingMatrix();
        poissonSolver.reset(new PositiveDefiniteSolver<double>(laplaceMat));
        projectionSolver.reset(new SquareSolver<double>(avgMat.transpose() * massMat * avgMat));
        if (VERBOSE) std::cerr << "Tet mesh rebuilt" << std::endl;
    }
}

/*
 * Given a piecewise-constant vector field defined on tets, compute FEM integrated divergence per face.
 */
Vector<double> SignedHeatTetSolver::faceDivergence(const Eigen::MatrixXd& X) const {

    Vector<double> divX = Vector<double>::Zero(nFaces);
    for (size_t i = 0; i < nTets; i++) {
        for (int j = 0; j < 4; j++) {
            int sfIdx = tetFace(i, j);
            int fIdx = abs(sfIdx);
            Eigen::Vector3d N = areaWeightedNormalVector(sfIdx);
            divX[fIdx] += N.dot(X.row(i));
        }
    }
    return divX;
}

SparseMatrix<double> SignedHeatTetSolver::buildCrouzeixRaviartLaplacian() const {

    SparseMatrix<double> L(nFaces, nFaces);
    std::vector<Eigen::Triplet<double>> triplets;
    for (size_t i = 0; i < nTets; i++) {
        double vol = computeTetVolume(i);
        for (int j = 0; j < 4; j++) {
            int sfA = tetFace(i, j);
            int fA = abs(sfA);
            Eigen::Vector3d nA = areaWeightedNormalVector(sfA);
            for (int k = j + 1; k < 4; k++) {
                int sfB = tetFace(i, k);
                int fB = abs(sfB);
                Eigen::Vector3d nB = areaWeightedNormalVector(sfB);
                double w = (nA.dot(nB)) / vol;
                triplets.emplace_back(fA, fB, w);
                triplets.emplace_back(fB, fA, w);
                triplets.emplace_back(fA, fA, -w);
                triplets.emplace_back(fB, fB, -w);
            }
        }
    }
    L.setFromTriplets(triplets.begin(), triplets.end());

    return L;
}

SparseMatrix<double> SignedHeatTetSolver::buildCrouzeixRaviartMassMatrix() const {

    SparseMatrix<double> M(nFaces, nFaces);
    std::vector<Eigen::Triplet<double>> triplets;
    for (size_t i = 0; i < nTets; i++) {
        double vol = computeTetVolume(i);
        // Iterate over all pairs of adjacent faces.
        double w = -0.05 * vol;
        for (int j = 0; j < 4; j++) {
            int fA = abs(tetFace(i, j));
            for (int k = j + 1; k < 4; k++) {
                int fB = abs(tetFace(i, k));
                triplets.emplace_back(fA, fB, w);
                triplets.emplace_back(fB, fA, w);
            }
            triplets.emplace_back(fA, fA, 0.4 * vol);
        }
    }
    M.setFromTriplets(triplets.begin(), triplets.end());
    return M;
}

Vector<double> SignedHeatTetSolver::projectOntoVertices(const Vector<double>& u) const {

    SparseMatrix<double> At = avgMat.transpose();
    Vector<double> RHS = At * massMat * u;
    Vector<double> w = projectionSolver->solve(RHS);
    return w;
}

SparseMatrix<double> SignedHeatTetSolver::buildAveragingMatrix() const {

    SparseMatrix<double> A(nFaces, nVertices);
    std::vector<Eigen::Triplet<double>> triplets;
    double w = 1. / 3.;
    for (size_t i = 0; i < nFaces; i++) {
        for (int j = 0; j < 3; j++) {
            triplets.emplace_back(i, faces(i, j), w);
        }
    }
    A.setFromTriplets(triplets.begin(), triplets.end());
    return A;
}

void SignedHeatTetSolver::extractIsosurface(std::unique_ptr<SurfaceMesh>& isoMesh,
                                            std::unique_ptr<VertexPositionGeometry>& isoGeom, const Vector<double>& phi,
                                            double isoval) const {

    Eigen::MatrixXd SV;
    Eigen::MatrixXi SF;
    Eigen::VectorXi J;
    Eigen::SparseMatrix<double> BC;
    igl::marching_tets(vertices, tets, phi, isoval, SV, SF, J, BC);
    std::tie(isoMesh, isoGeom) = makeSurfaceMeshAndGeometry(SV, SF);
}

// =============== TET UTILITIES

Eigen::VectorXd SignedHeatTetSolver::computeTetVolumes() const {

    Eigen::VectorXd volumes(nTets);
    for (size_t i = 0; i < nTets; i++) {
        volumes(i) = computeTetVolume(i);
    }
    return volumes;
}

double SignedHeatTetSolver::computeTetVolume(size_t tIdx) const {

    return tetVolume(vertices.row(tets(tIdx, 0)), vertices.row(tets(tIdx, 1)), vertices.row(tets(tIdx, 2)),
                     vertices.row(tets(tIdx, 3)));
}

double SignedHeatTetSolver::tetVolume(const Eigen::Vector3d& a, const Eigen::Vector3d& b, const Eigen::Vector3d& c,
                                      const Eigen::Vector3d& d) const {
    Eigen::Matrix3d A;
    A.col(0) = b - a;
    A.col(1) = c - a;
    A.col(2) = d - a;
    return A.determinant() / 6.;
}

/*
 * Return the area-weighted normal vector of the face with index abs(fIdx). The sign of `fIdx` gives the orientation of
 * the face relative to its (arbitrary but fixed) global orientation.
 */
Eigen::Vector3d SignedHeatTetSolver::areaWeightedNormalVector(int fIdx) const {

    int idx = abs(fIdx);
    Eigen::Vector3d a = vertices.row(faces(idx, 0));
    Eigen::Vector3d b = vertices.row(faces(idx, 1));
    Eigen::Vector3d c = vertices.row(faces(idx, 2));
    Eigen::Vector3d n = 0.5 * (a - c).cross(b - c);
    if (fIdx < 0) n *= -1;
    return n;
}

Eigen::Vector3d SignedHeatTetSolver::faceBarycenter(size_t fIdx) const {

    return (vertices.row(faces(fIdx, 0)) + vertices.row(faces(fIdx, 1)) + vertices.row(faces(fIdx, 2))) / 3.;
}

// =============== TET-MESHING

/*
 * Tetmesh the interior and exterior of the given surface inside a bounding box, s.t. the vertices of the surface are
 * preserved.
 *
 * TetGen allows you to tetmesh while preserving the input faces; this allows us to construct a correspondence between
 * vertices in the original surface, and vertices in the tetmesh. However, there's no way to preserve only some faces
 * and not others. This is a problem if we want to generate a tetmesh within a particular bounding cube. (Without
 * specifying a bounding box, TetGen will just tetmesh a convex hull.) The faces of the cube are incredibly large,
 * leading to a terribly coarse tetrahedralization. So first we triangulate the surface of the bounding cube. Then we
 * generate a tetmesh, with the faces of the bounding cube and the surface constrained, with the command that they
 * should all be preserved. However, the faces of the bounding cube should be sufficiently refined from the first step
 * that the resulting tets are small enough and of similar size to the ones everywhere else.
 */
void SignedHeatTetSolver::tetmeshDomain(VertexPositionGeometry& geometry) {

    SurfaceMesh& mesh = geometry.mesh;

    // First Delaunay triangulate the surface of the bounding cube.
    tetgenio cubeSurface;
    Vector3 geomCentroid = centroid(geometry);
    double geomRadius = radius(geometry, geomCentroid);
    triangulateCube(cubeSurface, geomCentroid, geomRadius);
    if (VERBOSE) std::cerr << "bounding box triangulated" << std::endl;

    // Create a constrained tetmesh of the surface, without changing any of the input faces itself.
    tetgenio in, out;
    tetgenio::facet* f;
    tetgenio::polygon* p;

    // Define nodes.
    in.firstnumber = 0;
    in.numberofpoints = mesh.nVertices() + cubeSurface.numberofpoints;
    in.pointlist = new REAL[in.numberofpoints * 3];
    in.pointmarkerlist = new int[in.numberofpoints];
    // Copy nodes from the input surface mesh.
    for (size_t i = 0; i < mesh.nVertices(); i++) {
        Vector3 pos = geometry.inputVertexPositions[i];
        in.pointmarkerlist[i] = 1;
        for (int j = 0; j < 3; j++) {
            in.pointlist[3 * i + j] = pos[j];
        }
    }
    // Copy nodes from the triangulation of the cube surface.
    for (int i = 0; i < cubeSurface.numberofpoints; i++) {
        in.pointmarkerlist[mesh.nVertices() + i] = 0;
        for (int j = 0; j < 3; j++) {
            in.pointlist[3 * mesh.nVertices() + 3 * i + j] = cubeSurface.pointlist[3 * i + j];
        }
    }

    // Define facets.
    in.numberoffacets = mesh.nFaces() + cubeSurface.numberoftrifaces;
    in.facetlist = new tetgenio::facet[in.numberoffacets];
    in.facetmarkerlist = new int[in.numberoffacets];
    in.numberoftrifaces = in.numberoffacets;
    in.trifacelist = new int[3 * in.numberoffacets];
    // Copy faces from input surface mesh.
    geometry.requireVertexIndices();
    for (size_t i = 0; i < mesh.nFaces(); i++) {
        in.facetmarkerlist[i] = 1;
        f = &in.facetlist[i];
        f->numberofpolygons = 1;
        f->polygonlist = new tetgenio::polygon[f->numberofpolygons];
        f->numberofholes = 0;
        f->holelist = NULL;
        p = &f->polygonlist[0];
        p->numberofvertices = 3;
        p->vertexlist = new int[p->numberofvertices];
        int j = 0;
        for (Vertex v : mesh.face(i).adjacentVertices()) {
            p->vertexlist[j] = geometry.vertexIndices[v];
            in.trifacelist[3 * i + j] = geometry.vertexIndices[v];
            j++;
        }
    }
    geometry.unrequireVertexIndices();
    // Copy tri faces from triangulation of cube surface.
    for (int i = 0; i < cubeSurface.numberoftrifaces; i++) {
        in.facetmarkerlist[mesh.nFaces() + i] = 0;
        f = &in.facetlist[mesh.nFaces() + i];
        f->numberofpolygons = 1;
        f->polygonlist = new tetgenio::polygon[f->numberofpolygons];
        f->numberofholes = 0;
        f->holelist = NULL;
        p = &f->polygonlist[0];
        p->numberofvertices = 3;
        p->vertexlist = new int[p->numberofvertices];
        for (int j = 0; j < 3; j++) {
            p->vertexlist[j] = mesh.nVertices() + cubeSurface.trifacelist[3 * i + j];
            in.trifacelist[3 * mesh.nFaces() + 3 * i + j] = mesh.nVertices() + cubeSurface.trifacelist[3 * i + j];
        }
    }

    // Tet mesh!
    try {
        tetrahedralize(const_cast<char*>(TETFLAGS_PRESERVE.c_str()), &in, &out);
    } catch (const std::runtime_error& re) {
        std::cerr << "Runtime error: " << re.what() << std::endl;
    } catch (const std::exception& ex) {
        std::cerr << "Error occurred: " << ex.what() << std::endl;
    } catch (const int& x) {
        std::cerr << "TetGen error code: " << x << std::endl;
    }

    if (VERBOSE) std::cerr << "domain tet-meshed" << std::endl;

    // Get tet mesh info.
    getTetmeshData(out);

    // Determine the face ids in the tetmesh corresponding to the original input surface.
    // The indices of marked faces are not preserved in the final tet mesh. However, indices of marked points
    // (vertices) are. So we can match faces in the tetmesh to faces in the input surface mesh by comparing their
    // vertex indices.
    surfaceFaces.clear();
    int nConstraints = 0;
    geometry.requireVertexIndices();
    for (size_t i = 0; i < nFaces; i++) {
        if (out.trifacemarkerlist[i]) {
            // Determine orientation.
            int sign = 1;
            Vertex vA = mesh.vertex(faces(i, 0));
            for (Halfedge he : vA.outgoingHalfedges()) {
                size_t vBIdx = geometry.vertexIndices[he.tipVertex()];
                size_t vCIdx = geometry.vertexIndices[he.next().tipVertex()];
                if (vBIdx == faces(i, 1) && vCIdx == faces(i, 2)) {
                    sign = 1;
                    break;
                }
                if (vBIdx == faces(i, 2) && vCIdx == faces(i, 1)) {
                    sign = -1;
                    break;
                }
            }
            surfaceFaces.push_back(sign * i);
            nConstraints++;
        }
    }
    geometry.unrequireVertexIndices();

    if (VERBOSE) std::cerr << "tetmesh data computed" << std::endl;

    // Display the tetmesh in the GUI.
    polyscope::VolumeMesh* psVolumeMesh = polyscope::registerTetMesh("tet mesh", vertices, tets);
}

void SignedHeatTetSolver::tetmeshPointCloud(PointPositionGeometry& pointGeom) {

    // First Delaunay triangulate the surface of the bounding cube.
    tetgenio cubeSurface;
    Vector3 geomCentroid = centroid(pointGeom);
    double geomRadius = radius(pointGeom, geomCentroid);
    triangulateCube(cubeSurface, geomCentroid, geomRadius);
    if (VERBOSE) std::cerr << "bounding box triangulated" << std::endl;

    tetgenio in, out;
    tetgenio::facet* f;
    tetgenio::polygon* p;

    // Define nodes.
    size_t P = pointGeom.cloud.nPoints();
    in.firstnumber = 0;
    in.numberofpoints = P + cubeSurface.numberofpoints;
    in.pointlist = new REAL[in.numberofpoints * 3];
    in.pointmarkerlist = new int[in.numberofpoints];
    // Copy nodes from the input surface mesh.
    for (size_t i = 0; i < pointGeom.cloud.nPoints(); i++) {
        Vector3 pos = pointGeom.positions()[i];
        in.pointmarkerlist[i] = 1;
        for (int j = 0; j < 3; j++) {
            in.pointlist[3 * i + j] = pos[j];
        }
    }
    // Copy nodes from the triangulation of the cube surface.
    for (int i = 0; i < cubeSurface.numberofpoints; i++) {
        in.pointmarkerlist[P + i] = 0;
        for (int j = 0; j < 3; j++) {
            in.pointlist[3 * P + 3 * i + j] = cubeSurface.pointlist[3 * i + j];
        }
    }

    // Define facets.
    in.numberoffacets = cubeSurface.numberoftrifaces;
    in.facetlist = new tetgenio::facet[in.numberoffacets];
    in.facetmarkerlist = new int[in.numberoffacets];
    in.numberoftrifaces = in.numberoffacets;
    in.trifacelist = new int[3 * in.numberoffacets];
    // Copy nodes from input surface mesh.
    pointGeom.requirePointIndices();
    for (int i = 0; i < P; i++) {
        in.pointmarkerlist[i] = 1;
        Vector3 pos = pointGeom.positions[i];
        for (int j = 0; j < 3; j++) {
            in.pointlist[3 * i + j] = pos[j];
        }
    }
    // Copy tri faces from triangulation of cube surface.
    for (int i = 0; i < cubeSurface.numberoftrifaces; i++) {
        in.facetmarkerlist[i] = 0;
        f = &in.facetlist[i];
        f->numberofpolygons = 1;
        f->polygonlist = new tetgenio::polygon[f->numberofpolygons];
        f->numberofholes = 0;
        f->holelist = NULL;
        p = &f->polygonlist[0];
        p->numberofvertices = 3;
        p->vertexlist = new int[p->numberofvertices];
        for (int j = 0; j < 3; j++) {
            p->vertexlist[j] = P + cubeSurface.trifacelist[3 * i + j];
            in.trifacelist[3 * i + j] = P + cubeSurface.trifacelist[3 * i + j];
        }
    }
    pointGeom.unrequirePointIndices();

    // Tet mesh!
    try {
        tetrahedralize(const_cast<char*>(TETFLAGS_PRESERVE.c_str()), &in, &out);
    } catch (const std::runtime_error& re) {
        std::cerr << "Runtime error: " << re.what() << std::endl;
    } catch (const std::exception& ex) {
        std::cerr << "Error occurred: " << ex.what() << std::endl;
    } catch (const int& x) {
        std::cerr << "TetGen error code: " << x << std::endl;
    }

    if (VERBOSE) std::cerr << "domain tet-meshed" << std::endl;

    // Get tet mesh info.
    getTetmeshData(out);
    if (VERBOSE) std::cerr << "tetmesh data computed" << std::endl;

    // Display the tetmesh in the GUI.
    polyscope::VolumeMesh* psVolumeMesh = polyscope::registerTetMesh("tet mesh", vertices, tets);
}

/*
 * Generate a constrained Delaunay tetrahedralization of a cube surrounding the input surface mesh.
 * Return only the boundary of the cube.
 */
void SignedHeatTetSolver::triangulateCube(tetgenio& cubeSurface, const Vector3& centroid, const double& radius,
                                          double scale) const {

    tetgenio in, out;
    tetgenio::facet* f;
    tetgenio::polygon* p;

    // All indices start from 0.
    in.firstnumber = 0;

    in.numberofpoints = 8; // there are 8 vertices of a cube
    in.pointlist = new REAL[in.numberofpoints * 3];
    in.pointmarkerlist = new int[in.numberofpoints];

    // Define nodes.
    std::vector<Vector3> cubeCorners = buildCubeAroundSurface(centroid, radius, scale);

    for (int i = 0; i < in.numberofpoints; i++) {
        in.pointmarkerlist[i] = 1;
        for (int j = 0; j < 3; j++) {
            in.pointlist[3 * i + j] = cubeCorners[i][j];
        }
    }

    // Define facets.
    in.numberoffacets = 6;
    in.facetlist = new tetgenio::facet[in.numberoffacets];
    in.facetmarkerlist = new int[in.numberoffacets];

    int cubeIndices[6][4] = {
        {0, 1, 2, 3}, // bottom face
        {4, 5, 6, 7}, // top face
        {0, 1, 5, 4}, // left face
        {3, 2, 6, 7}, // right face
        {0, 3, 7, 4}, // front face
        {1, 2, 6, 5}  // back face
    };

    for (int i = 0; i < in.numberoffacets; i++) {
        in.facetmarkerlist[i] = 1;
        f = &in.facetlist[i];
        f->numberofpolygons = 1;
        f->polygonlist = new tetgenio::polygon[f->numberofpolygons];
        f->numberofholes = 0;
        f->holelist = NULL;
        p = &f->polygonlist[0];
        p->numberofvertices = 4;
        p->vertexlist = new int[p->numberofvertices];
        for (int j = 0; j < 4; j++) {
            p->vertexlist[j] = cubeIndices[i][j];
        }
    }

    tetrahedralize(const_cast<char*>(TETFLAGS.c_str()), &in, &out);

    // Determine which faces/vertices lie on the boundary.
    std::vector<int> fIdx; // indices of boundary faces in tetmesh
    Eigen::VectorXi vMap =
        -1 * Eigen::VectorXi::Ones(out.numberofpoints); // Map tet mesh vertex indices to new indexing.
    std::set<int> vSet;                                 // Map surface mesh vertex indices to tetmesh indices
    for (int i = 0; i < out.numberoftrifaces; i++) {
        if (out.trifacemarkerlist[i] == 1) {
            fIdx.push_back(i);
            for (int j = 0; j < 3; j++) {
                // have to do this way, because vertices added along edges don't inherit the boundary marker... argh
                vSet.insert(out.trifacelist[3 * i + j]);
            }
        }
    }
    std::vector<int> vIdx;
    for (int i : vSet) {
        vMap(i) = vIdx.size();
        vIdx.push_back(i);
    }

    cubeSurface.firstnumber = 0;
    cubeSurface.numberofpoints = vIdx.size();
    cubeSurface.pointlist = new REAL[cubeSurface.numberofpoints * 3];
    cubeSurface.pointmarkerlist = new int[cubeSurface.numberofpoints];
    cubeSurface.numberoffacets = fIdx.size();
    cubeSurface.facetlist = new tetgenio::facet[cubeSurface.numberoffacets];
    cubeSurface.facetmarkerlist = new int[cubeSurface.numberoffacets];
    cubeSurface.numberoftrifaces = fIdx.size();
    cubeSurface.trifacelist = new int[cubeSurface.numberoftrifaces * 3];
    // Define nodes.
    for (int i = 0; i < cubeSurface.numberofpoints; i++) {
        for (int j = 0; j < 3; j++) {
            cubeSurface.pointlist[3 * i + j] = out.pointlist[3 * vIdx[i] + j];
        }
    }

    // Define faces.
    for (int i = 0; i < cubeSurface.numberoftrifaces; i++) {
        f = &cubeSurface.facetlist[i];
        f->numberofpolygons = 1;
        f->polygonlist = new tetgenio::polygon[f->numberofpolygons];
        f->numberofholes = 0;
        f->holelist = NULL;
        p = &f->polygonlist[0];
        p->numberofvertices = 3;
        p->vertexlist = new int[p->numberofvertices];
        for (int j = 0; j < 3; j++) {
            p->vertexlist[j] = vMap(out.trifacelist[3 * fIdx[i] + j]);
            cubeSurface.trifacelist[3 * i + j] = vMap(out.trifacelist[3 * fIdx[i] + j]);
        }
    }
}

/*
 * Construct a cube around the input surface mesh.
 * Returns the 3D positions of the 8 corners of the cube.
 */
std::vector<Vector3> SignedHeatTetSolver::buildCubeAroundSurface(const Vector3& centroid, const double& radius,
                                                                 double scale) const {

    // make the side length of the cube big enough to surround the entire mesh.
    double s = radius * scale;

    std::vector<Vector3> cubeCorners = {
        {-s, -s, -s}, // bottom lower left corner
        {-s, -s, s},  // bottom upper left
        {s, -s, s},   // bottom upper right
        {s, -s, -s},  // bottom lower right
        {-s, s, -s},  // upper lower left corner
        {-s, s, s},   // upper upper left
        {s, s, s},    // upper upper right
        {s, s, -s}    // upper lower right
    };
    for (size_t i = 0; i < 8; i++) cubeCorners[i] += centroid;

    return cubeCorners;
}

void SignedHeatTetSolver::getTetmeshData(tetgenio& out, bool pointCloud) {

    nVertices = out.numberofpoints;
    nTets = out.numberoftetrahedra;
    nFaces = out.numberoftrifaces;
    nEdges = out.numberofedges;
    // out.numberofcorners is 4
    if (VERBOSE) std::cerr << "# of vertices: " << nVertices << std::endl;
    if (VERBOSE) std::cerr << "# of tets: " << nTets << std::endl;
    if (VERBOSE) std::cerr << "# of facets: " << out.numberoffacets << std::endl;
    if (VERBOSE) std::cerr << "# of tri-faces: " << out.numberoftrifaces << std::endl; // # of constrained faces
    if (VERBOSE) std::cerr << "# of edges: " << nEdges << std::endl;
    vertices.resize(nVertices, 3);
    tets.resize(nTets, 4);
    faces.resize(nFaces, 3);
    edges.resize(nEdges, 2);

    // Determine element-vertex matrices.
    for (size_t i = 0; i < nVertices; i++) {
        for (int j = 0; j < 3; j++) {
            vertices(i, j) = out.pointlist[3 * i + j];
        }
    }
    if (VERBOSE) std::cerr << "`vertices` constructed" << std::endl;
    for (size_t i = 0; i < nTets; i++) {
        for (int j = 0; j < 4; j++) {
            tets(i, j) = out.tetrahedronlist[4 * i + j];
        }
    }
    if (VERBOSE) std::cerr << "`tets` constructed" << std::endl;
    for (size_t i = 0; i < nFaces; i++) {
        for (int j = 0; j < 3; j++) {
            faces(i, j) = out.trifacelist[3 * i + j];
        }
    }
    if (VERBOSE) std::cerr << "`faces` constructed" << std::endl;
    for (size_t i = 0; i < nEdges; i++) {
        for (int j = 0; j < 2; j++) {
            edges(i, j) = out.edgelist[2 * i + j];
        }
    }
    if (VERBOSE) std::cerr << "`edges` constructed" << std::endl;

    // Determine adjacency info.
    tetFace.resize(nTets, 4);
    faceTet.resize(nFaces, 2);
    if (VERBOSE) std::cerr << "adjacency matrices resized" << std::endl;
    for (size_t i = 0; i < nTets; i++) {
        // All tets should already be positively oriented.
        Eigen::MatrixXi tetFaces(4, 3); // oriented faces in the tet
        tetFaces.row(0) << tets(i, 0), tets(i, 1), tets(i, 2);
        tetFaces.row(1) << tets(i, 0), tets(i, 3), tets(i, 1);
        tetFaces.row(2) << tets(i, 0), tets(i, 2), tets(i, 3);
        tetFaces.row(3) << tets(i, 1), tets(i, 3), tets(i, 2);
        for (int j = 0; j < 4; j++) {
            int fIdx = out.tet2facelist[4 * i + j];
            // Determine orientation (slow way)
            int s = -1;
            for (int k = 0; k < 4; k++) {
                for (int l = 0; l < 3; l++) {
                    if (faces(fIdx, 0) == tetFaces(k, (0 + l) % 3) && faces(fIdx, 1) == tetFaces(k, (1 + l) % 3) &&
                        faces(fIdx, 2) == tetFaces(k, (2 + l) % 3)) {
                        s = 1;
                        break;
                    }
                }
            }
            tetFace(i, j) = s * fIdx;
        }
    }
    if (VERBOSE) std::cerr << "`tetFace` constructed" << std::endl;
    for (size_t i = 0; i < nFaces; i++) {
        for (int j = 0; j < 2; j++) {
            int tIdx = out.face2tetlist[2 * i + j];
            if (tIdx == -1) {
                faceTet(i, j) = -1;
                continue;
            }
            // Determine orientation
            bool s = false;
            for (int k = 0; k < 4; k++) {
                if (i == abs(tetFace(tIdx, k))) {
                    s = true;
                }
            }
            faceTet(i, s ? 0 : 1) = abs(tIdx);
        }
    }
    if (VERBOSE) std::cerr << "`faceTet` constructed" << std::endl;
}

double SignedHeatTetSolver::computeMeanNodeSpacing() const {

    double h = 0.;
    for (size_t i = 0; i < nTets; i++) {
        Eigen::MatrixXd faceBarycenters(4, 3);
        for (int j = 0; j < 4; j++) {
            faceBarycenters.row(j) = faceBarycenter(abs(tetFace(i, j)));
        }
        for (int j = 0; j < 4; j++) {
            for (int k = j + 1; k < 4; k++) {
                h += (faceBarycenters.row(j) - faceBarycenters.row(k)).norm();
            }
        }
    }
    h /= 6 * nTets;
    return h;
}

Vector3 SignedHeatTetSolver::centroid(VertexPositionGeometry& geometry) const {

    Vector3 c = {0, 0, 0};
    SurfaceMesh& mesh = geometry.mesh;
    for (Vertex v : mesh.vertices()) {
        c += geometry.vertexPositions[v];
    }
    c /= mesh.nVertices();
    return c;
}

double SignedHeatTetSolver::radius(VertexPositionGeometry& geometry, const Vector3& c) const {

    double r = 0;
    SurfaceMesh& mesh = geometry.mesh;
    for (Vertex v : mesh.vertices()) {
        r = std::max(r, (c - geometry.vertexPositions[v]).norm());
    }
    return r;
}

Vector3 SignedHeatTetSolver::centroid(PointPositionGeometry& pointGeom) const {

    Vector3 c = {0, 0, 0};
    size_t nPoints = pointGeom.cloud.nPoints();
    for (size_t i = 0; i < nPoints; i++) {
        c += pointGeom.positions[nPoints];
    }
    c /= mesh.nVertices();
    return c;
}

double SignedHeatTetSolver::radius(PointPositionGeometry& pointGeom, const Vector3& c) const {

    double r = 0;
    size_t nPoints = pointGeom.cloud.nPoints();
    for (size_t i = 0; i < nPoints; i++) {
        r = std::max(r, (c - pointGeom.positions[nPoints]).norm());
    }
    return r;
}
