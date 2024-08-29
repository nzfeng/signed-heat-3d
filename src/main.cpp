#include "geometrycentral/pointcloud/point_cloud_"
#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/meshio.h"
#include "geometrycentral/surface/surface_mesh_factories.h"
#include "geometrycentral/surface/vertex_position_geometry.h"

#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"
#include "polyscope/volume_mesh.h"

#include "signed_heat_grid_solver.h"
#include "signed_heat_tet_solver.h"

#include "args/args.hxx"
#include "imgui.h"

#include <chrono>
using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

using namespace geometrycentral;
using namespace geometrycentral::surface;

// == Geometry-central data
std::unique_ptr<SurfaceMesh> mesh;
std::unique_ptr<VertexPositionGeometry> geometry;

// Contouring
float ISOVAL = 0.;
Vector<double> PHI;
std::unique_ptr<SurfaceMesh> isoMesh;
std::unique_ptr<VertexPositionGeometry> isoGeom;

// Polyscope data
polyscope::SurfaceMesh* psMesh;

// Tet mesh data
Eigen::MatrixXd VERTICES;
Eigen::MatrixXi TETS, FACES, EDGES;

// Solvers & parameters
float TCOEF = 1.0;
float HCOEF = 1.0;
std::unique_ptr<SignedHeatTetSolver> tetSolver;
std::unique_ptr<SignedHeatGridSolver> gridSolver;
SignedHeat3DOptions SHM_OPTIONS;
int CONSTRAINT_MODE = static_cast<int>(LevelSetConstraint::ZeroSet);

// Program variables
enum MeshMode { Tet = 0, Grid };
int MESH_MODE = MeshMode::Tet;
std::string MESHNAME = "input mesh";
std::string OUTPUT_DIR = "../export";
std::string OUTPUT_FILENAME;

void solve() {
    if (MESH_MODE == MeshMode::Tet) {
        SHM_OPTIONS.levelSetConstraint = CONSTRAINT_MODE;
        SHM_OPTIONS.tCoef = TCOEF;
        SHM_OPTIONS.hCoef = HCOEF;
        PHI = tetSolver->computeDistance(*geometry, OPTIONS);
        psVolumeMesh->addVertexScalarQuantity("GSD", PHI)->setIsolinesEnabled(true)->setEnabled(true);
        SHM_OPTIONS.rebuild = false;
    } else if (MeshMode::Tet == MeshMode::Grid) {
    }

    ImGui::RadioButton("Constrain zero set", &CONSTRAINT_MODE, static_cast<int>(LevelSetConstraint::ZeroSet));
    ImGui::RadioButton("Constrain multiple levelsets", &CONSTRAINT_MODE,
                       static_cast<int>(LevelSetConstraint::Multiple));
    ImGui::RadioButton("No levelset constraints", &CONSTRAINT_MODE, static_cast<int>(LevelSetConstraint::None));
}

void callback() {

    if (ImGui::Button("Solve")) {
        solve();
    }

    ImGui::InputFloat("tCoef", &TCOEF);
    if (ImGui::InputFloat("hCoef (mesh spacing)", &HCOEF)) {
        SHM_OPTIONS.rebuild = true;
    }

    // Contouring
    if (ImGui::InputFloat("Isovalue (enter value)", &ISOVAL)) {
        GSD3->extractIsosurface(isoMesh, isoGeom, PHI, ISOVAL);
        polyscope::registerSurfaceMesh("isosurface", isoGeom->vertexPositions, isoMesh->getFaceVertexList());
    }
    if (ImGui::SliderFloat("Contour (drag slider)", &ISOVAL, PHI.minCoeff(), PHI.maxCoeff())) {
        GSD3->extractIsosurface(isoMesh, isoGeom, PHI, ISOVAL);
        polyscope::registerSurfaceMesh("isosurface", isoGeom->vertexPositions, isoMesh->getFaceVertexList());
    }
    if (ImGui::Button("Export isosurface")) {
        writeSurfaceMesh(*isoMesh, *isoGeom, OUTPUT_DIR + "/isosurface.obj");
    }
}

int main(int argc, char** argv) {

    // Configure the argument parser
    args::ArgumentParser parser("Solve for generalized signed distance (3D domains).");
    args::HelpFlag help(parser, "help", "Display this help menu", {"help"});
    args::Positional<std::string> meshFilename(parser, "mesh", "A mesh or point cloud file.");

    args::Group group(parser);
    args::Flag points(group, "grid", "Solve on a background grid (vs. tet mesh).", {"g", "grid"});
    args::Flag verbose(group, "verbose", "Verbose output", {"V", "verbose"});

    // Parse args
    try {
        parser.ParseCLI(argc, argv);
    } catch (args::Help&) {
        std::cout << parser;
        return 0;
    } catch (args::ParseError& e) {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        return 1;
    }

    if (!meshFilename) {
        std::cerr << "Please specify a mesh file as argument." << std::endl;
        return EXIT_FAILURE;
    }

    // Load mesh
    std::string meshFilepath = args::get(meshFilename);
    DATA_DIR = getHomeDirectory(meshFilepath);
    MESH_MODE = grid ? MeshMode::Grid : MeshMode::Tet;
    OUTPUT_FILENAME = OUTPUT_DIR + "/GSD.obj";
    VERBOSE = verbose;

    // TODO: Read in point cloud (possibly with normals)
    std::tie(mesh, geometry) = readSurfaceMesh(meshFilepath);
    if (!mesh->isTriangular()) {
        std::cerr << "Input mesh is non-triangular, reverting to grid mode." << std::endl;
        MESH_MODE = MeshMode::Grid;
    }
    tetSolver = std::unique_ptr<SignedHeatTetSolver>(new SignedHeatTetSolver());
    gridSolver = std::unique_ptr<SignedHeatGridSolver>(new SignedHeatGridSolver());

    // Visualize data.
    polyscope::init();
    polyscope::state::userCallback = callback;
    psMesh = polyscope::registerSurfaceMesh(MESHNAME, geometry->vertexPositions, mesh->getFaceVertexList());
    if (mesh->isTriangular()) psMesh->setAllPermutations(polyscopePermutations(*mesh));
    polyscope::show();

    return EXIT_SUCCESS;
}