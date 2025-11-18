#pragma once

#include "geometrycentral/pointcloud/point_position_normal_geometry.h"
#include "geometrycentral/surface/intrinsic_geometry_interface.h"
#include "geometrycentral/surface/signed_heat_method.h"
#include "geometrycentral/surface/surface_mesh.h"
#include "geometrycentral/surface/vertex_position_geometry.h"

#include <queue>

#include <chrono>
using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

#ifndef SHM_NO_AMGCL
#include <amgcl/adapter/eigen.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/backend/eigen.hpp>
#include <amgcl/coarsening/runtime.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/make_solver.hpp>
#include <amgcl/preconditioner/cpr.hpp>
#include <amgcl/profiler.hpp>
#include <amgcl/relaxation/as_preconditioner.hpp>
#include <amgcl/relaxation/runtime.hpp>
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/solver/bicgstab.hpp>
#include <amgcl/solver/cg.hpp>
#include <amgcl/solver/runtime.hpp>
#endif

using namespace geometrycentral;
using namespace geometrycentral::surface;

struct SignedHeat3DOptions {
    LevelSetConstraint levelSetConstraint = LevelSetConstraint::ZeroSet;
    double tCoef = 1.0;
    Vector3 bboxMin = {1., 1., 1.};
    Vector3 bboxMax = {-1., -1., -1.};
    std::array<size_t, 3> resolution = {0, 0, 0};
    bool useCrouzeixRaviart = true;
    bool fastIntegration = false;
    bool exportData = false;
    std::string meshname;

    bool operator==(const SignedHeat3DOptions& other) const {
        if (bboxMin != other.bboxMin) return false;
        if (bboxMax != other.bboxMax) return false;
        if (resolution != other.resolution) return false;
        return true;
    }

    bool operator!=(const SignedHeat3DOptions& other) const {
        return !(*this == other);
    }
};
// bool operator==(const SignedHeat3DOptions& A, const SignedHeat3DOptions& B) {
//     if (A.bboxMin != B.bboxMin) return false;
//     if (A.bboxMax != B.bboxMax) return false;
//     if (A.resolution != B.resolution) return false;
//     return true;
// }

Vector3 centroid(VertexPositionGeometry& geometry);
Vector3 centroid(pointcloud::PointPositionGeometry& pointGeom);
double radius(VertexPositionGeometry& geometry, const Vector3& centroid);
double radius(pointcloud::PointPositionGeometry& pointGeom, const Vector3& c);
double yukawaPotential(const Vector3& x, const Vector3& y, const double& shortTime);
double meanEdgeLength(IntrinsicGeometryInterface& geom);
void setFaceVectorAreas(VertexPositionGeometry& geometry, FaceData<double>& areas, FaceData<Vector3>& normals);

bool isBoundingBoxValid(const Vector3& bboxMin, const Vector3& bboxMax);
bool isResolutionValid(const std::array<size_t, 3>& resolution);
std::pair<Vector3, Vector3> computeBBox(VertexPositionGeometry& geometry);
std::pair<Vector3, Vector3> computeBBox(pointcloud::PointPositionNormalGeometry& pointGeom);

Vector<double> solveSquareSystem(SparseMatrix<double>& LHS, const Vector<double>& RHS, bool verbose = false);
Vector<double> solvePositiveDefiniteSystem(SparseMatrix<double>& LHS, const Vector<double>& RHS, bool verbose = false);
Vector<double> solvePositiveDefiniteSystem(SparseMatrix<double>& LHS, const Vector<double>& RHS,
                                           std::unique_ptr<PositiveDefiniteSolver<double>>& solver, bool factorize,
                                           bool verbose = false);
Vector<double> AMGCL_solve(SparseMatrix<double>& LHS, const Vector<double>& RHS, bool& success, bool verbose = false);
Vector<double> AMGCL_blockSolve(const SparseMatrix<double>& L, const SparseMatrix<double>& A,
                                const SparseMatrix<double>& Z, const Vector<double>& rhs, bool verbose = false);