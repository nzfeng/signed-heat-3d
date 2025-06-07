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
    double hCoef = 0.0;
    bool rebuild = true;
    double scale = 2.;
    bool useCrouzeixRaviart = true;
    bool fastIntegration = false;
    bool exportData = false;
    std::string meshname;
};

Vector3 centroid(VertexPositionGeometry& geometry);
Vector3 centroid(pointcloud::PointPositionGeometry& pointGeom);
double radius(VertexPositionGeometry& geometry, const Vector3& centroid);
double radius(pointcloud::PointPositionGeometry& pointGeom, const Vector3& c);
double yukawaPotential(const Vector3& x, const Vector3& y, const double& shortTime);
double meanEdgeLength(IntrinsicGeometryInterface& geom);
void setFaceVectorAreas(VertexPositionGeometry& geometry, FaceData<double>& areas, FaceData<Vector3>& normals);

Vector<double> AMGCL_solve(const SparseMatrix<double>& LHS, const Vector<double>& RHS, bool verbose = false);
Vector<double> AMGCL_blockSolve(const SparseMatrix<double>& L, const SparseMatrix<double>& A,
                                const SparseMatrix<double>& Z, const Vector<double>& rhs, bool verbose = false);