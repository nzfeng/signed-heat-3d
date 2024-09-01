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

using namespace geometrycentral;
using namespace geometrycentral::surface;

struct SignedHeat3DOptions {
    LevelSetConstraint levelSetConstraint = LevelSetConstraint::ZeroSet;
    double tCoef = 1.0;
    double hCoef = 0.0;
    bool rebuild = true;
    double scale = 2.;
    bool fastIntegration = false;
};

Vector3 centroid(VertexPositionGeometry& geometry);
Vector3 centroid(pointcloud::PointPositionGeometry& pointGeom);
double radius(VertexPositionGeometry& geometry, const Vector3& centroid);
double radius(pointcloud::PointPositionGeometry& pointGeom, const Vector3& c);
double yukawaPotential(const Vector3& x, const Vector3& y, const double& shortTime);
double meanEdgeLength(IntrinsicGeometryInterface& geom);