#pragma once

#include "geometrycentral/numerical/linear_solvers.h"
#include "geometrycentral/pointcloud/point_position_normal_geometry.h"
#include "geometrycentral/surface/signed_heat_method.h"
#include "geometrycentral/surface/surface_mesh.h"
#include "geometrycentral/surface/vertex_position_geometry.h"

#include "polyscope/volume_mesh.h"

#include "signed_heat_3d.h"

#define TETLIBRARY
#include "tetgen.h"

using namespace geometrycentral;
using namespace geometrycentral::surface;

Vector<double> computeDistanceOnGrid(VertexPositionGeometry& geometry, const SignedHeat3DOptions& options);
Vector<double> computeDistanceOnGrid(PointPositionNormalGeometry& pointGeom, const SignedHeat3DOptions& options);