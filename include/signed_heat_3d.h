#pragma once

#include "geometrycentral/surface/signed_heat_method.h"

struct SignedHeat3DOptions {
    LevelSetConstraint levelSetConstraint = LevelSetConstraint::ZeroSet;
    double tCoef = 1.0;
    double hCoef = 1.0;
    bool rebuild = false;
};