#pragma once

#include "Math/Vec2.h"
#include "Math/Vec3.h"

struct Vertex {
    Vec3 position;
    Vec3 normal;
    Vec2 texCoord;
    
    Vertex() : position(0, 0, 0), normal(0, 1, 0), texCoord(0, 0) {}
};
