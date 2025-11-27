#pragma once

#include "Vec3.h"
#include <algorithm>

struct AABB {
    Vec3 min;
    Vec3 max;

    AABB() : min(0, 0, 0), max(0, 0, 0) {}
    AABB(const Vec3& min, const Vec3& max) : min(min), max(max) {}

    bool Intersects(const AABB& other) const {
        return (min.x <= other.max.x && max.x >= other.min.x) &&
               (min.y <= other.max.y && max.y >= other.min.y) &&
               (min.z <= other.max.z && max.z >= other.min.z);
    }
};
