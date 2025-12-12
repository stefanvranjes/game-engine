#pragma once

#include "Vec3.h"

struct Ray {
    Vec3 origin;
    Vec3 direction;

    Ray(const Vec3& o, const Vec3& d) : origin(o), direction(d.Normalized()) {}

    Vec3 GetPoint(float distance) const {
        return origin + direction * distance;
    }
};

struct RaycastHit {
    bool hit;
    float distance;
    Vec3 point;
    Vec3 normal;
    class GameObject* object; // Forward declared, fine if only ptr used. But need GameObject defined for Raycast method.
                              // Ray.h is included in GameObject.cpp, so it sees Ray.
                              // Application.cpp sees GameObject.h (which uses Ray).
                              // Circular dep check? GameObject.h needs Ray (so include it). Ray needs GameObject ptr (forward decl).
                              // This file looks correct.
    RaycastHit() : hit(false), distance(-1.0f), object(nullptr) {}
};
