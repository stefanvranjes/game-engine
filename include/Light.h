#pragma once

#include "Math/Vec3.h"

struct Light {
    Vec3 position;
    Vec3 color;
    float intensity;
    float radius; // For future attenuation
    bool castsShadows;

    Light(Vec3 pos = Vec3(0,0,0), Vec3 col = Vec3(1,1,1), float inten = 1.0f, bool shadows = false)
        : position(pos), color(col), intensity(inten), radius(10.0f), castsShadows(shadows) {}
};
