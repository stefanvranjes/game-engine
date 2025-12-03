#pragma once

#include "Math/Vec3.h"
#include "Math/Vec4.h"

struct Particle {
    Vec3 position;
    Vec3 velocity;
    Vec4 color;
    float size;
    float lifetime;  // Total lifetime in seconds
    float age;       // Current age in seconds
    bool active;
    
    Particle()
        : position(0, 0, 0)
        , velocity(0, 0, 0)
        , color(1, 1, 1, 1)
        , size(1.0f)
        , lifetime(1.0f)
        , age(0.0f)
        , active(false)
    {}
};
