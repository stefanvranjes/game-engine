#pragma once

#include "Math/Vec3.h"
#include "Math/Vec4.h"
#include <memory>

// Forward declaration
class ParticleTrail;

struct Particle {
    Vec3 position;
    Vec3 velocity;
    Vec4 color;
    float size;
    float lifetime;  // Total lifetime in seconds
    float age;       // Current age in seconds
    float lifeRatio; // 0.0 to 1.0
    bool active;
    
    // Collision properties
    float mass;         // Particle mass for collision response
    float restitution;  // Bounciness (0.0 = no bounce, 1.0 = perfect bounce)
    float friction;     // Surface friction coefficient
    
    // Trail properties
    std::unique_ptr<ParticleTrail> trail;  // Optional trail data
    bool hasTrail;                         // Whether this particle has a trail
    
    Particle()
        : position(0, 0, 0)
        , velocity(0, 0, 0)
        , color(1, 1, 1, 1)
        , size(1.0f)
        , lifetime(1.0f)
        , age(0.0f)
        , lifeRatio(0.0f)
        , active(false)
        , mass(1.0f)
        , restitution(0.5f)
        , friction(0.1f)
        , trail(nullptr)
        , hasTrail(false)
    {}
};
