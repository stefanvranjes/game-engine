#pragma once

#include "Math/Vec3.h"
#include <vector>

/**
 * @brief Fluid particle data structure for PBD simulation
 * 
 * Contains all per-particle data needed for Position-Based Dynamics fluid simulation
 */
struct FluidParticle {
    // Kinematic state
    Vec3 position;           // Current position
    Vec3 velocity;           // Current velocity
    Vec3 predictedPosition;  // Predicted position (before constraint solving)
    Vec3 deltaPosition;      // Position correction from constraints
    
    // Dynamic properties
    float mass;              // Particle mass
    float density;           // Current density
    float pressure;          // Current pressure
    float lambda;            // Lagrange multiplier for density constraint
    
    // Force accumulators
    Vec3 force;              // External forces (gravity, user forces)
    
    // Neighbor data
    std::vector<int> neighbors;  // Indices of neighboring particles
    
    // Fluid type and appearance
    int fluidType;           // Fluid type ID (for multi-fluid support)
    Vec3 color;              // Particle color (for visualization and mixing)
    
    // Lifetime and state
    float age;               // Time since particle creation
    float lifetime;          // Maximum lifetime (-1 for infinite)
    bool active;             // Is particle active?
    
    // Constructor
    FluidParticle() 
        : position(0, 0, 0)
        , velocity(0, 0, 0)
        , predictedPosition(0, 0, 0)
        , deltaPosition(0, 0, 0)
        , mass(1.0f)
        , density(0.0f)
        , pressure(0.0f)
        , lambda(0.0f)
        , force(0, 0, 0)
        , fluidType(0)
        , color(0.2f, 0.5f, 1.0f)  // Default blue water color
        , age(0.0f)
        , lifetime(-1.0f)
        , active(true)
    {}
    
    FluidParticle(const Vec3& pos, const Vec3& vel, int type = 0)
        : position(pos)
        , velocity(vel)
        , predictedPosition(pos)
        , deltaPosition(0, 0, 0)
        , mass(1.0f)
        , density(0.0f)
        , pressure(0.0f)
        , lambda(0.0f)
        , force(0, 0, 0)
        , fluidType(type)
        , color(0.2f, 0.5f, 1.0f)
        , age(0.0f)
        , lifetime(-1.0f)
        , active(true)
    {}
};

/**
 * @brief GPU-compatible particle data (for CUDA)
 * Simplified structure with only essential data for GPU kernels
 */
struct FluidParticleGPU {
    float position[3];
    float velocity[3];
    float predictedPosition[3];
    float deltaPosition[3];
    float mass;
    float density;
    float pressure;
    float lambda;
    int fluidType;
    
    // Convert from CPU particle
    void FromCPU(const FluidParticle& p) {
        position[0] = p.position.x;
        position[1] = p.position.y;
        position[2] = p.position.z;
        velocity[0] = p.velocity.x;
        velocity[1] = p.velocity.y;
        velocity[2] = p.velocity.z;
        predictedPosition[0] = p.predictedPosition.x;
        predictedPosition[1] = p.predictedPosition.y;
        predictedPosition[2] = p.predictedPosition.z;
        deltaPosition[0] = p.deltaPosition.x;
        deltaPosition[1] = p.deltaPosition.y;
        deltaPosition[2] = p.deltaPosition.z;
        mass = p.mass;
        density = p.density;
        pressure = p.pressure;
        lambda = p.lambda;
        fluidType = p.fluidType;
    }
    
    // Convert to CPU particle
    void ToCPU(FluidParticle& p) const {
        p.position = Vec3(position[0], position[1], position[2]);
        p.velocity = Vec3(velocity[0], velocity[1], velocity[2]);
        p.predictedPosition = Vec3(predictedPosition[0], predictedPosition[1], predictedPosition[2]);
        p.deltaPosition = Vec3(deltaPosition[0], deltaPosition[1], deltaPosition[2]);
        p.mass = mass;
        p.density = density;
        p.pressure = pressure;
        p.lambda = lambda;
        p.fluidType = fluidType;
    }
};
