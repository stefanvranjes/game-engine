#pragma once

#include "Math/Vec3.h"
#include "FluidParticle.h"
#include <vector>

/**
 * @brief Fluid particle emitter
 * 
 * Emits fluid particles at a specified rate with configurable properties
 */
class FluidEmitter {
public:
    enum class EmissionShape {
        Point,      // Emit from a single point
        Box,        // Emit from within a box volume
        Sphere,     // Emit from within a sphere volume
        Disc,       // Emit from a disc surface
        Mesh        // Emit from mesh surface (future)
    };
    
    FluidEmitter();
    ~FluidEmitter();
    
    /**
     * @brief Update emitter and emit particles
     * @param deltaTime Time step
     * @param outParticles Output vector to add new particles to
     */
    void Update(float deltaTime, std::vector<FluidParticle>& outParticles);
    
    /**
     * @brief Set emitter properties
     */
    void SetPosition(const Vec3& pos) { m_Position = pos; }
    void SetVelocity(const Vec3& vel) { m_Velocity = vel; }
    void SetEmissionRate(float particlesPerSecond) { m_EmissionRate = particlesPerSecond; }
    void SetFluidType(int type) { m_FluidType = type; }
    void SetEnabled(bool enabled) { m_Enabled = enabled; }
    void SetEmissionShape(EmissionShape shape) { m_EmissionShape = shape; }
    void SetBoxExtents(const Vec3& extents) { m_BoxExtents = extents; }
    void SetSphereRadius(float radius) { m_SphereRadius = radius; }
    void SetDiscRadius(float radius) { m_DiscRadius = radius; }
    void SetVelocitySpread(float spread) { m_VelocitySpread = spread; }
    void SetLifetime(float lifetime) { m_ParticleLifetime = lifetime; }
    
    /**
     * @brief Get emitter properties
     */
    Vec3 GetPosition() const { return m_Position; }
    Vec3 GetVelocity() const { return m_Velocity; }
    float GetEmissionRate() const { return m_EmissionRate; }
    int GetFluidType() const { return m_FluidType; }
    bool IsEnabled() const { return m_Enabled; }
    EmissionShape GetEmissionShape() const { return m_EmissionShape; }
    
    /**
     * @brief Get statistics
     */
    int GetTotalEmittedParticles() const { return m_TotalEmitted; }
    void ResetStatistics() { m_TotalEmitted = 0; }

private:
    // Emitter properties
    Vec3 m_Position;
    Vec3 m_Velocity;
    float m_EmissionRate;           // Particles per second
    int m_FluidType;
    bool m_Enabled;
    
    // Emission shape
    EmissionShape m_EmissionShape;
    Vec3 m_BoxExtents;              // For box emission
    float m_SphereRadius;           // For sphere emission
    float m_DiscRadius;             // For disc emission
    Vec3 m_DiscNormal;              // For disc emission
    
    // Particle properties
    float m_VelocitySpread;         // Random velocity variation
    float m_ParticleLifetime;       // -1 for infinite
    
    // Internal state
    float m_AccumulatedTime;
    int m_TotalEmitted;
    
    // Helper methods
    Vec3 GetRandomPositionInShape() const;
    Vec3 GetRandomVelocity() const;
    float RandomFloat(float min, float max) const;
};
