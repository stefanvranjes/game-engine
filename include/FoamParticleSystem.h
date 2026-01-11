#pragma once

#include "Math/Vec3.h"
#include <vector>

struct FluidParticle;
struct FluidType;
class PhysXBackend;

/**
 * @brief Foam/spray particle for secondary fluid effects
 * 
 * Lightweight particles spawned from high-velocity fluid regions
 */
struct FoamParticle {
    Vec3 position;
    Vec3 velocity;
    float lifetime;        // Remaining lifetime
    float maxLifetime;     // Initial lifetime
    float size;            // Particle size
    int textureIndex;      // Texture atlas index (0-3)
    float animationTime;   // Animation time (0-1, loops)
    
    enum class Type {
        Foam,    // Floating on surface
        Spray,   // Airborne droplets
        Bubble   // Rising bubbles
    };
    Type type;
    
    bool active;
    
    FoamParticle()
        : position(0, 0, 0)
        , velocity(0, 0, 0)
        , lifetime(1.0f)
        , maxLifetime(1.0f)
        , size(0.02f)
        , textureIndex(0)
        , animationTime(0.0f)
        , type(Type::Spray)
        , active(true)
    {}
    
    FoamParticle(const Vec3& pos, const Vec3& vel, Type particleType, float life)
        : position(pos)
        , velocity(vel)
        , lifetime(life)
        , maxLifetime(life)
        , size(0.02f)
        , textureIndex(0)
        , animationTime(0.0f)
        , type(particleType)
        , active(true)
    {}
    
    float GetLifeRatio() const {
        return lifetime / maxLifetime;
    }
};

/**
 * @brief Foam particle system for secondary fluid effects
 * 
 * Generates and simulates foam, spray, and bubble particles
 * based on fluid velocity and surface properties
 */
class FoamParticleSystem {
public:
    FoamParticleSystem();
    ~FoamParticleSystem();
    
    /**
     * @brief Update foam particles
     * @param deltaTime Time step
     * @param gravity Gravity vector
     */
    /**
     * @brief Update foam particles
     * @param deltaTime Time step
     * @param gravity Gravity vector
     * @param backend Physics backend for collision (optional)
     */
    void Update(float deltaTime, const Vec3& gravity, PhysXBackend* backend = nullptr);
    
    /**
     * @brief Generate foam particles from fluid particles
     * @param fluidParticles Source fluid particles
     * @param fluidTypes Fluid types (for rest density)
     * @param velocityThreshold Minimum velocity to spawn foam
     * @param spawnRate Particles to spawn per high-velocity particle
     */
    void GenerateFromFluid(const std::vector<struct FluidParticle>& fluidParticles,
                          const std::vector<struct FluidType>& fluidTypes,
                          float velocityThreshold,
                          float spawnRate);
    
    /**
     * @brief Clear all foam particles
     */
    void Clear();
    
    /**
     * @brief Get all foam particles
     */
    const std::vector<FoamParticle>& GetParticles() const { return m_Particles; }
    std::vector<FoamParticle>& GetParticles() { return m_Particles; }
    
    /**
     * @brief Set parameters
     */
    void SetMaxParticles(int maxParticles) { m_MaxParticles = maxParticles; }
    void SetDragCoefficient(float drag) { m_DragCoefficient = drag; }
    void SetBuoyancy(float buoyancy) { m_Buoyancy = buoyancy; }
    void SetFoamLifetime(float lifetime) { m_FoamLifetime = lifetime; }
    void SetSprayLifetime(float lifetime) { m_SprayLifetime = lifetime; }
    void SetBubbleLifetime(float lifetime) { m_BubbleLifetime = lifetime; }
    void SetAnimationSpeed(float speed) { m_AnimationSpeed = speed; }
    void SetSyncAnimationToLifetime(bool sync) { m_SyncAnimationToLifetime = sync; }
    void SetMergeRadius(float radius) { m_MergeRadius = radius; }
    void SetSurfaceAdhesion(float strength) { m_SurfaceAdhesion = strength; }
    
    /**
     * @brief Merge nearby particles to optimize performance
     * @param radius Merge radius
     */
    void MergeParticles(float radius);
    
    /**
     * @brief Get parameters
     */
    int GetMaxParticles() const { return m_MaxParticles; }
    int GetActiveParticleCount() const;
    
private:
    std::vector<FoamParticle> m_Particles;
    
    // Parameters
    int m_MaxParticles;
    float m_DragCoefficient;
    float m_Buoyancy;
    float m_FoamLifetime;
    float m_SprayLifetime;
    float m_BubbleLifetime;
    float m_AnimationSpeed;
    bool m_SyncAnimationToLifetime;
    float m_MergeRadius;
    float m_SurfaceAdhesion;
    
    // Spawn accumulator for fractional spawning
    float m_SpawnAccumulator;
    
    // Helper methods
    void UpdateParticle(FoamParticle& particle, float deltaTime, const Vec3& gravity);
    void RemoveDeadParticles();
    void EnforceParticleLimit();
    FoamParticle::Type DetermineParticleType(const Vec3& velocity);
};
