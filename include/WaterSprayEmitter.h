#pragma once

#include "ParticleEmitter.h"
#include "Math/Vec3.h"
#include "Math/Vec2.h"
#include <memory>
#include <random>

class Water;
class GameObject;
class Texture;

/**
 * @brief Specialized particle emitter for water spray and foam effects.
 * 
 * This emitter samples FFT displacement data to spawn particles at wave crests,
 * creating realistic spray effects that follow wave motion.
 */
class WaterSprayEmitter {
public:
    WaterSprayEmitter();
    ~WaterSprayEmitter();

    /**
     * @brief Initialize the spray emitter with a water component.
     * @param waterObject The game object containing the Water component.
     * @param maxParticles Maximum number of spray particles.
     * @return true if initialization successful.
     */
    bool Init(GameObject* waterObject, int maxParticles = 500);

    /**
     * @brief Update spray particles based on wave state.
     * @param deltaTime Time since last frame.
     * @param cameraPos Camera position for LOD and sorting.
     */
    void Update(float deltaTime, const Vec3& cameraPos);

    /**
     * @brief Get the underlying particle emitter for rendering.
     */
    ParticleEmitter* GetEmitter() { return m_Emitter.get(); }

    /**
     * @brief Set the spray texture.
     */
    void SetTexture(std::shared_ptr<Texture> texture);

    /**
     * @brief Enable or disable the spray effect.
     */
    void SetEnabled(bool enabled) { m_Enabled = enabled; }
    bool IsEnabled() const { return m_Enabled; }

    // Configuration
    void SetSpawnAreaSize(float size) { m_SpawnAreaSize = size; }
    void SetSampleResolution(int samples) { m_SampleResolution = samples; }

private:
    /**
     * @brief Sample a point on the water surface for particle spawning.
     * @param worldPos World position to sample.
     * @param outHeight Output wave height at position.
     * @param outVelocity Output velocity direction for spray.
     * @return true if this position should spawn particles.
     */
    bool SampleWavePoint(const Vec3& worldPos, float& outHeight, Vec3& outVelocity);

    /**
     * @brief Spawn spray particles based on current conditions.
     */
    void SpawnSprayParticles(float deltaTime);

    std::unique_ptr<ParticleEmitter> m_Emitter;
    GameObject* m_WaterObject = nullptr;
    Water* m_Water = nullptr;

    bool m_Enabled = true;
    float m_SpawnAreaSize = 50.0f;      // Size of area to check for spray
    int m_SampleResolution = 8;          // Grid resolution for sampling
    float m_SpawnAccumulator = 0.0f;     // Fractional particle accumulator

    // Random number generation
    std::mt19937 m_RNG;
    std::uniform_real_distribution<float> m_Dist01;

    // Previous frame data for velocity calculation
    std::vector<float> m_PreviousHeights;
    float m_PreviousTime = 0.0f;
};
