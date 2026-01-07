#include "WaterSprayEmitter.h"
#include "Water.h"
#include "GameObject.h"
#include "Texture.h"
#include "GLExtensions.h"
#include <cmath>

WaterSprayEmitter::WaterSprayEmitter()
    : m_RNG(std::random_device{}())
    , m_Dist01(0.0f, 1.0f)
{
}

WaterSprayEmitter::~WaterSprayEmitter() = default;

bool WaterSprayEmitter::Init(GameObject* waterObject, int maxParticles) {
    if (!waterObject || !waterObject->GetWater()) {
        return false;
    }

    m_WaterObject = waterObject;
    m_Water = waterObject->GetWater();

    // Create the underlying particle emitter
    m_Emitter = std::make_unique<ParticleEmitter>(Vec3(0, 0, 0), maxParticles);

    // Configure emitter for spray particles
    m_Emitter->SetEmitterShape(EmitterShape::Point);
    m_Emitter->SetParticleLifetime(m_Water->m_SprayLifetime);
    m_Emitter->SetSizeRange(m_Water->m_SpraySizeMin, m_Water->m_SpraySizeMax);
    m_Emitter->SetGravity(Vec3(0.0f, m_Water->m_SprayGravity, 0.0f));
    m_Emitter->SetSpawnRate(0.0f); // We control spawning manually

    // White spray with alpha fade
    m_Emitter->SetColorRange(
        Vec4(1.0f, 1.0f, 1.0f, 0.8f),
        Vec4(1.0f, 1.0f, 1.0f, 0.0f)
    );

    // Additive blending for spray
    m_Emitter->SetBlendMode(BlendMode::Additive);

    // Initialize previous heights for velocity sampling
    int totalSamples = m_SampleResolution * m_SampleResolution;
    m_PreviousHeights.resize(totalSamples, 0.0f);

    return true;
}

void WaterSprayEmitter::Update(float deltaTime, const Vec3& cameraPos) {
    if (!m_Enabled || !m_Water || !m_Emitter || !m_Water->m_UseSprayParticles) {
        return;
    }

    // Update spray parameters from water component
    m_Emitter->SetParticleLifetime(m_Water->m_SprayLifetime);
    m_Emitter->SetSizeRange(m_Water->m_SpraySizeMin, m_Water->m_SpraySizeMax);
    m_Emitter->SetGravity(Vec3(0.0f, m_Water->m_SprayGravity, 0.0f));

    // Spawn new spray particles
    SpawnSprayParticles(deltaTime);

    // Update particle physics
    m_Emitter->Update(deltaTime, cameraPos);

    m_PreviousTime += deltaTime;
}

bool WaterSprayEmitter::SampleWavePoint(const Vec3& worldPos, float& outHeight, Vec3& outVelocity) {
    if (!m_Water || m_Water->m_Displacement == 0) {
        outHeight = 0.0f;
        outVelocity = Vec3(0, 1, 0);
        return false;
    }

    // Calculate UV for FFT texture sampling
    float L = m_Water->m_OceanSize;
    Vec2 uv;
    uv.x = worldPos.x / L;
    uv.y = worldPos.z / L;

    // Wrap UV to [0, 1]
    uv.x = uv.x - std::floor(uv.x);
    uv.y = uv.y - std::floor(uv.y);

    // Read displacement from GPU texture (CPU readback for sampling)
    // Note: In a full implementation, you'd use a GPU compute shader or
    // cache the displacement map. For now, use Gerstner approximation.
    
    // Approximate wave height using Gerstner formula
    float time = m_PreviousTime;
    float height = 0.0f;
    Vec3 velocity(0, 0, 0);

    if (m_Water->m_UseGerstnerWaves) {
        float amp = m_Water->m_GerstnerAmplitude;
        float freq = m_Water->m_GerstnerFrequency;
        float steep = m_Water->m_GerstnerSteepness;

        // Wave 1
        Vec2 dir1 = Vec2(1.0f, 0.3f).Normalized();
        float k1 = 2.0f * 3.14159f * freq;
        float c1 = std::sqrt(9.8f / k1);
        float f1 = k1 * (dir1.x * worldPos.x + dir1.y * worldPos.z - c1 * time);
        height += amp * std::sin(f1);
        velocity.x += amp * k1 * dir1.x * std::cos(f1);
        velocity.y += amp * k1 * std::cos(f1);
        velocity.z += amp * k1 * dir1.y * std::cos(f1);

        // Wave 2 (secondary)
        Vec2 dir2 = Vec2(0.5f, 0.8f).Normalized();
        float f2 = k1 * 1.3f * (dir2.x * worldPos.x + dir2.y * worldPos.z - c1 * 1.1f * time);
        height += amp * 0.5f * std::sin(f2);
        velocity.y += amp * 0.5f * k1 * 1.3f * std::cos(f2);
    }

    outHeight = height;
    outVelocity = velocity.Normalized();
    
    // Spawn particles if wave height exceeds threshold
    return height > m_Water->m_SpraySpawnThreshold;
}

void WaterSprayEmitter::SpawnSprayParticles(float deltaTime) {
    if (!m_Water || !m_WaterObject) return;

    Mat4 worldMatrix = m_WaterObject->GetWorldMatrix();
    Vec3 waterPos = worldMatrix.GetTranslation();

    float halfSize = m_SpawnAreaSize * 0.5f;
    float step = m_SpawnAreaSize / m_SampleResolution;

    // Accumulate particles to spawn
    float particlesToSpawn = m_Water->m_SprayIntensity * deltaTime + m_SpawnAccumulator;
    int spawnCount = static_cast<int>(particlesToSpawn);
    m_SpawnAccumulator = particlesToSpawn - spawnCount;

    if (spawnCount <= 0) return;

    int spawned = 0;

    // Sample grid for potential spawn points
    for (int i = 0; i < m_SampleResolution && spawned < spawnCount; ++i) {
        for (int j = 0; j < m_SampleResolution && spawned < spawnCount; ++j) {
            // Random offset within cell
            float offsetX = (m_Dist01(m_RNG) - 0.5f) * step;
            float offsetZ = (m_Dist01(m_RNG) - 0.5f) * step;

            Vec3 samplePos(
                waterPos.x - halfSize + i * step + offsetX,
                waterPos.y,
                waterPos.z - halfSize + j * step + offsetZ
            );

            float waveHeight;
            Vec3 waveVelocity;

            if (SampleWavePoint(samplePos, waveHeight, waveVelocity)) {
                // Spawn particle at wave crest
                Vec3 spawnPos = samplePos;
                spawnPos.y = waterPos.y + waveHeight + m_Water->m_PlaneY;

                // Calculate spray velocity (upward with some horizontal spread)
                float speedScale = m_Water->m_SprayVelocityScale * (waveHeight / m_Water->m_SpraySpawnThreshold);
                speedScale = std::min(speedScale, m_Water->m_SprayVelocityScale * 2.0f);

                Vec3 sprayVel;
                sprayVel.x = waveVelocity.x * speedScale * 0.5f + (m_Dist01(m_RNG) - 0.5f) * 2.0f;
                sprayVel.y = std::abs(waveVelocity.y) * speedScale + m_Dist01(m_RNG) * speedScale;
                sprayVel.z = waveVelocity.z * speedScale * 0.5f + (m_Dist01(m_RNG) - 0.5f) * 2.0f;

                // Position the emitter and spawn
                m_Emitter->SetPosition(spawnPos);
                m_Emitter->SetVelocityRange(
                    sprayVel * 0.8f,
                    sprayVel * 1.2f
                );

                // Manually emit a particle
                // Access internal spawn (this requires the emitter to have a manual spawn method)
                // For now, we set spawn rate briefly
                m_Emitter->SetSpawnRate(1000.0f); // High rate for burst
                m_Emitter->Update(0.001f, Vec3(0, 0, 0)); // Micro-update to spawn
                m_Emitter->SetSpawnRate(0.0f); // Reset

                spawned++;
            }
        }
    }
}

void WaterSprayEmitter::SetTexture(std::shared_ptr<Texture> texture) {
    if (m_Emitter) {
        m_Emitter->SetTexture(texture);
    }
}
