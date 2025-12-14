#include "AudioOcclusion.h"
#include "GameObject.h"
#include "AudioSource.h"
#include "Physics/PhysicsManager.h"
#include <iostream>
#include <cmath>

AudioOcclusion& AudioOcclusion::Get() {
    static AudioOcclusion instance;
    return instance;
}

AudioOcclusion::AudioOcclusion() {
    // Initialize default material properties
    m_materialProperties[static_cast<int>(MaterialType::Air)] = {
        0.0f,  // No occlusion
        0.0f,  // No damping
        0.9f   // Highly reflective to sound
    };

    m_materialProperties[static_cast<int>(MaterialType::Glass)] = {
        0.1f,   // Slight occlusion
        0.05f,  // Minimal damping
        0.8f
    };

    m_materialProperties[static_cast<int>(MaterialType::Drywall)] = {
        0.4f,
        0.3f,
        0.5f
    };

    m_materialProperties[static_cast<int>(MaterialType::Brick)] = {
        0.6f,
        0.5f,
        0.4f
    };

    m_materialProperties[static_cast<int>(MaterialType::Wood)] = {
        0.5f,
        0.35f,
        0.45f
    };

    m_materialProperties[static_cast<int>(MaterialType::Metal)] = {
        0.7f,   // Heavy occlusion
        0.15f,  // Low absorption
        0.9f    // Very reflective
    };

    m_materialProperties[static_cast<int>(MaterialType::Concrete)] = {
        0.75f,
        0.45f,
        0.3f
    };

    m_materialProperties[static_cast<int>(MaterialType::Stone)] = {
        0.8f,
        0.55f,
        0.25f
    };

    m_materialProperties[static_cast<int>(MaterialType::Water)] = {
        0.85f,  // Severe attenuation (sound travels differently in water)
        0.65f,  // Heavy damping
        0.2f
    };
}

AudioOcclusion::~AudioOcclusion() {
    Shutdown();
}

bool AudioOcclusion::Initialize() {
    if (m_initialized) return true;
    m_initialized = true;
    return true;
}

void AudioOcclusion::Shutdown() {
    m_obstacles.clear();
    m_initialized = false;
}

void AudioOcclusion::RegisterObstacle(GameObject* gameObject, MaterialType material) {
    if (!gameObject) return;

    uintptr_t key = reinterpret_cast<uintptr_t>(gameObject);
    m_obstacles[key] = { gameObject, material };
}

void AudioOcclusion::UnregisterObstacle(GameObject* gameObject) {
    if (!gameObject) return;

    uintptr_t key = reinterpret_cast<uintptr_t>(gameObject);
    m_obstacles.erase(key);
}

void AudioOcclusion::UpdateObstacleMaterial(GameObject* gameObject, MaterialType material) {
    if (!gameObject) return;

    uintptr_t key = reinterpret_cast<uintptr_t>(gameObject);
    auto it = m_obstacles.find(key);
    if (it != m_obstacles.end()) {
        it->second.second = material;
    }
}

OcclusionResult AudioOcclusion::ComputeOcclusion(const Vec3& listenerPos, const Vec3& sourcePos,
                                                 GameObject* sourceObject) {
    OcclusionResult result;
    m_lastResult = result;

    if (!m_enabled || m_obstacles.empty()) {
        return result;
    }

    // Check distance
    Vec3 toSource = sourcePos - listenerPos;
    float distance = toSource.Length();

    if (distance > m_maxOcclusionDistance) {
        return result; // Too far, no occlusion computation
    }

    // Perform raycast
    RaycastToSource(listenerPos, sourcePos, sourceObject, result);

    // Compute filters based on occlusion
    ComputeFilters(result.occlusionStrength, result.dampingStrength, result.lpfCutoff, result.hpfCutoff);

    m_lastResult = result;
    return result;
}

void AudioOcclusion::ApplyOcclusionToSource(AudioSource* audioSource, const Vec3& listenerPos, const Vec3& sourcePos) {
    if (!audioSource) return;

    auto result = ComputeOcclusion(listenerPos, sourcePos);
    audioSource->SetOcclusion(result.occlusionStrength);
}

bool AudioOcclusion::RaycastToSource(const Vec3& from, const Vec3& to, GameObject* excludeObject,
                                     OcclusionResult& outResult) {
    Vec3 direction = (to - from).Normalized();
    float totalDistance = (to - from).Length();

    outResult.isOccluded = false;
    outResult.occlusionStrength = 0.0f;
    outResult.dampingStrength = 0.0f;
    outResult.distanceThroughMaterial = 0.0f;
    outResult.occludingObstaclesCount = 0;

    // Simple raycast: check if any obstacle intersects the line segment
    for (const auto& pair : m_obstacles) {
        GameObject* obstacle = pair.second.first;
        MaterialType materialType = pair.second.second;

        if (obstacle == excludeObject) continue; // Skip source object

        // Basic AABB intersection test (assumes GameObject has bounding box)
        // For more accurate results, use full physics raycasting
        
        // Simplified approach: check if obstacle is roughly in the path
        Vec3 toObstacle = obstacle->GetTransform().GetPosition() - from;
        float projectedDistance = toObstacle.Dot(direction);

        if (projectedDistance > 0.0f && projectedDistance < totalDistance) {
            // Obstacle is roughly in the path
            float lateralDistance = (toObstacle - direction * projectedDistance).Length();
            float obstacleRadius = 1.0f; // Assume unit size; could use actual collision bounds

            if (lateralDistance < obstacleRadius) {
                // Intersection likely
                const MaterialProperties& props = GetMaterialProperties(materialType);
                
                outResult.isOccluded = true;
                outResult.occludingObstaclesCount++;
                outResult.occlusionStrength = std::max(outResult.occlusionStrength, props.occlusionFactor);
                outResult.dampingStrength = std::max(outResult.dampingStrength, props.dampingFactor);
                outResult.distanceThroughMaterial += props.defaultThickness;
            }
        }
    }

    // Accumulate occlusion from multiple obstacles (not purely additive)
    if (outResult.occludingObstaclesCount > 0) {
        // Combine occlusion: don't just add them
        // Use a logarithmic scale for more natural sound behavior
        outResult.occlusionStrength = std::pow(outResult.occlusionStrength, 0.8f);
    }

    return outResult.isOccluded;
}

void AudioOcclusion::ComputeFilters(float occlusionStrength, float dampingStrength,
                                    float& outLPF, float& outHPF) const {
    occlusionStrength = std::max(0.0f, std::min(1.0f, occlusionStrength));
    dampingStrength = std::max(0.0f, std::min(1.0f, dampingStrength));

    if (m_advancedFiltering) {
        // Advanced filtering: separate damping and occlusion effects
        
        // Damping -> LPF (remove high frequencies)
        float lpfRange = m_lpfMaxFrequency - m_lpfMinFrequency;
        outLPF = m_lpfMaxFrequency - (lpfRange * dampingStrength * m_lpfOcclusionScale);

        // Occlusion strength -> more damping
        outLPF -= (lpfRange * occlusionStrength * 0.3f * m_lpfOcclusionScale);

        outLPF = std::max(m_lpfMinFrequency, outLPF);

        // HPF: slightly boost (reduce) with occlusion to avoid muddiness
        float hpfOffset = (occlusionStrength + dampingStrength) * 50.0f;
        outHPF = 20.0f + hpfOffset;
    } else {
        // Simple filtering: combined effect
        float combinedOcclusion = (occlusionStrength + dampingStrength) / 2.0f;
        float lpfRange = m_lpfMaxFrequency - m_lpfMinFrequency;

        outLPF = m_lpfMaxFrequency - (lpfRange * combinedOcclusion * m_lpfOcclusionScale);
        outLPF = std::max(m_lpfMinFrequency, outLPF);

        outHPF = 20.0f + (combinedOcclusion * 100.0f);
    }
}

const AudioOcclusion::MaterialProperties& AudioOcclusion::GetMaterialProperties(MaterialType material) const {
    auto it = m_materialProperties.find(static_cast<int>(material));
    if (it != m_materialProperties.end()) {
        return it->second;
    }

    // Return air (transparent) as default
    return m_materialProperties.at(static_cast<int>(MaterialType::Air));
}

void AudioOcclusion::SetMaterialProperties(MaterialType material, const MaterialProperties& props) {
    m_materialProperties[static_cast<int>(material)] = props;
}

void AudioOcclusion::SetCustomMaterial(const std::string& name, const MaterialProperties& props) {
    m_customMaterials[name] = props;
}

void AudioOcclusion::SetRaycastSampleCount(int samples) {
    m_raycastSampleCount = std::max(1, samples);
}

void AudioOcclusion::SetMaxOcclusionDistance(float distance) {
    m_maxOcclusionDistance = std::max(1.0f, distance);
}

void AudioOcclusion::SetLPFParameters(float minFrequency, float maxFrequency, float occlusionScale) {
    m_lpfMinFrequency = minFrequency;
    m_lpfMaxFrequency = maxFrequency;
    m_lpfOcclusionScale = occlusionScale;
}
