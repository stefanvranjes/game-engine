#include "AudioOcclusion.h"
#include "PhysicsSystem.h" // Keep for legacy if needed or remove?
#include "Application.h"
#ifdef USE_PHYSX
#include "PhysXBackend.h"
#endif
#include "IPhysicsBackend.h"
#include "GameObject.h"
#include "AudioSource.h"
#include <string>

AudioOcclusion& AudioOcclusion::Get() {
    static AudioOcclusion instance;
    return instance;
}

bool AudioOcclusion::Initialize() {
    if (m_initialized) return true;
    
    // Default Materials
    m_materialProperties[static_cast<int>(MaterialType::Air)]      = {0.0f, 0.0f, 0.0f, 0.0f};
    m_materialProperties[static_cast<int>(MaterialType::Glass)]    = {0.1f, 0.05f, 0.3f, 0.02f};
    m_materialProperties[static_cast<int>(MaterialType::Drywall)]  = {0.4f, 0.2f, 0.4f, 0.1f};
    m_materialProperties[static_cast<int>(MaterialType::Wood)]     = {0.5f, 0.3f, 0.4f, 0.15f};
    m_materialProperties[static_cast<int>(MaterialType::Brick)]    = {0.6f, 0.4f, 0.6f, 0.2f};
    m_materialProperties[static_cast<int>(MaterialType::Metal)]    = {0.7f, 0.6f, 0.8f, 0.05f};
    m_materialProperties[static_cast<int>(MaterialType::Concrete)] = {0.75f, 0.7f, 0.7f, 0.3f};
    m_materialProperties[static_cast<int>(MaterialType::Stone)]    = {0.8f, 0.8f, 0.8f, 0.4f};
    
    m_initialized = true;
    return true;
}

void AudioOcclusion::Shutdown() {
    m_obstacles.clear();
    m_initialized = false;
}

void AudioOcclusion::RegisterObstacle(GameObject* gameObject, MaterialType material) {
    if (!gameObject) return;
    m_obstacles[reinterpret_cast<uintptr_t>(gameObject)] = {gameObject, material};
}

void AudioOcclusion::UnregisterObstacle(GameObject* gameObject) {
    if (!gameObject) return;
    m_obstacles.erase(reinterpret_cast<uintptr_t>(gameObject));
}

AudioOcclusion::OcclusionResult AudioOcclusion::ComputeOcclusion(const Vec3& listenerPos, const Vec3& sourcePos, GameObject* sourceObject) {
    OcclusionResult result;
    if (!m_enabled) return result;

    Vec3 dir = sourcePos - listenerPos;
    float dist = dir.Length();
    
    if (dist > m_maxOcclusionDistance || dist < 0.001f) {
        return result; 
    }
    
    dir /= dist; // Normalize

    PhysicsRaycastHit hit;
#ifndef USE_PHYSX
    return result; // Or use legacy PhysicsSystem::RaycastHit
#endif
    
    Vec3 currentRayStart = listenerPos;
    float remainingDist = dist;
    int maxSteps = 5;
    const float kEpsilon = 0.05f; 

#ifdef USE_PHYSX
    auto* backend = Application::Get().GetPhysXBackend();
#else
    IPhysicsBackend* backend = nullptr;
#endif

    if (!backend) return result;

    // Filter: Ignore the source object itself if provided
    // Not easily possible with standard filter bitmask unless we put source in a specific layer.
    // But we can check hit.userData.

    for (int i = 0; i < maxSteps; ++i) {
        if (remainingDist <= kEpsilon) break;

        bool hasHit = backend->Raycast(currentRayStart, sourcePos, hit);
        
        if (hasHit) {
            // Check if we hit the source object (destination reached)
            if (sourceObject && hit.userData == sourceObject) {
                break; // Reached source
            }
            
            // If hits something else close to source (within epsilon), assume we reached source
            if (hit.distance >= remainingDist - kEpsilon) {
                break;
            }

            // We hit an obstacle!
            GameObject* hitObj = static_cast<GameObject*>(hit.userData); // Assuming userData is GameObject
            
            // Determine material
            MaterialType matType = MaterialType::Concrete; // Default
            
            // Check registered obstacles
            if (hitObj) {
                auto it = m_obstacles.find(reinterpret_cast<uintptr_t>(hitObj));
                if (it != m_obstacles.end()) {
                    matType = it->second.second;
                }
            }
            
            const MaterialProperties& props = GetMaterialProperties(matType);
            
            result.isOccluded = true;
            result.occludingObstaclesCount++;
            
            // Accumulate occlusion
            // Simplistic: Occlusion adds up, clamped to 1.0f
            // Or multiplicative transmission: NewTrans = OldTrans * (1 - occlusion)
            // Let's use multiplicative for 'clearness' -> 1 - occlusion
            float transmission = 1.0f - result.occlusionStrength;
            transmission *= (1.0f - props.occlusionFactor);
            result.occlusionStrength = 1.0f - transmission;

            result.dampingStrength = std::max(result.dampingStrength, props.dampingFactor);
            
            // Step forward through the wall
            // Ideal: Raycast back from other side or assume thickness.
            // Simplified: Advance ray by fixed thickness or material property default thickness
            float thickness = props.defaultThickness;
            result.distanceThroughMaterial += thickness;
            
            currentRayStart = hit.point + (dir * thickness) + (dir * kEpsilon);
            remainingDist -= (hit.distance + thickness);
        } else {
            // No hit, path clear to source
            break;
        }
    }
    
    // Compute Cutoffs based on occlusion
    ComputeFilters(result.occlusionStrength, result.dampingStrength, result.lpfCutoff, result.hpfCutoff);
    
    m_lastResult = result;
    return result;
}

void AudioOcclusion::ComputeFilters(float occlusionStrength, float dampingStrength, float& outLPF, float& outHPF) const {
    if (occlusionStrength <= 0.001f) {
        outLPF = 22000.0f; // Open
        outHPF = 0.0f;
        return;
    }
    
    // LPF: Closed gets lower.
    // range: 20000 -> 500
    float lpfLog = std::log(m_lpfMaxFrequency);
    float lpfMinLog = std::log(m_lpfMinFrequency);
    
    // Linear interp in log space sounds better
    float t = 1.0f - (occlusionStrength * m_lpfOcclusionScale); 
    if (t < 0) t = 0;
    
    outLPF = std::exp(lpfMinLog + t * (lpfLog - lpfMinLog));
    
    // Damping affects high freqs additionally?
    if (dampingStrength > 0) {
        outLPF *= (1.0f - dampingStrength * 0.5f);
    }
    
    // HPF: Not usually affected much by occlusion unless accurate diffraction modeling.
    // But distance might affect it (handled in spatializer).
    outHPF = 0.0f;
}

const AudioOcclusion::MaterialProperties& AudioOcclusion::GetMaterialProperties(MaterialType material) const {
    auto it = m_materialProperties.find(static_cast<int>(material));
    if (it != m_materialProperties.end()) {
        return it->second;
    }
    static MaterialProperties defaultProps;
    return defaultProps;
}

void AudioOcclusion::ApplyOcclusionToSource(AudioSource* audioSource, const Vec3& listenerPos, const Vec3& sourcePos) {
    if (!audioSource) return;
    
    // Optimization: Don't compute every frame?
    // Or maybe we do for smoothness.
    
    OcclusionResult res = ComputeOcclusion(listenerPos, sourcePos, nullptr); // Need reference to Source GameObject ideally
    
    // Apply to source
    // AudioSource::SetOcclusion maps 0-1 to internal filter
    audioSource->SetOcclusion(res.occlusionStrength);
}
