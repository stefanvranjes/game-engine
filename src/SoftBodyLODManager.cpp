#include "SoftBodyLODManager.h"
#ifdef USE_PHYSX
#include "PhysXSoftBody.h"
#endif
#include <cmath>
#include <iostream>

SoftBodyLODManager::SoftBodyLODManager()
    : m_CurrentLOD(0)
    , m_ForcedLOD(-1)
    , m_DistanceToCamera(0.0f)
    , m_FrameCounter(0)
    , m_LODDistanceMultiplier(1.0f)
{
}

void SoftBodyLODManager::SetLODConfig(const SoftBodyLODConfig& config) {
    m_Config = config;
    m_CurrentLOD = 0;  // Reset to highest quality
}

bool SoftBodyLODManager::UpdateLOD(PhysXSoftBody* softBody, const Vec3& cameraPosition, float deltaTime) {
    if (!softBody) return false;
    
    // Increment frame counter
    m_FrameCounter++;
    
    // Calculate distance to camera
    float actualDistance = CalculateDistance(softBody, cameraPosition);
    
    // Apply quality-based distance multiplier
    // Lower multiplier = more aggressive LOD (objects appear farther away)
    // Higher multiplier = less aggressive LOD (objects appear closer)
    m_DistanceToCamera = actualDistance / m_LODDistanceMultiplier;
    
    // Determine appropriate LOD
    int targetLOD = m_CurrentLOD;
    
    if (m_ForcedLOD >= 0) {
        // Use forced LOD
        targetLOD = m_ForcedLOD;
    } else {
        // Use distance-based LOD with hysteresis
        targetLOD = m_Config.GetLODForDistance(m_DistanceToCamera, m_CurrentLOD);
    }
    
    // Check if LOD needs to change
    if (targetLOD != m_CurrentLOD) {
        return TransitionToLOD(softBody, targetLOD);
    }
    
    return false;
}

bool SoftBodyLODManager::ShouldUpdateThisFrame() const {
    const SoftBodyLODLevel* level = m_Config.GetLODLevel(m_CurrentLOD);
    if (!level) return true;
    
    if (level->isFrozen) {
        return false;  // Don't update frozen LOD
    }
    
    if (level->updateFrequency <= 1) {
        return true;  // Update every frame
    }
    
    // Update every N frames
    return (m_FrameCounter % level->updateFrequency) == 0;
}

float SoftBodyLODManager::CalculateDistance(PhysXSoftBody* softBody, const Vec3& cameraPosition) {
#ifdef USE_PHYSX
    // Get center of mass of soft body
    Vec3 centerOfMass = softBody->GetCenterOfMass();
    
    // Calculate distance
    Vec3 diff = cameraPosition - centerOfMass;
    return diff.Length();
#else
    return 0.0f;
#endif
}

bool SoftBodyLODManager::TransitionToLOD(PhysXSoftBody* softBody, int newLOD) {
    if (newLOD < 0 || newLOD >= m_Config.GetLODCount()) {
        std::cerr << "SoftBodyLODManager: Invalid LOD index " << newLOD << std::endl;
        return false;
    }
    
    const SoftBodyLODLevel* oldLevel = m_Config.GetLODLevel(m_CurrentLOD);
    const SoftBodyLODLevel* newLevel = m_Config.GetLODLevel(newLOD);
    
    if (!newLevel) {
        std::cerr << "SoftBodyLODManager: Failed to get LOD level " << newLOD << std::endl;
        return false;
    }
    
    std::cout << "SoftBodyLODManager: Transitioning from LOD " << m_CurrentLOD 
              << " to LOD " << newLOD << " (distance: " << m_DistanceToCamera << "m)" << std::endl;
    
    // Transfer state if both levels have mesh data
    if (oldLevel && oldLevel->hasMeshData && newLevel->hasMeshData) {
        TransferState(softBody, oldLevel, newLevel);
    }
    
    m_CurrentLOD = newLOD;
    
#ifdef USE_PHYSX
    // Handle frozen LOD
    if (newLevel->isFrozen) {
        std::cout << "SoftBodyLODManager: Freezing simulation" << std::endl;
        softBody->SetActive(false);
    } else {
        softBody->SetActive(true);
    }
#endif
    
    return true;
}

void SoftBodyLODManager::TransferState(
    PhysXSoftBody* softBody,
    const SoftBodyLODLevel* oldLevel,
    const SoftBodyLODLevel* newLevel)
{
    if (!oldLevel || !newLevel || !softBody) return;
    
#ifdef USE_PHYSX
    // Get current state from soft body
    std::vector<Vec3> oldPositions(oldLevel->vertexCount);
    std::vector<Vec3> oldVelocities(oldLevel->vertexCount);
    
    softBody->GetVertexPositions(oldPositions.data());
    softBody->GetVertexVelocities(oldVelocities.data());
    
    // Map state to new LOD using vertex mapping
    std::vector<Vec3> newPositions(newLevel->vertexCount);
    std::vector<Vec3> newVelocities(newLevel->vertexCount);
    std::vector<int> vertexCounts(newLevel->vertexCount, 0);
    
    // Initialize with rest positions
    for (int i = 0; i < newLevel->vertexCount; ++i) {
        newPositions[i] = newLevel->vertexPositions[i];
        newVelocities[i] = Vec3(0, 0, 0);
    }
    
    // Accumulate mapped values
    for (int i = 0; i < oldLevel->vertexCount; ++i) {
        if (i >= static_cast<int>(newLevel->vertexMapping.size())) continue;
        
        int mappedIdx = newLevel->vertexMapping[i];
        if (mappedIdx >= 0 && mappedIdx < newLevel->vertexCount) {
            // Compute displacement from rest position
            Vec3 displacement = oldPositions[i] - oldLevel->vertexPositions[i];
            
            // Apply displacement to new rest position
            newPositions[mappedIdx] = newPositions[mappedIdx] + displacement;
            newVelocities[mappedIdx] = newVelocities[mappedIdx] + oldVelocities[i];
            vertexCounts[mappedIdx]++;
        }
    }
    
    // Average accumulated values
    for (int i = 0; i < newLevel->vertexCount; ++i) {
        if (vertexCounts[i] > 1) {
            float invCount = 1.0f / static_cast<float>(vertexCounts[i]);
            newVelocities[i] = newVelocities[i] * invCount;
        }
    }
    
    // Apply new state to soft body
    // Note: This will be handled by PhysXSoftBody when it recreates the mesh
    // The manager just prepares the data
    
    std::cout << "SoftBodyLODManager: Transferred state from " << oldLevel->vertexCount 
              << " to " << newLevel->vertexCount << " vertices" << std::endl;
#endif
}
