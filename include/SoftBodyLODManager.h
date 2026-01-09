#pragma once

#include "Math/Vec3.h"
#include "SoftBodyLOD.h"
#include <memory>

class PhysXSoftBody;

/**
 * @brief Manages LOD transitions for soft bodies
 * 
 * Handles distance calculation, LOD selection, state transfer between levels,
 * and update frequency control.
 */
class SoftBodyLODManager {
public:
    SoftBodyLODManager();
    
    /**
     * @brief Update LOD based on camera distance
     * @param softBody Soft body to update
     * @param cameraPosition Current camera position
     * @param deltaTime Time since last update
     * @return True if LOD changed
     */
    bool UpdateLOD(PhysXSoftBody* softBody, const Vec3& cameraPosition, float deltaTime);
    
    /**
     * @brief Set LOD configuration
     */
    void SetLODConfig(const SoftBodyLODConfig& config);
    
    /**
     * @brief Get current LOD configuration
     */
    const SoftBodyLODConfig& GetLODConfig() const { return m_Config; }
    
    /**
     * @brief Get current LOD index
     */
    int GetCurrentLOD() const { return m_CurrentLOD; }
    
    /**
     * @brief Force specific LOD level
     * @param lodIndex LOD level to force (-1 for automatic)
     */
    void ForceLOD(int lodIndex) { m_ForcedLOD = lodIndex; }
    
    /**
     * @brief Check if update should occur this frame based on update frequency
     * @return True if simulation should update this frame
     */
    bool ShouldUpdateThisFrame() const;
    
    /**
     * @brief Get distance to camera
     */
    float GetDistanceToCamera() const { return m_DistanceToCamera; }
    
private:
    SoftBodyLODConfig m_Config;
    int m_CurrentLOD;
    int m_ForcedLOD;  // -1 for automatic
    float m_DistanceToCamera;
    int m_FrameCounter;
    
    /**
     * @brief Calculate distance from soft body to camera
     */
    float CalculateDistance(PhysXSoftBody* softBody, const Vec3& cameraPosition);
    
    /**
     * @brief Transition to new LOD level
     */
    bool TransitionToLOD(PhysXSoftBody* softBody, int newLOD);
    
    /**
     * @brief Transfer state from current LOD to new LOD
     */
    void TransferState(
        PhysXSoftBody* softBody,
        const SoftBodyLODLevel* oldLevel,
        const SoftBodyLODLevel* newLevel
    );
};
