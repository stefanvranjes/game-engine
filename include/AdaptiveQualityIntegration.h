#pragma once

#include "PerformanceMonitor.h"
#include "AdaptiveQualityController.h"
#include "SoftBodyLODManager.h"
#include <vector>
#include <memory>

class PhysXSoftBody;

/**
 * @brief Integrates adaptive quality system with soft body LOD management
 * 
 * This class coordinates between performance monitoring, quality control,
 * and LOD management to provide optimal performance across different hardware.
 */
class AdaptiveQualityIntegration {
public:
    AdaptiveQualityIntegration();
    
    /**
     * @brief Update all systems
     * @param deltaTime Frame time
     * @param cameraPosition Current camera position
     */
    void Update(float deltaTime, const Vec3& cameraPosition);
    
    /**
     * @brief Register soft body for adaptive quality management
     */
    void RegisterSoftBody(PhysXSoftBody* softBody, SoftBodyLODManager* lodManager);
    
    /**
     * @brief Unregister soft body
     */
    void UnregisterSoftBody(PhysXSoftBody* softBody);
    
    /**
     * @brief Get performance monitor
     */
    PerformanceMonitor& GetPerformanceMonitor() { return m_PerformanceMonitor; }
    
    /**
     * @brief Get quality controller
     */
    AdaptiveQualityController& GetQualityController() { return m_QualityController; }
    
    /**
     * @brief Enable/disable adaptive quality
     */
    void EnableAdaptiveQuality(bool enable);
    
    /**
     * @brief Set target FPS
     */
    void SetTargetFPS(float fps);

private:
    struct SoftBodyEntry {
        PhysXSoftBody* softBody;
        SoftBodyLODManager* lodManager;
    };
    
    PerformanceMonitor m_PerformanceMonitor;
    AdaptiveQualityController m_QualityController;
    std::vector<SoftBodyEntry> m_SoftBodies;
    
    /**
     * @brief Apply quality settings to all registered soft bodies
     */
    void ApplyQualitySettings(const QualitySettings& settings);
    
    /**
     * @brief Callback for quality changes
     */
    void OnQualityChanged(QualityLevel oldLevel, QualityLevel newLevel);
};
