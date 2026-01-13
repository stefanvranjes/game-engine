#include "AdaptiveQualityIntegration.h"
#ifdef USE_PHYSX
#include "PhysXSoftBody.h"
#endif
#include <iostream>
#include <iostream>

AdaptiveQualityIntegration::AdaptiveQualityIntegration() {
    // Set default target FPS
    m_PerformanceMonitor.SetTargetFPS(60.0f);
    
    // Enable auto-adaptation
    m_QualityController.EnableAutoAdapt(true);
    
    // Set quality change callback
    m_QualityController.SetQualityChangeCallback(
        [this](QualityLevel oldLevel, QualityLevel newLevel) {
            OnQualityChanged(oldLevel, newLevel);
        }
    );
}

void AdaptiveQualityIntegration::Update(float deltaTime, const Vec3& cameraPosition) {
    // Update performance monitor
    m_PerformanceMonitor.Update(deltaTime);
    
    // Update quality controller
    m_QualityController.Update(deltaTime, m_PerformanceMonitor);
    
    // Update LOD for all soft bodies
#ifdef USE_PHYSX
    for (auto& entry : m_SoftBodies) {
        if (entry.lodManager && entry.softBody) {
            entry.lodManager->UpdateLOD(entry.softBody, cameraPosition, deltaTime);
        }
    }
#endif
}

#ifdef USE_PHYSX
void AdaptiveQualityIntegration::RegisterSoftBody(PhysXSoftBody* softBody, SoftBodyLODManager* lodManager) {
    if (!softBody || !lodManager) {
        return;
    }
    
    // Check if already registered
    for (const auto& entry : m_SoftBodies) {
        if (entry.softBody == softBody) {
            return;  // Already registered
        }
    }
    
    // Add to list
    m_SoftBodies.push_back({softBody, lodManager});
    
    // Apply current quality settings
    const auto& settings = m_QualityController.GetQualitySettings();
    lodManager->SetLODDistanceMultiplier(settings.lodDistanceMultiplier);
    
    std::cout << "Registered soft body for adaptive quality management" << std::endl;
}

void AdaptiveQualityIntegration::UnregisterSoftBody(PhysXSoftBody* softBody) {
    for (auto it = m_SoftBodies.begin(); it != m_SoftBodies.end(); ++it) {
        if (it->softBody == softBody) {
            m_SoftBodies.erase(it);
            std::cout << "Unregistered soft body from adaptive quality management" << std::endl;
            return;
        }
    }
}
#endif

void AdaptiveQualityIntegration::EnableAdaptiveQuality(bool enable) {
    m_QualityController.EnableAutoAdapt(enable);
    std::cout << "Adaptive quality " << (enable ? "enabled" : "disabled") << std::endl;
}

void AdaptiveQualityIntegration::SetTargetFPS(float fps) {
    m_PerformanceMonitor.SetTargetFPS(fps);
    std::cout << "Target FPS set to " << fps << std::endl;
}

void AdaptiveQualityIntegration::ApplyQualitySettings(const QualitySettings& settings) {
    // Apply LOD distance multiplier to all LOD managers
    for (auto& entry : m_SoftBodies) {
#ifdef USE_PHYSX
        if (entry.lodManager) {
            entry.lodManager->SetLODDistanceMultiplier(settings.lodDistanceMultiplier);
        }
#endif
    }
    
    // Note: Solver iterations and substeps would be applied to PhysXSoftBody
    // when those methods are added to the PhysXSoftBody class
    
    std::cout << "Applied quality settings:" << std::endl;
    std::cout << "  Solver Iterations: " << settings.baseSolverIterations << std::endl;
    std::cout << "  Substeps: " << settings.baseSubsteps << std::endl;
    std::cout << "  LOD Distance Multiplier: " << settings.lodDistanceMultiplier << "x" << std::endl;
    std::cout << "  Update Frequency: " << settings.updateFrequency << std::endl;
}

void AdaptiveQualityIntegration::OnQualityChanged(QualityLevel oldLevel, QualityLevel newLevel) {
    const auto& settings = m_QualityController.GetQualitySettings();
    ApplyQualitySettings(settings);
    
    const char* qualityNames[] = {"Low", "Medium", "High", "Ultra"};
    std::cout << "Quality changed: " << qualityNames[static_cast<int>(oldLevel)] 
              << " -> " << qualityNames[static_cast<int>(newLevel)] << std::endl;
}
