#include "AdaptiveQualityController.h"
#include <iostream>

AdaptiveQualityController::AdaptiveQualityController()
    : m_CurrentQuality(QualityLevel::High)
    , m_AutoAdaptEnabled(true)
    , m_AdaptCooldown(2.0f)
    , m_TimeSinceLastAdapt(0.0f)
    , m_ConsecutiveFramesThreshold(30)
    , m_ConsecutiveLowFrames(0)
    , m_ConsecutiveHighFrames(0)
{
    InitializeDefaultPresets();
}

void AdaptiveQualityController::Update(float deltaTime, const PerformanceMonitor& perfMonitor) {
    if (!m_AutoAdaptEnabled) {
        return;
    }
    
    m_TimeSinceLastAdapt += deltaTime;
    
    // Track consecutive frames
    if (perfMonitor.IsPerformanceLow()) {
        m_ConsecutiveLowFrames++;
        m_ConsecutiveHighFrames = 0;
    } else if (perfMonitor.IsPerformanceHigh() && perfMonitor.IsPerformanceStable()) {
        m_ConsecutiveHighFrames++;
        m_ConsecutiveLowFrames = 0;
    } else {
        // Performance is in acceptable range
        m_ConsecutiveLowFrames = 0;
        m_ConsecutiveHighFrames = 0;
    }
    
    if (!CanAdjustQuality()) {
        return;
    }
    
    // Decrease quality if performance is consistently low
    if (m_ConsecutiveLowFrames >= m_ConsecutiveFramesThreshold) {
        AdjustQuality(false);
        m_ConsecutiveLowFrames = 0;
    }
    // Increase quality if performance is consistently high and stable
    else if (m_ConsecutiveHighFrames >= m_ConsecutiveFramesThreshold * 2) {
        // Require more frames for quality increase to prevent thrashing
        AdjustQuality(true);
        m_ConsecutiveHighFrames = 0;
    }
}

void AdaptiveQualityController::SetQualityLevel(QualityLevel level) {
    if (level == m_CurrentQuality) {
        return;
    }
    
    QualityLevel oldLevel = m_CurrentQuality;
    m_CurrentQuality = level;
    
    // Reset counters
    m_ConsecutiveLowFrames = 0;
    m_ConsecutiveHighFrames = 0;
    m_TimeSinceLastAdapt = 0.0f;
    
    // Notify callback
    if (m_OnQualityChange) {
        m_OnQualityChange(oldLevel, m_CurrentQuality);
    }
    
    std::cout << "Quality level changed: " << static_cast<int>(oldLevel) 
              << " -> " << static_cast<int>(m_CurrentQuality) << std::endl;
}

const QualitySettings& AdaptiveQualityController::GetQualitySettings() const {
    return GetQualityPreset(m_CurrentQuality);
}

void AdaptiveQualityController::SetQualityPreset(QualityLevel level, const QualitySettings& settings) {
    m_QualityPresets[level] = settings;
}

const QualitySettings& AdaptiveQualityController::GetQualityPreset(QualityLevel level) const {
    auto it = m_QualityPresets.find(level);
    if (it != m_QualityPresets.end()) {
        return it->second;
    }
    
    // Return default if not found
    static QualitySettings defaultSettings;
    return defaultSettings;
}

void AdaptiveQualityController::InitializeDefaultPresets() {
    // Ultra Quality: Maximum fidelity
    m_QualityPresets[QualityLevel::Ultra] = QualitySettings(
        20,     // solver iterations
        4,      // substeps
        1.5f,   // LOD distance multiplier (larger = more detail at distance)
        1       // update every frame
    );
    
    // High Quality: Balanced performance
    m_QualityPresets[QualityLevel::High] = QualitySettings(
        15,     // solver iterations
        2,      // substeps
        1.0f,   // LOD distance multiplier
        1       // update every frame
    );
    
    // Medium Quality: Performance focused
    m_QualityPresets[QualityLevel::Medium] = QualitySettings(
        10,     // solver iterations
        1,      // substeps
        0.75f,  // LOD distance multiplier
        1       // update every frame
    );
    
    // Low Quality: Maximum performance
    m_QualityPresets[QualityLevel::Low] = QualitySettings(
        5,      // solver iterations
        1,      // substeps
        0.5f,   // LOD distance multiplier
        2       // update every 2 frames
    );
}

void AdaptiveQualityController::AdjustQuality(bool increase) {
    QualityLevel newLevel = GetNextQualityLevel(increase);
    
    if (newLevel != m_CurrentQuality) {
        SetQualityLevel(newLevel);
    }
}

bool AdaptiveQualityController::CanAdjustQuality() const {
    return m_TimeSinceLastAdapt >= m_AdaptCooldown;
}

QualityLevel AdaptiveQualityController::GetNextQualityLevel(bool increase) const {
    int currentLevel = static_cast<int>(m_CurrentQuality);
    
    if (increase) {
        // Increase quality (higher enum value)
        if (currentLevel < static_cast<int>(QualityLevel::Ultra)) {
            return static_cast<QualityLevel>(currentLevel + 1);
        }
    } else {
        // Decrease quality (lower enum value)
        if (currentLevel > static_cast<int>(QualityLevel::Low)) {
            return static_cast<QualityLevel>(currentLevel - 1);
        }
    }
    
    return m_CurrentQuality;
}
