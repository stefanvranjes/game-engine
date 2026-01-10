#pragma once

#include "PerformanceMonitor.h"
#include <map>
#include <functional>

/**
 * @brief Quality levels for adaptive quality system
 */
enum class QualityLevel {
    Low = 0,
    Medium = 1,
    High = 2,
    Ultra = 3
};

/**
 * @brief Quality settings that affect simulation parameters
 */
struct QualitySettings {
    int baseSolverIterations;      // Constraint solver iterations
    int baseSubsteps;               // Physics substeps per frame
    float lodDistanceMultiplier;    // Multiplier for LOD distances
    int updateFrequency;            // 1 = every frame, 2 = every 2 frames, etc.
    
    QualitySettings()
        : baseSolverIterations(10)
        , baseSubsteps(1)
        , lodDistanceMultiplier(1.0f)
        , updateFrequency(1)
    {}
    
    QualitySettings(int iterations, int substeps, float lodMult, int updateFreq)
        : baseSolverIterations(iterations)
        , baseSubsteps(substeps)
        , lodDistanceMultiplier(lodMult)
        , updateFrequency(updateFreq)
    {}
};

/**
 * @brief Controls adaptive quality adjustment based on performance
 */
class AdaptiveQualityController {
public:
    using QualityChangeCallback = std::function<void(QualityLevel oldLevel, QualityLevel newLevel)>;
    
    AdaptiveQualityController();
    
    /**
     * @brief Update quality controller
     * @param deltaTime Frame time
     * @param perfMonitor Performance monitor
     */
    void Update(float deltaTime, const PerformanceMonitor& perfMonitor);
    
    // Quality management
    /**
     * @brief Set quality level manually
     */
    void SetQualityLevel(QualityLevel level);
    
    /**
     * @brief Get current quality level
     */
    QualityLevel GetCurrentQuality() const { return m_CurrentQuality; }
    
    // Auto-adaptation
    /**
     * @brief Enable/disable automatic quality adaptation
     */
    void EnableAutoAdapt(bool enable) { m_AutoAdaptEnabled = enable; }
    
    /**
     * @brief Check if auto-adaptation is enabled
     */
    bool IsAutoAdaptEnabled() const { return m_AutoAdaptEnabled; }
    
    // Quality settings
    /**
     * @brief Get quality settings for current level
     */
    const QualitySettings& GetQualitySettings() const;
    
    /**
     * @brief Set quality preset for a specific level
     */
    void SetQualityPreset(QualityLevel level, const QualitySettings& settings);
    
    /**
     * @brief Get quality preset for a specific level
     */
    const QualitySettings& GetQualityPreset(QualityLevel level) const;
    
    // Callbacks
    /**
     * @brief Set callback for quality changes
     */
    void SetQualityChangeCallback(QualityChangeCallback callback) {
        m_OnQualityChange = callback;
    }
    
    // Configuration
    /**
     * @brief Set cooldown period between quality adjustments
     */
    void SetAdaptCooldown(float seconds) { m_AdaptCooldown = seconds; }
    
    /**
     * @brief Get cooldown period
     */
    float GetAdaptCooldown() const { return m_AdaptCooldown; }
    
    /**
     * @brief Set number of consecutive frames needed to trigger adjustment
     */
    void SetConsecutiveFramesThreshold(int frames) { m_ConsecutiveFramesThreshold = frames; }
    
    /**
     * @brief Initialize default quality presets
     */
    void InitializeDefaultPresets();

private:
    QualityLevel m_CurrentQuality;
    bool m_AutoAdaptEnabled;
    float m_AdaptCooldown;              // Seconds between adjustments
    float m_TimeSinceLastAdapt;
    int m_ConsecutiveFramesThreshold;   // Frames needed to trigger change
    int m_ConsecutiveLowFrames;
    int m_ConsecutiveHighFrames;
    
    std::map<QualityLevel, QualitySettings> m_QualityPresets;
    QualityChangeCallback m_OnQualityChange;
    
    /**
     * @brief Adjust quality up or down
     */
    void AdjustQuality(bool increase);
    
    /**
     * @brief Check if quality can be adjusted
     */
    bool CanAdjustQuality() const;
    
    /**
     * @brief Get next quality level (up or down)
     */
    QualityLevel GetNextQualityLevel(bool increase) const;
};
