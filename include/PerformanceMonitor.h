#pragma once

#include <deque>
#include <algorithm>
#include <numeric>

/**
 * @brief Monitors application performance metrics
 */
class PerformanceMonitor {
public:
    PerformanceMonitor();
    
    /**
     * @brief Update performance metrics
     * @param deltaTime Frame time in seconds
     */
    void Update(float deltaTime);
    
    /**
     * @brief Record simulation time for current frame
     * @param simulationTime Time spent in physics simulation (seconds)
     */
    void RecordSimulationTime(float simulationTime);
    
    // Metrics
    /**
     * @brief Get average FPS over smoothing window
     */
    float GetAverageFPS() const;
    
    /**
     * @brief Get average frame time in milliseconds
     */
    float GetAverageFrameTime() const;
    
    /**
     * @brief Get current FPS (instantaneous)
     */
    float GetCurrentFPS() const;
    
    /**
     * @brief Get average simulation time in milliseconds
     */
    float GetAverageSimulationTime() const;
    
    /**
     * @brief Get simulation time as percentage of frame time
     */
    float GetSimulationPercentage() const;
    
    // Performance state
    /**
     * @brief Check if performance is below target
     */
    bool IsPerformanceLow() const;
    
    /**
     * @brief Check if performance is above target with margin
     */
    bool IsPerformanceHigh() const;
    
    /**
     * @brief Check if performance is stable (low variance)
     */
    bool IsPerformanceStable() const;
    
    // Configuration
    /**
     * @brief Set target FPS
     */
    void SetTargetFPS(float fps) { m_TargetFPS = fps; }
    
    /**
     * @brief Get target FPS
     */
    float GetTargetFPS() const { return m_TargetFPS; }
    
    /**
     * @brief Set smoothing window size (number of frames)
     */
    void SetSmoothingWindow(int frames);
    
    /**
     * @brief Set low performance threshold (fraction of target FPS)
     */
    void SetLowThreshold(float threshold) { m_LowThreshold = threshold; }
    
    /**
     * @brief Set high performance threshold (fraction of target FPS)
     */
    void SetHighThreshold(float threshold) { m_HighThreshold = threshold; }
    
    /**
     * @brief Reset all metrics
     */
    void Reset();

private:
    std::deque<float> m_FrameTimes;           // Frame times in seconds
    std::deque<float> m_SimulationTimes;      // Simulation times in seconds
    
    float m_TargetFPS;
    float m_LowThreshold;    // e.g., 0.9 = 90% of target
    float m_HighThreshold;   // e.g., 1.1 = 110% of target
    int m_SmoothingWindow;
    
    float m_CurrentSimulationTime;
    
    /**
     * @brief Calculate variance of frame times
     */
    float CalculateVariance() const;
};
