#include "PerformanceMonitor.h"
#include <cmath>

PerformanceMonitor::PerformanceMonitor()
    : m_TargetFPS(60.0f)
    , m_LowThreshold(0.9f)
    , m_HighThreshold(1.1f)
    , m_SmoothingWindow(60)
    , m_CurrentSimulationTime(0.0f)
{
}

void PerformanceMonitor::Update(float deltaTime) {
    // Add frame time to history
    m_FrameTimes.push_back(deltaTime);
    
    // Maintain window size
    while (m_FrameTimes.size() > static_cast<size_t>(m_SmoothingWindow)) {
        m_FrameTimes.pop_front();
    }
    
    // Add simulation time to history
    m_SimulationTimes.push_back(m_CurrentSimulationTime);
    while (m_SimulationTimes.size() > static_cast<size_t>(m_SmoothingWindow)) {
        m_SimulationTimes.pop_front();
    }
    
    // Reset simulation time for next frame
    m_CurrentSimulationTime = 0.0f;
}

void PerformanceMonitor::RecordSimulationTime(float simulationTime) {
    m_CurrentSimulationTime += simulationTime;
}

float PerformanceMonitor::GetAverageFPS() const {
    if (m_FrameTimes.empty()) {
        return 0.0f;
    }
    
    float avgFrameTime = std::accumulate(m_FrameTimes.begin(), m_FrameTimes.end(), 0.0f) / m_FrameTimes.size();
    
    if (avgFrameTime < 0.0001f) {
        return 0.0f;
    }
    
    return 1.0f / avgFrameTime;
}

float PerformanceMonitor::GetAverageFrameTime() const {
    if (m_FrameTimes.empty()) {
        return 0.0f;
    }
    
    float avgFrameTime = std::accumulate(m_FrameTimes.begin(), m_FrameTimes.end(), 0.0f) / m_FrameTimes.size();
    return avgFrameTime * 1000.0f;  // Convert to milliseconds
}

float PerformanceMonitor::GetCurrentFPS() const {
    if (m_FrameTimes.empty()) {
        return 0.0f;
    }
    
    float lastFrameTime = m_FrameTimes.back();
    if (lastFrameTime < 0.0001f) {
        return 0.0f;
    }
    
    return 1.0f / lastFrameTime;
}

float PerformanceMonitor::GetAverageSimulationTime() const {
    if (m_SimulationTimes.empty()) {
        return 0.0f;
    }
    
    float avgSimTime = std::accumulate(m_SimulationTimes.begin(), m_SimulationTimes.end(), 0.0f) / m_SimulationTimes.size();
    return avgSimTime * 1000.0f;  // Convert to milliseconds
}

float PerformanceMonitor::GetSimulationPercentage() const {
    if (m_FrameTimes.empty() || m_SimulationTimes.empty()) {
        return 0.0f;
    }
    
    float avgFrameTime = std::accumulate(m_FrameTimes.begin(), m_FrameTimes.end(), 0.0f) / m_FrameTimes.size();
    float avgSimTime = std::accumulate(m_SimulationTimes.begin(), m_SimulationTimes.end(), 0.0f) / m_SimulationTimes.size();
    
    if (avgFrameTime < 0.0001f) {
        return 0.0f;
    }
    
    return (avgSimTime / avgFrameTime) * 100.0f;
}

bool PerformanceMonitor::IsPerformanceLow() const {
    float avgFPS = GetAverageFPS();
    float targetLow = m_TargetFPS * m_LowThreshold;
    
    return avgFPS < targetLow;
}

bool PerformanceMonitor::IsPerformanceHigh() const {
    float avgFPS = GetAverageFPS();
    float targetHigh = m_TargetFPS * m_HighThreshold;
    
    return avgFPS > targetHigh;
}

bool PerformanceMonitor::IsPerformanceStable() const {
    if (m_FrameTimes.size() < static_cast<size_t>(m_SmoothingWindow / 2)) {
        return false;  // Not enough data
    }
    
    float variance = CalculateVariance();
    float avgFrameTime = std::accumulate(m_FrameTimes.begin(), m_FrameTimes.end(), 0.0f) / m_FrameTimes.size();
    
    // Stable if variance is less than 10% of average
    return variance < (avgFrameTime * 0.1f);
}

void PerformanceMonitor::SetSmoothingWindow(int frames) {
    m_SmoothingWindow = std::max(1, frames);
    
    // Trim existing data if needed
    while (m_FrameTimes.size() > static_cast<size_t>(m_SmoothingWindow)) {
        m_FrameTimes.pop_front();
    }
    while (m_SimulationTimes.size() > static_cast<size_t>(m_SmoothingWindow)) {
        m_SimulationTimes.pop_front();
    }
}

void PerformanceMonitor::Reset() {
    m_FrameTimes.clear();
    m_SimulationTimes.clear();
    m_CurrentSimulationTime = 0.0f;
}

float PerformanceMonitor::CalculateVariance() const {
    if (m_FrameTimes.empty()) {
        return 0.0f;
    }
    
    float mean = std::accumulate(m_FrameTimes.begin(), m_FrameTimes.end(), 0.0f) / m_FrameTimes.size();
    
    float variance = 0.0f;
    for (float frameTime : m_FrameTimes) {
        float diff = frameTime - mean;
        variance += diff * diff;
    }
    
    return variance / m_FrameTimes.size();
}
