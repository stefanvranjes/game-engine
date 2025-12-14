#pragma once

#include <string>
#include <unordered_map>
#include <chrono>

namespace Tools {

class PerformanceProfiler {
public:
    static PerformanceProfiler& Get();
    
    // Frame profiling
    void BeginFrame();
    void EndFrame();
    
    // Section timing
    void BeginSection(const std::string& name);
    void EndSection(const std::string& name);
    
    // Memory tracking
    void TrackMemory(const std::string& category, size_t bytes);
    
    // GPU metrics
    void TrackGPUTime(const std::string& label, float milliseconds);
    
    // Reporting
    void PrintReport();
    void ExportReport(const std::string& filepath);
    
    float GetFrameTime() const { return m_FrameTime; }
    float GetFPS() const { return m_FPS; }

private:
    struct TimingData {
        float totalTime = 0.0f;
        float minTime = FLT_MAX;
        float maxTime = 0.0f;
        int sampleCount = 0;
    };
    
    std::unordered_map<std::string, TimingData> m_Timings;
    std::chrono::high_resolution_clock::time_point m_FrameStart;
    std::chrono::high_resolution_clock::time_point m_SectionStart;
    float m_FrameTime = 0.0f;
    float m_FPS = 0.0f;
    int m_FrameCount = 0;
};

} // namespace Tools