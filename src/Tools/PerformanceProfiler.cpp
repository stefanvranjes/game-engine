#include "PerformanceProfiler.h"
#include <iostream>
#include <fstream>
#include <algorithm>

namespace Tools {

PerformanceProfiler& PerformanceProfiler::Get() {
    static PerformanceProfiler instance;
    return instance;
}

void PerformanceProfiler::BeginFrame() {
    m_FrameStart = std::chrono::high_resolution_clock::now();
}

void PerformanceProfiler::EndFrame() {
    auto frameEnd = std::chrono::high_resolution_clock::now();
    m_FrameTime = std::chrono::duration<float, std::milli>(frameEnd - m_FrameStart).count();
    
    m_FrameCount++;
    if (m_FrameCount % 60 == 0) {
        m_FPS = 1000.0f / m_FrameTime;
    }
}

void PerformanceProfiler::BeginSection(const std::string& name) {
    m_SectionStart = std::chrono::high_resolution_clock::now();
}

void PerformanceProfiler::EndSection(const std::string& name) {
    auto sectionEnd = std::chrono::high_resolution_clock::now();
    float elapsed = std::chrono::duration<float, std::milli>(sectionEnd - m_SectionStart).count();
    
    if (m_Timings.find(name) == m_Timings.end()) {
        m_Timings[name] = TimingData();
    }
    
    auto& timing = m_Timings[name];
    timing.totalTime += elapsed;
    timing.minTime = std::min(timing.minTime, elapsed);
    timing.maxTime = std::max(timing.maxTime, elapsed);
    timing.sampleCount++;
}

void PerformanceProfiler::TrackMemory(const std::string& category, size_t bytes) {
    std::cout << "Memory [" << category << "]: " << (bytes / 1024.0f) << " KB" << std::endl;
}

void PerformanceProfiler::TrackGPUTime(const std::string& label, float milliseconds) {
    std::cout << "GPU Time [" << label << "]: " << milliseconds << " ms" << std::endl;
}

void PerformanceProfiler::PrintReport() {
    std::cout << "\n=== Performance Report ===" << std::endl;
    std::cout << "FPS: " << m_FPS << std::endl;
    std::cout << "Frame Time: " << m_FrameTime << " ms\n" << std::endl;
    
    for (const auto& [name, timing] : m_Timings) {
        float avgTime = timing.totalTime / timing.sampleCount;
        std::cout << name << ":\n";
        std::cout << "  Average: " << avgTime << " ms\n";
        std::cout << "  Min: " << timing.minTime << " ms\n";
        std::cout << "  Max: " << timing.maxTime << " ms\n";
    }
}

void PerformanceProfiler::ExportReport(const std::string& filepath) {
    std::cout << "Exporting performance report to: " << filepath << std::endl;
    
    std::ofstream file(filepath);
    if (file.is_open()) {
        file << "Performance Report\n";
        file << "==================\n\n";
        file << "FPS: " << m_FPS << "\n";
        file << "Frame Time: " << m_FrameTime << " ms\n\n";
        
        file << "Detailed Timings:\n";
        for (const auto& [name, timing] : m_Timings) {
            float avgTime = timing.totalTime / timing.sampleCount;
            file << name << "\n";
            file << "  Average: " << avgTime << " ms\n";
            file << "  Min: " << timing.minTime << " ms\n";
            file << "  Max: " << timing.maxTime << " ms\n";
            file << "  Samples: " << timing.sampleCount << "\n\n";
        }
        
        file.close();
        std::cout << "Report exported successfully" << std::endl;
    }
}

} // namespace Tools