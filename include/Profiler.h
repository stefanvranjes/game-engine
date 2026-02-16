#pragma once

#include <chrono>
#include <string>
#include <vector>
#include <unordered_map>
#include <stack>
#include <memory>
#include <mutex>
#include <thread>
#include <nlohmann/json.hpp>
#include <glm/glm.hpp>

using json = nlohmann::json;

/**
 * @brief CPU Profiler with scoped markers and frame timing
 * 
 * Tracks CPU performance metrics with hierarchical scopes.
 * Thread-safe and designed for real-time game performance monitoring.
 * 
 * Usage:
 *   Profiler::BeginScope("RenderPass");
 *   // ... code ...
 *   Profiler::EndScope();
 * 
 * Or use SCOPED_PROFILE macro for automatic timing.
 */
class Profiler
{
public:
    struct Marker
    {
        std::string name;
        std::chrono::high_resolution_clock::time_point start_time;
        std::chrono::high_resolution_clock::time_point end_time;
        uint32_t depth;  // Hierarchy depth (0 = root)
        uint32_t frame;
        std::vector<std::shared_ptr<Marker>> children;

        double GetDurationMs() const
        {
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            return duration.count() / 1000.0;
        }
    };

    struct FrameStats
    {
        uint32_t frame_number;
        double frame_time_ms;
        double cpu_time_ms;
        std::shared_ptr<Marker> root_marker;
        std::chrono::high_resolution_clock::time_point frame_start;
        std::chrono::high_resolution_clock::time_point frame_end;
        std::unordered_map<std::string, std::vector<double>> marker_times;  // Marker name -> durations
    };

    static Profiler& Instance();

    // Frame management
    void BeginFrame();
    void EndFrame();
    uint32_t GetCurrentFrame() const { return current_frame_; }

    // Marker management
    void BeginScope(const std::string& name);
    void EndScope();

    // Data retrieval
    const FrameStats& GetFrameStats(uint32_t frame) const;
    const std::vector<FrameStats>& GetFrameHistory() const { return frame_history_; }
    json ToJSON(int max_frames = -1) const;
    json GetCurrentFrameJSON() const;

    // Statistics
    double GetAverageFrameTime() const;
    double GetAverageMarkerTime(const std::string& name) const;
    double GetMaxMarkerTime(const std::string& name) const;
    double GetMinMarkerTime(const std::string& name) const;

    // Configuration
    void SetMaxFrameHistory(size_t max_frames) { max_frame_history_ = max_frames; }
    void SetEnabled(bool enabled) { enabled_ = enabled; }
    bool IsEnabled() const { return enabled_; }

    void Clear();

private:
    Profiler();
    ~Profiler() = default;

    std::vector<FrameStats> frame_history_;
    std::stack<std::shared_ptr<Marker>> scope_stack_;
    uint32_t current_frame_;
    size_t max_frame_history_;
    bool enabled_;
    mutable std::mutex mutex_;

    void PushMarker(const std::string& name);
    void PopMarker();
};

/**
 * @brief RAII scope for automatic profiling
 * 
 * Usage:
 *   {
 *       ScopedProfile profile("MyFunction");
 *       // ... code to profile ...
 *   } // Automatically ends scope
 */
class ScopedProfile
{
public:
    explicit ScopedProfile(const std::string& name)
        : name_(name)
    {
        if (Profiler::Instance().IsEnabled())
        {
            Profiler::Instance().BeginScope(name);
        }
    }

    ~ScopedProfile()
    {
        if (Profiler::Instance().IsEnabled())
        {
            Profiler::Instance().EndScope();
        }
    }

private:
    std::string name_;
};

// Convenience macro for scoped profiling
#define SCOPED_PROFILE(name) ScopedProfile __profile_scope(name)
#define PROFILE_SCOPE(name) SCOPED_PROFILE(name)

/**
 * @brief GPU Profiler using GL_ARB_query_buffer_object and GL_KHR_debug
 * 
 * Tracks GPU timing using hardware queries and debug labels.
 * Thread-safe with automatic GPU stall prevention.
 */
class GPUProfiler
{
public:
    struct GPUMarker
    {
        std::string name;
        uint32_t query_id;
        double time_ms;
        bool result_ready;
    };

    struct GPUFrameStats
    {
        uint32_t frame_number;
        std::vector<GPUMarker> markers;
        double gpu_time_ms;
        std::chrono::high_resolution_clock::time_point timestamp;
    };

    static GPUProfiler& Instance();

    // Frame management
    void BeginFrame();
    void EndFrame();
    uint32_t GetCurrentFrame() const { return current_frame_; }

    // Marker management
    void BeginMarker(const std::string& name, const glm::vec4& color = glm::vec4(1.0f));
    void EndMarker();
    void InsertMarker(const std::string& name, const glm::vec4& color = glm::vec4(1.0f));

    // Data retrieval
    const GPUFrameStats& GetFrameStats(uint32_t frame) const;
    const std::vector<GPUFrameStats>& GetFrameHistory() const { return frame_history_; }
    json ToJSON(int max_frames = -1) const;
    json GetCurrentFrameJSON() const;

    // Statistics
    double GetAverageGPUTime() const;
    double GetAverageMarkerTime(const std::string& name) const;

    // Configuration
    void SetMaxFrameHistory(size_t max_frames) { max_frame_history_ = max_frames; }
    void SetEnabled(bool enabled) { enabled_ = enabled; }
    bool IsEnabled() const { return enabled_; }

    void Clear();

private:
    GPUProfiler();
    ~GPUProfiler();

    std::vector<GPUFrameStats> frame_history_;
    std::stack<std::string> marker_stack_;
    uint32_t current_frame_;
    size_t max_frame_history_;
    bool enabled_;
    mutable std::mutex mutex_;

    std::unordered_map<std::string, uint32_t> query_map_;
    std::vector<uint32_t> query_pool_;

    uint32_t GetOrCreateQuery();
    void ReleaseQuery(uint32_t query_id);
};

/**
 * @brief RAII scope for automatic GPU profiling
 */
class ScopedGPUProfile
{
public:
    explicit ScopedGPUProfile(const std::string& name, const glm::vec4& color = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f))
        : name_(name)
    {
        if (GPUProfiler::Instance().IsEnabled())
        {
            GPUProfiler::Instance().BeginMarker(name, color);
        }
    }

    ~ScopedGPUProfile()
    {
        if (GPUProfiler::Instance().IsEnabled())
        {
            GPUProfiler::Instance().EndMarker();
        }
    }

private:
    std::string name_;
};

// Convenience macro for GPU scoped profiling
#define SCOPED_GPU_PROFILE(name) ScopedGPUProfile __gpu_profile_scope(name)
#define SCOPED_GPU_PROFILE_COLOR(name, color) ScopedGPUProfile __gpu_profile_scope(name, color)
#define PROFILE_GPU(name) SCOPED_GPU_PROFILE(name)
#define PROFILE_GPU_COLOR(name, color) SCOPED_GPU_PROFILE_COLOR(name, color)

/**
 * @brief Combined profiler combining CPU and GPU metrics
 */
class PerformanceMonitor
{
public:
    struct FrameMetrics
    {
        uint32_t frame_number;
        double cpu_time_ms;
        double gpu_time_ms;
        double total_time_ms;
        double fps;
        std::chrono::high_resolution_clock::time_point timestamp;
    };

    static PerformanceMonitor& Instance();

    void Update();
    const std::vector<FrameMetrics>& GetMetricsHistory() const { return metrics_; }
    const FrameMetrics& GetCurrentMetrics() const { return metrics_.back(); }
    
    double GetAverageFPS() const;
    double GetAverageCPUTime() const;
    double GetAverageGPUTime() const;

    json ToJSON(int max_frames = -1) const;

    void SetMaxFrameHistory(size_t max_frames)
    {
        Profiler::Instance().SetMaxFrameHistory(max_frames);
        GPUProfiler::Instance().SetMaxFrameHistory(max_frames);
    }

    void SetEnabled(bool enabled)
    {
        Profiler::Instance().SetEnabled(enabled);
        GPUProfiler::Instance().SetEnabled(enabled);
    }

    void Clear()
    {
        Profiler::Instance().Clear();
        GPUProfiler::Instance().Clear();
        metrics_.clear();
    }

private:
    PerformanceMonitor();
    std::vector<FrameMetrics> metrics_;
    size_t max_frame_history_ = 600;  // ~10 seconds at 60 FPS
};
