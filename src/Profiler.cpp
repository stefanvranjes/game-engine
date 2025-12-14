#include "Profiler.h"
#include <glad/glad.h>
#include <glm/glm.hpp>
#include <algorithm>
#include <iostream>

// ============================================================================
// Profiler Implementation
// ============================================================================

Profiler& Profiler::Instance()
{
    static Profiler instance;
    return instance;
}

Profiler::Profiler()
    : current_frame_(0),
      max_frame_history_(600),  // ~10 seconds at 60 FPS
      enabled_(true)
{
}

void Profiler::BeginFrame()
{
    std::lock_guard<std::mutex> lock(mutex_);
    
    frame_history_.emplace_back();
    auto& frame = frame_history_.back();
    frame.frame_number = current_frame_;
    frame.frame_start = std::chrono::high_resolution_clock::now();
    
    // Create root marker
    frame.root_marker = std::make_shared<Marker>();
    frame.root_marker->name = "Frame";
    frame.root_marker->start_time = frame.frame_start;
    frame.root_marker->depth = 0;
    frame.root_marker->frame = current_frame_;
    
    scope_stack_.push(frame.root_marker);
}

void Profiler::EndFrame()
{
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (frame_history_.empty()) return;
    
    auto& frame = frame_history_.back();
    frame.frame_end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        frame.frame_end - frame.frame_start);
    frame.frame_time_ms = duration.count() / 1000.0;
    frame.cpu_time_ms = frame.frame_time_ms;
    
    // Finalize root marker
    if (!scope_stack_.empty())
    {
        auto root = scope_stack_.top();
        scope_stack_.pop();
        root->end_time = frame.frame_end;
    }
    
    // Keep only recent frame history
    if (frame_history_.size() > max_frame_history_)
    {
        frame_history_.erase(frame_history_.begin());
    }
    
    current_frame_++;
}

void Profiler::BeginScope(const std::string& name)
{
    if (!enabled_) return;
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (frame_history_.empty()) return;
    
    auto marker = std::make_shared<Marker>();
    marker->name = name;
    marker->start_time = std::chrono::high_resolution_clock::now();
    marker->frame = current_frame_;
    
    if (!scope_stack_.empty())
    {
        marker->depth = scope_stack_.top()->depth + 1;
        scope_stack_.top()->children.push_back(marker);
    }
    else
    {
        marker->depth = 1;
    }
    
    scope_stack_.push(marker);
}

void Profiler::EndScope()
{
    if (!enabled_) return;
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (scope_stack_.empty()) return;
    
    auto marker = scope_stack_.top();
    scope_stack_.pop();
    
    marker->end_time = std::chrono::high_resolution_clock::now();
    
    if (!frame_history_.empty())
    {
        auto& frame = frame_history_.back();
        auto duration = marker->GetDurationMs();
        
        if (frame.marker_times.find(marker->name) == frame.marker_times.end())
        {
            frame.marker_times[marker->name] = std::vector<double>();
        }
        frame.marker_times[marker->name].push_back(duration);
    }
}

const Profiler::FrameStats& Profiler::GetFrameStats(uint32_t frame) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (frame >= current_frame_ || frame_history_.empty())
    {
        static FrameStats empty;
        return empty;
    }
    
    size_t idx = frame_history_.size() - (current_frame_ - frame);
    if (idx < frame_history_.size())
    {
        return frame_history_[idx];
    }
    
    static FrameStats empty;
    return empty;
}

double Profiler::GetAverageFrameTime() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (frame_history_.empty()) return 0.0;
    
    double total = 0.0;
    for (const auto& frame : frame_history_)
    {
        total += frame.frame_time_ms;
    }
    return total / frame_history_.size();
}

double Profiler::GetAverageMarkerTime(const std::string& name) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    
    double total = 0.0;
    size_t count = 0;
    
    for (const auto& frame : frame_history_)
    {
        auto it = frame.marker_times.find(name);
        if (it != frame.marker_times.end())
        {
            for (double time : it->second)
            {
                total += time;
                count++;
            }
        }
    }
    
    return count > 0 ? total / count : 0.0;
}

double Profiler::GetMaxMarkerTime(const std::string& name) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    
    double max_time = 0.0;
    
    for (const auto& frame : frame_history_)
    {
        auto it = frame.marker_times.find(name);
        if (it != frame.marker_times.end())
        {
            for (double time : it->second)
            {
                max_time = std::max(max_time, time);
            }
        }
    }
    
    return max_time;
}

double Profiler::GetMinMarkerTime(const std::string& name) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    
    double min_time = std::numeric_limits<double>::max();
    bool found = false;
    
    for (const auto& frame : frame_history_)
    {
        auto it = frame.marker_times.find(name);
        if (it != frame.marker_times.end())
        {
            for (double time : it->second)
            {
                min_time = std::min(min_time, time);
                found = true;
            }
        }
    }
    
    return found ? min_time : 0.0;
}

json Profiler::ToJSON() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    
    json data = json::array();
    
    for (const auto& frame : frame_history_)
    {
        json frame_data;
        frame_data["frame"] = frame.frame_number;
        frame_data["time_ms"] = frame.frame_time_ms;
        
        json markers = json::object();
        for (const auto& [name, times] : frame.marker_times)
        {
            json marker_times = json::array();
            for (double t : times)
            {
                marker_times.push_back(t);
            }
            markers[name] = marker_times;
        }
        frame_data["markers"] = markers;
        
        data.push_back(frame_data);
    }
    
    return data;
}

json Profiler::GetCurrentFrameJSON() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (frame_history_.empty()) return json::object();
    
    const auto& frame = frame_history_.back();
    json frame_data;
    frame_data["frame"] = frame.frame_number;
    frame_data["time_ms"] = frame.frame_time_ms;
    
    json markers = json::object();
    for (const auto& [name, times] : frame.marker_times)
    {
        if (!times.empty())
        {
            markers[name] = times.back();  // Last measurement
        }
    }
    frame_data["markers"] = markers;
    
    return frame_data;
}

void Profiler::Clear()
{
    std::lock_guard<std::mutex> lock(mutex_);
    frame_history_.clear();
    while (!scope_stack_.empty())
    {
        scope_stack_.pop();
    }
    current_frame_ = 0;
}

// ============================================================================
// GPUProfiler Implementation
// ============================================================================

GPUProfiler& GPUProfiler::Instance()
{
    static GPUProfiler instance;
    return instance;
}

GPUProfiler::GPUProfiler()
    : current_frame_(0),
      max_frame_history_(600),
      enabled_(true)
{
}

GPUProfiler::~GPUProfiler()
{
    // Clean up GPU queries
    if (!query_pool_.empty())
    {
        glDeleteQueries(query_pool_.size(), query_pool_.data());
    }
}

void GPUProfiler::BeginFrame()
{
    std::lock_guard<std::mutex> lock(mutex_);
    
    frame_history_.emplace_back();
    auto& frame = frame_history_.back();
    frame.frame_number = current_frame_;
    frame.timestamp = std::chrono::high_resolution_clock::now();
}

void GPUProfiler::EndFrame()
{
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (frame_history_.empty()) return;
    
    auto& frame = frame_history_.back();
    
    // Calculate total GPU time
    double gpu_time = 0.0;
    for (const auto& marker : frame.markers)
    {
        gpu_time += marker.time_ms;
    }
    frame.gpu_time_ms = gpu_time;
    
    // Keep only recent frame history
    if (frame_history_.size() > max_frame_history_)
    {
        frame_history_.erase(frame_history_.begin());
    }
    
    current_frame_++;
}

void GPUProfiler::BeginMarker(const std::string& name, const glm::vec4& color)
{
    if (!enabled_) return;
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (frame_history_.empty()) return;
    
    // Use glDebugMessageInsert if GL_KHR_debug is available
    if (GLAD_GL_KHR_debug)
    {
        uint32_t rgb = 
            (uint32_t)(color.r * 255) << 16 |
            (uint32_t)(color.g * 255) << 8 |
            (uint32_t)(color.b * 255);
        
        glDebugMessageInsert(GL_DEBUG_SOURCE_APPLICATION, GL_DEBUG_TYPE_MARKER,
                           0, GL_DEBUG_SEVERITY_NOTIFICATION,
                           name.length(), name.c_str());
    }
    
    marker_stack_.push(name);
}

void GPUProfiler::EndMarker()
{
    if (!enabled_) return;
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (marker_stack_.empty()) return;
    
    std::string name = marker_stack_.top();
    marker_stack_.pop();
    
    if (frame_history_.empty()) return;
    
    auto& frame = frame_history_.back();
    
    GPUMarker marker;
    marker.name = name;
    marker.query_id = 0;  // Simplified - would use actual queries in production
    marker.time_ms = 0.0;
    marker.result_ready = true;
    
    frame.markers.push_back(marker);
}

void GPUProfiler::InsertMarker(const std::string& name, const glm::vec4& color)
{
    if (!enabled_) return;
    
    if (GLAD_GL_KHR_debug)
    {
        glDebugMessageInsert(GL_DEBUG_SOURCE_APPLICATION, GL_DEBUG_TYPE_MARKER,
                           0, GL_DEBUG_SEVERITY_NOTIFICATION,
                           name.length(), name.c_str());
    }
}

const GPUProfiler::GPUFrameStats& GPUProfiler::GetFrameStats(uint32_t frame) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (frame >= current_frame_ || frame_history_.empty())
    {
        static GPUFrameStats empty;
        return empty;
    }
    
    size_t idx = frame_history_.size() - (current_frame_ - frame);
    if (idx < frame_history_.size())
    {
        return frame_history_[idx];
    }
    
    static GPUFrameStats empty;
    return empty;
}

double GPUProfiler::GetAverageGPUTime() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (frame_history_.empty()) return 0.0;
    
    double total = 0.0;
    for (const auto& frame : frame_history_)
    {
        total += frame.gpu_time_ms;
    }
    return total / frame_history_.size();
}

double GPUProfiler::GetAverageMarkerTime(const std::string& name) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    
    double total = 0.0;
    size_t count = 0;
    
    for (const auto& frame : frame_history_)
    {
        for (const auto& marker : frame.markers)
        {
            if (marker.name == name)
            {
                total += marker.time_ms;
                count++;
            }
        }
    }
    
    return count > 0 ? total / count : 0.0;
}

json GPUProfiler::ToJSON() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    
    json data = json::array();
    
    for (const auto& frame : frame_history_)
    {
        json frame_data;
        frame_data["frame"] = frame.frame_number;
        frame_data["gpu_time_ms"] = frame.gpu_time_ms;
        
        json markers = json::array();
        for (const auto& marker : frame.markers)
        {
            json marker_data;
            marker_data["name"] = marker.name;
            marker_data["time_ms"] = marker.time_ms;
            marker_data["ready"] = marker.result_ready;
            markers.push_back(marker_data);
        }
        frame_data["markers"] = markers;
        
        data.push_back(frame_data);
    }
    
    return data;
}

json GPUProfiler::GetCurrentFrameJSON() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (frame_history_.empty()) return json::object();
    
    const auto& frame = frame_history_.back();
    json frame_data;
    frame_data["frame"] = frame.frame_number;
    frame_data["gpu_time_ms"] = frame.gpu_time_ms;
    
    json markers = json::array();
    for (const auto& marker : frame.markers)
    {
        json marker_data;
        marker_data["name"] = marker.name;
        marker_data["time_ms"] = marker.time_ms;
        markers.push_back(marker_data);
    }
    frame_data["markers"] = markers;
    
    return frame_data;
}

void GPUProfiler::Clear()
{
    std::lock_guard<std::mutex> lock(mutex_);
    frame_history_.clear();
    while (!marker_stack_.empty())
    {
        marker_stack_.pop();
    }
    current_frame_ = 0;
}

uint32_t GPUProfiler::GetOrCreateQuery()
{
    if (query_pool_.empty())
    {
        uint32_t query;
        glGenQueries(1, &query);
        return query;
    }
    
    uint32_t query = query_pool_.back();
    query_pool_.pop_back();
    return query;
}

void GPUProfiler::ReleaseQuery(uint32_t query_id)
{
    query_pool_.push_back(query_id);
}

// ============================================================================
// PerformanceMonitor Implementation
// ============================================================================

PerformanceMonitor& PerformanceMonitor::Instance()
{
    static PerformanceMonitor instance;
    return instance;
}

PerformanceMonitor::PerformanceMonitor()
{
    metrics_.resize(max_frame_history_);
}

void PerformanceMonitor::Update()
{
    const auto& cpu_frame = Profiler::Instance().GetCurrentFrame();
    const auto& gpu_frame = GPUProfiler::Instance().GetCurrentFrame();
    
    FrameMetrics metrics;
    metrics.frame_number = cpu_frame;
    metrics.timestamp = std::chrono::high_resolution_clock::now();
    
    if (cpu_frame > 0)
    {
        const auto& cpu_stats = Profiler::Instance().GetFrameStats(cpu_frame - 1);
        metrics.cpu_time_ms = cpu_stats.frame_time_ms;
    }
    
    if (gpu_frame > 0)
    {
        const auto& gpu_stats = GPUProfiler::Instance().GetFrameStats(gpu_frame - 1);
        metrics.gpu_time_ms = gpu_stats.gpu_time_ms;
    }
    
    metrics.total_time_ms = std::max(metrics.cpu_time_ms, metrics.gpu_time_ms);
    metrics.fps = metrics.total_time_ms > 0.0 ? 1000.0 / metrics.total_time_ms : 0.0;
    
    metrics_.push_back(metrics);
    
    if (metrics_.size() > max_frame_history_)
    {
        metrics_.erase(metrics_.begin());
    }
}

double PerformanceMonitor::GetAverageFPS() const
{
    if (metrics_.empty()) return 0.0;
    
    double total_fps = 0.0;
    for (const auto& m : metrics_)
    {
        total_fps += m.fps;
    }
    return total_fps / metrics_.size();
}

double PerformanceMonitor::GetAverageCPUTime() const
{
    return Profiler::Instance().GetAverageFrameTime();
}

double PerformanceMonitor::GetAverageGPUTime() const
{
    return GPUProfiler::Instance().GetAverageGPUTime();
}

json PerformanceMonitor::ToJSON() const
{
    json data = json::object();
    
    data["cpu_profiler"] = Profiler::Instance().ToJSON();
    data["gpu_profiler"] = GPUProfiler::Instance().ToJSON();
    
    json metrics = json::array();
    for (const auto& m : metrics_)
    {
        json metric_data;
        metric_data["frame"] = m.frame_number;
        metric_data["cpu_ms"] = m.cpu_time_ms;
        metric_data["gpu_ms"] = m.gpu_time_ms;
        metric_data["total_ms"] = m.total_time_ms;
        metric_data["fps"] = m.fps;
        metrics.push_back(metric_data);
    }
    data["metrics"] = metrics;
    
    data["stats"] = json::object({
        {"avg_fps", GetAverageFPS()},
        {"avg_cpu_ms", GetAverageCPUTime()},
        {"avg_gpu_ms", GetAverageGPUTime()}
    });
    
    return data;
}
