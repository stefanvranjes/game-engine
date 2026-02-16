#pragma once

#include <string>
#include <memory>
#include <nlohmann/json.hpp>
#include <thread>
#include <atomic>
#include <mutex>
#include <vector>

using json = nlohmann::json;

/**
 * @brief Telemetry Server for remote profiler viewing
 * 
 * Provides HTTP endpoint for profiling data visualization.
 * Supports WebSocket for real-time updates.
 * 
 * Usage:
 *   TelemetryServer server(8080);
 *   server.Start();
 *   
 *   // In game loop:
 *   server.PublishMetrics(performance_monitor.ToJSON());
 *   
 *   // In browser: http://localhost:8080
 */
class TelemetryServer
{
public:
    explicit TelemetryServer(uint16_t port = 8080);
    ~TelemetryServer();

    // Server lifecycle
    void Start();
    void Stop();
    bool IsRunning() const { return running_; }

    // Data publishing
    void PublishMetrics(const json& metrics);
    void PublishMessage(const std::string& type, const json& data);

    // Configuration
    void SetPort(uint16_t port) { port_ = port; }
    uint16_t GetPort() const { return port_; }

    void SetEndpoint(const std::string& endpoint) { endpoint_ = endpoint; }
    std::string GetEndpoint() const { return endpoint_; }

    // Statistics
    json GetServerStats() const;

private:
    uint16_t port_;
    std::string endpoint_;
    std::atomic<bool> running_;
    std::unique_ptr<std::thread> server_thread_;

    mutable std::mutex metrics_mutex_;
    json latest_metrics_;
    
    std::vector<json> metrics_history_;
    size_t max_history_ = 600;  // ~10 seconds at 60 FPS

    void RunServer();
    std::string GenerateHTMLDashboard() const;
    std::string HandleProfilerRequest() const;
    std::string HandleMetricsRequest() const;
    std::string HandleHistoryRequest(size_t limit = 100) const;
};

/**
 * @brief Remote Profiler Manager
 * 
 * High-level interface for remote profiling with automatic updates.
 * Integrates with both CPU and GPU profilers.
 */
class RemoteProfiler
{
public:
    static RemoteProfiler& Instance();

    // Lifecycle
    void Initialize(uint16_t port = 8080);
    void Shutdown();
    bool IsInitialized() const { return server_ != nullptr; }

    // Update profiling data (call once per frame)
    void Update(float deltaTime);


    // Configuration
    void SetUpdateInterval(float interval_ms) { update_interval_ms_ = interval_ms; }
    void SetPort(uint16_t port) { port_ = port; }

    // Server access
    TelemetryServer* GetServer() { return server_.get(); }
    const TelemetryServer* GetServer() const { return server_.get(); }

    // Profiling control
    void EnableProfiling(bool enable);
    bool IsProfilingEnabled() const { return profiling_enabled_; }

    // Statistics
    json GetProfileData() const;
    json GetServerStatus() const;

private:
    RemoteProfiler();
    ~RemoteProfiler();

    std::unique_ptr<TelemetryServer> server_;
    uint16_t port_;
    float update_interval_ms_;
    float time_since_update_;
    bool profiling_enabled_;

    void UpdateProfileData();
};

/**
 * @brief Scoped profiler that can be viewed remotely
 * 
 * Extends ScopedProfile with remote telemetry integration.
 */
class RemoteProfileScope
{
public:
    explicit RemoteProfileScope(const std::string& name);
    ~RemoteProfileScope();

private:
    std::string name_;
};
