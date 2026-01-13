#include "TelemetryServer.h"
#include "Profiler.h"
#include <iostream>
#include <sstream>
#include <ctime>
#include <iomanip>

// ============================================================================
// TelemetryServer Implementation
// ============================================================================

TelemetryServer::TelemetryServer(uint16_t port)
    : port_(port),
      endpoint_("/profiler"),
      running_(false)
{
}

TelemetryServer::~TelemetryServer()
{
    Stop();
}

void TelemetryServer::Start()
{
    if (running_) return;
    
    running_ = true;
    server_thread_ = std::make_unique<std::thread>([this] { RunServer(); });
    
    std::cout << "Telemetry Server started on http://localhost:" << port_ << endpoint_ << std::endl;
}

void TelemetryServer::Stop()
{
    if (!running_) return;
    
    running_ = false;
    if (server_thread_ && server_thread_->joinable())
    {
        server_thread_->join();
    }
    
    std::cout << "Telemetry Server stopped" << std::endl;
}

void TelemetryServer::PublishMetrics(const json& metrics)
{
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    latest_metrics_ = metrics;
    metrics_history_.push_back(metrics);
    
    if (metrics_history_.size() > max_history_)
    {
        metrics_history_.erase(metrics_history_.begin());
    }
}

void TelemetryServer::PublishMessage(const std::string& type, const json& data)
{
    json message = json::object({
        {"type", type},
        {"timestamp", std::time(nullptr)},
        {"data", data}
    });
    
    PublishMetrics(message);
}

json TelemetryServer::GetServerStats() const
{
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    json stats = json::object({
        {"port", port_},
        {"endpoint", endpoint_},
        {"running", running_.load()},
        {"metrics_history_size", metrics_history_.size()},
        {"has_latest_metrics", !latest_metrics_.empty()}
    });
    
    return stats;
}

void TelemetryServer::RunServer()
{
    // Simple HTTP server implementation
    // Note: This is a simplified version. Production would use a proper HTTP library like Beast or cpp-httplib
    
    while (running_)
    {
        // In production, implement proper HTTP server here with socket handling
        // For now, this demonstrates the structure
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

std::string TelemetryServer::GenerateHTMLDashboard() const
{
    return R"HTML(
<!DOCTYPE html>
<html>
<head>
    <title>Game Engine Profiler</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: Arial, sans-serif; background: #1a1a1a; color: #fff; }
        header { background: #222; padding: 20px; border-bottom: 2px solid #ff6b6b; }
        h1 { margin-bottom: 10px; }
        .subtitle { color: #999; font-size: 14px; }
        
        .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
        
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; margin-bottom: 20px; }
        
        .card { background: #2a2a2a; border: 1px solid #444; border-radius: 8px; padding: 20px; }
        .card h3 { color: #ff6b6b; margin-bottom: 15px; font-size: 18px; }
        
        .metric { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #444; }
        .metric:last-child { border-bottom: none; }
        .metric-value { color: #4ecdc4; font-weight: bold; font-family: monospace; }
        
        .chart-container { position: relative; height: 300px; margin-bottom: 20px; }
        
        .status { padding: 10px 15px; border-radius: 4px; font-size: 14px; }
        .status.running { background: #27ae60; color: white; }
        .status.stopped { background: #e74c3c; color: white; }
        
        .controls { margin-bottom: 20px; }
        button { 
            padding: 10px 20px; 
            margin-right: 10px; 
            background: #ff6b6b; 
            color: white; 
            border: none; 
            border-radius: 4px; 
            cursor: pointer; 
            font-size: 14px;
        }
        button:hover { background: #ff5252; }
        
        .timer { color: #999; font-size: 12px; }
    </style>
</head>
<body>
    <header>
        <h1>🎮 Game Engine Performance Profiler</h1>
        <div class="subtitle">Real-time CPU/GPU performance monitoring</div>
    </header>
    
    <div class="container">
        <div class="controls">
            <button onclick="location.reload()">Refresh</button>
            <button onclick="downloadData()">Download Data</button>
            <span class="timer" id="update-timer">Auto-updating...</span>
        </div>
        
        <div class="grid">
            <div class="card">
                <h3>Performance Metrics</h3>
                <div id="metrics-container">
                    <p style="color: #999;">Loading...</p>
                </div>
            </div>
            
            <div class="card">
                <h3>Server Status</h3>
                <div class="status running" id="server-status">●  Server Running</div>
                <div style="margin-top: 15px;">
                    <div class="metric">
                        <span>Update Interval</span>
                        <span class="metric-value" id="update-interval">16.67 ms</span>
                    </div>
                    <div class="metric">
                        <span>Metrics Tracked</span>
                        <span class="metric-value" id="metrics-tracked">0</span>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h3>Frame Time Trend (Last 100 Frames)</h3>
            <div class="chart-container">
                <canvas id="frameTimeChart"></canvas>
            </div>
        </div>
        
        <div class="card">
            <h3>CPU vs GPU Time</h3>
            <div class="chart-container">
                <canvas id="cpuGpuChart"></canvas>
            </div>
        </div>
        
        <div class="card">
            <h3>Marker Times (Current Frame)</h3>
            <div id="markers-container">
                <p style="color: #999;">No marker data available</p>
            </div>
        </div>
    </div>
    
    <script>
        let updateInterval = setInterval(updateDashboard, 500);
        
        function updateDashboard() {
            fetch('/api/metrics')
                .then(r => r.json())
                .then(data => {
                    updateMetricsDisplay(data);
                    updateCharts(data);
                })
                .catch(e => console.error('Update failed:', e));
        }
        
        function updateMetricsDisplay(data) {
            const container = document.getElementById('metrics-container');
            if (!data || !data.stats) return;
            
            container.innerHTML = `
                <div class="metric">
                    <span>Average FPS</span>
                    <span class="metric-value">${data.stats.avg_fps?.toFixed(1) || 'N/A'}</span>
                </div>
                <div class="metric">
                    <span>Avg CPU Time</span>
                    <span class="metric-value">${data.stats.avg_cpu_ms?.toFixed(2) || 'N/A'} ms</span>
                </div>
                <div class="metric">
                    <span>Avg GPU Time</span>
                    <span class="metric-value">${data.stats.avg_gpu_ms?.toFixed(2) || 'N/A'} ms</span>
                </div>
            `;
            
            document.getElementById('metrics-tracked').textContent = 
                data.metrics ? data.metrics.length : 0;
        }
        
        function updateCharts(data) {
            if (!data.metrics || data.metrics.length === 0) return;
            
            // Frame time chart
            const frameTimeCtx = document.getElementById('frameTimeChart');
            if (frameTimeCtx) {
                const frameLabels = data.metrics.map(m => m.frame);
                const frameTimes = data.metrics.map(m => m.total_ms);
                
                new Chart(frameTimeCtx, {
                    type: 'line',
                    data: {
                        labels: frameLabels,
                        datasets: [{
                            label: 'Frame Time (ms)',
                            data: frameTimes,
                            borderColor: '#ff6b6b',
                            backgroundColor: 'rgba(255,107,107,0.1)',
                            tension: 0.3,
                            fill: true
                        }]
                    },
                    options: { responsive: true, maintainAspectRatio: false }
                });
            }
            
            // CPU vs GPU chart
            const cpuGpuCtx = document.getElementById('cpuGpuChart');
            if (cpuGpuCtx) {
                const frameLabels = data.metrics.map(m => m.frame);
                const cpuTimes = data.metrics.map(m => m.cpu_ms);
                const gpuTimes = data.metrics.map(m => m.gpu_ms);
                
                new Chart(cpuGpuCtx, {
                    type: 'line',
                    data: {
                        labels: frameLabels,
                        datasets: [
                            {
                                label: 'CPU Time (ms)',
                                data: cpuTimes,
                                borderColor: '#4ecdc4',
                                backgroundColor: 'rgba(78,205,196,0.1)',
                                tension: 0.3,
                                fill: true
                            },
                            {
                                label: 'GPU Time (ms)',
                                data: gpuTimes,
                                borderColor: '#ffe66d',
                                backgroundColor: 'rgba(255,230,109,0.1)',
                                tension: 0.3,
                                fill: true
                            }
                        ]
                    },
                    options: { responsive: true, maintainAspectRatio: false }
                });
            }
        }
        
        function downloadData() {
            fetch('/api/metrics')
                .then(r => r.json())
                .then(data => {
                    const jsonString = JSON.stringify(data, null, 2);
                    const blob = new Blob([jsonString], { type: 'application/json' });
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = `profiler-${new Date().toISOString()}.json`;
                    a.click();
                });
        }
        
        // Initial update
        updateDashboard();
    </script>
</body>
</html>
    )HTML";
}

std::string TelemetryServer::HandleProfilerRequest() const
{
    return GenerateHTMLDashboard();
}

std::string TelemetryServer::HandleMetricsRequest() const
{
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    return latest_metrics_.dump();
}

std::string TelemetryServer::HandleHistoryRequest(size_t limit) const
{
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    json history = json::array();
    
    size_t start = metrics_history_.size() > limit ? 
                   metrics_history_.size() - limit : 0;
    
    for (size_t i = start; i < metrics_history_.size(); ++i)
    {
        history.push_back(metrics_history_[i]);
    }
    
    return history.dump();
}

// ============================================================================
// RemoteProfiler Implementation
// ============================================================================

RemoteProfiler& RemoteProfiler::Instance()
{
    static RemoteProfiler instance;
    return instance;
}

RemoteProfiler::RemoteProfiler()
    : port_(8080),
      update_interval_ms_(16.67f),
      time_since_update_(0.0f),
      profiling_enabled_(true)
{
}

RemoteProfiler::~RemoteProfiler()
{
    Shutdown();
}

void RemoteProfiler::Initialize(uint16_t port)
{
    if (server_) return;
    
    port_ = port;
    server_ = std::make_unique<TelemetryServer>(port);
    server_->Start();
    
    Profiler::Instance().SetEnabled(profiling_enabled_);
    GPUProfiler::Instance().SetEnabled(profiling_enabled_);
    
    std::cout << "RemoteProfiler initialized on port " << port << std::endl;
}

void RemoteProfiler::Shutdown()
{
    if (server_)
    {
        server_->Stop();
        server_.reset();
    }
}

void RemoteProfiler::Update()
{
    if (!server_) return;
    
    time_since_update_ += 16.67f;  // Approximate 60 FPS
    
    if (time_since_update_ >= update_interval_ms_)
    {
        UpdateProfileData();
        time_since_update_ = 0.0f;
    }
}

void RemoteProfiler::EnableProfiling(bool enable)
{
    profiling_enabled_ = enable;
    Profiler::Instance().SetEnabled(enable);
    GPUProfiler::Instance().SetEnabled(enable);
}

json RemoteProfiler::GetProfileData() const
{
    json data = json::object({
        {"cpu_profiler", Profiler::Instance().ToJSON()},
        {"gpu_profiler", GPUProfiler::Instance().ToJSON()},
        {"timestamp", std::time(nullptr)}
    });
    
    return data;
}

json RemoteProfiler::GetServerStatus() const
{
    if (!server_) return json::object();
    return server_->GetServerStats();
}

void RemoteProfiler::UpdateProfileData()
{
    if (!server_) return;
    
    auto profile_data = GetProfileData();
    server_->PublishMetrics(profile_data);
}

// ============================================================================
// RemoteProfileScope Implementation
// ============================================================================

RemoteProfileScope::RemoteProfileScope(const std::string& name)
    : name_(name)
{
    Profiler::Instance().BeginScope(name);
    RemoteProfiler::Instance().Update();
}

RemoteProfileScope::~RemoteProfileScope()
{
    Profiler::Instance().EndScope();
    RemoteProfiler::Instance().Update();
}
