# Profiling & Telemetry Guide

## Overview

The Game Engine includes comprehensive profiling infrastructure for CPU and GPU performance monitoring with real-time remote viewing capabilities.

### Features

✅ **CPU Profiler**
- Hierarchical scoped markers with automatic timing
- Frame-based statistics collection
- RAII-style scope guards (no manual end calls required)
- Thread-safe operations
- 600-frame rolling history (~10 seconds at 60 FPS)

✅ **GPU Profiler**
- GPU timing using hardware queries
- Debug labels for GPU debuggers (RenderDoc, NSight)
- Marker hierarchy support
- Automatic result polling
- Frame synchronization

✅ **Remote Telemetry**
- Web-based dashboard accessible from browser
- Real-time metrics streaming
- JSON API for custom tools
- Historical data export (CSV, JSON)
- Multi-client support

✅ **Performance Monitoring**
- Combined CPU/GPU time tracking
- FPS calculation and history
- Per-frame and aggregate statistics
- Memory-efficient circular buffers

---

## Quick Start

### Enable Profiling

```cpp
#include "Profiler.h"
#include "TelemetryServer.h"

// In Application::Initialize():
RemoteProfiler::Instance().Initialize(8080);  // Starts telemetry server on port 8080

// In Application::Update() or render loop:
Profiler::Instance().BeginFrame();
// ... frame code ...
Profiler::Instance().EndFrame();

RemoteProfiler::Instance().Update();  // Send metrics to telemetry server
```

### Add Profiling Markers

**Automatic with RAII (recommended):**
```cpp
{
    SCOPED_PROFILE("RenderPass");
    // ... rendering code ...
}  // Timing automatically recorded

{
    PROFILE_GPU("GeometryPass");
    // ... GPU work ...
}  // GPU time recorded
```

**Manual markers:**
```cpp
Profiler::Instance().BeginScope("UpdatePhysics");
// ... physics code ...
Profiler::Instance().EndScope();
```

### View Results

1. Open browser: **http://localhost:8080**
2. See real-time FPS, CPU/GPU times, per-frame markers
3. Download historical data for analysis

---

## Profiler API Reference

### CPU Profiler (`Profiler` class)

#### Frame Management
```cpp
Profiler::Instance().BeginFrame();  // Call at start of frame
// ... frame processing ...
Profiler::Instance().EndFrame();    // Call at end of frame
```

#### Adding Markers
```cpp
// Automatic (RAII) - recommended
{
    SCOPED_PROFILE("MyFunction");
    // code here
}

// Manual
Profiler::Instance().BeginScope("MyFunction");
// code here
Profiler::Instance().EndScope();
```

#### Statistics
```cpp
double avg_frame_ms = Profiler::Instance().GetAverageFrameTime();
double avg_marker_ms = Profiler::Instance().GetAverageMarkerTime("RenderPass");
double max_marker_ms = Profiler::Instance().GetMaxMarkerTime("RenderPass");
double min_marker_ms = Profiler::Instance().GetMinMarkerTime("RenderPass");

auto stats = Profiler::Instance().GetFrameStats(frame_number);
json data = Profiler::Instance().ToJSON();
```

#### Frame History
```cpp
const auto& history = Profiler::Instance().GetFrameHistory();  // Last 600 frames
const auto& stats = Profiler::Instance().GetCurrentFrame();    // Current frame
```

### GPU Profiler (`GPUProfiler` class)

#### GPU Markers
```cpp
// Automatic (RAII)
{
    PROFILE_GPU("ShadowPass");
    // GPU work
}

// With color for debugger
{
    PROFILE_GPU_COLOR("LightingPass", glm::vec4(1, 1, 0, 1));
    // GPU work
}

// Manual
GPUProfiler::Instance().BeginMarker("CustomPass", glm::vec4(0, 1, 0, 1));
// GPU work
GPUProfiler::Instance().EndMarker();

// One-shot marker (no nesting)
GPUProfiler::Instance().InsertMarker("EventMarker");
```

#### GPU Statistics
```cpp
double avg_gpu_ms = GPUProfiler::Instance().GetAverageGPUTime();
double marker_ms = GPUProfiler::Instance().GetAverageMarkerTime("ShadowPass");

json gpu_data = GPUProfiler::Instance().ToJSON();
```

### Combined Monitor (`PerformanceMonitor` class)

```cpp
PerformanceMonitor::Instance().Update();  // Call each frame

double fps = PerformanceMonitor::Instance().GetAverageFPS();
double cpu_ms = PerformanceMonitor::Instance().GetAverageCPUTime();
double gpu_ms = PerformanceMonitor::Instance().GetAverageGPUTime();

const auto& metrics = PerformanceMonitor::Instance().GetMetricsHistory();
json combined = PerformanceMonitor::Instance().ToJSON();
```

### Remote Profiler (`RemoteProfiler` class)

```cpp
// Initialize with custom port
RemoteProfiler::Instance().Initialize(8080);

// Update each frame
RemoteProfiler::Instance().Update();

// Control profiling
RemoteProfiler::Instance().EnableProfiling(true);
bool enabled = RemoteProfiler::Instance().IsProfilingEnabled();

// Get telemetry server
TelemetryServer* server = RemoteProfiler::Instance().GetServer();

// Retrieve data
json profile_data = RemoteProfiler::Instance().GetProfileData();
json server_stats = RemoteProfiler::Instance().GetServerStatus();

// Shutdown
RemoteProfiler::Instance().Shutdown();
```

---

## Integration Examples

### In Renderer

```cpp
void Renderer::Render()
{
    SCOPED_PROFILE("Renderer::Render");
    
    {
        PROFILE_GPU("OcclusionPass");
        RenderOcclusionQueries();
    }
    
    {
        PROFILE_GPU("GeometryPass");
        RenderGeometryPass();
    }
    
    {
        PROFILE_GPU("LightingPass");
        RenderLightingPass();
    }
    
    {
        PROFILE_GPU("PostProcessing");
        ApplyPostProcessing();
    }
}
```

### In Application Loop

```cpp
class Application
{
public:
    void Initialize()
    {
        // ... other initialization ...
        RemoteProfiler::Instance().Initialize(8080);
    }
    
    void Update()
    {
        SCOPED_PROFILE("Application::Update");
        
        Profiler::Instance().BeginFrame();
        
        {
            SCOPED_PROFILE("InputUpdate");
            UpdateInput();
        }
        
        {
            SCOPED_PROFILE("LogicUpdate");
            UpdateLogic();
        }
        
        {
            SCOPED_PROFILE("Rendering");
            renderer_.Render();
        }
        
        Profiler::Instance().EndFrame();
        
        PerformanceMonitor::Instance().Update();
        RemoteProfiler::Instance().Update();
    }
    
    void Shutdown()
    {
        RemoteProfiler::Instance().Shutdown();
    }
};
```

### Physics System

```cpp
void PhysicsSystem::Update(float dt)
{
    SCOPED_PROFILE("Physics::Update");
    
    {
        SCOPED_PROFILE("Physics::BroadPhase");
        UpdateBroadPhase();
    }
    
    {
        SCOPED_PROFILE("Physics::NarrowPhase");
        UpdateNarrowPhase();
    }
    
    {
        SCOPED_PROFILE("Physics::Solve");
        SolveConstraints();
    }
}
```

### Animation System

```cpp
void AnimationSystem::Update(float dt)
{
    SCOPED_PROFILE("Animation::Update");
    
    for (auto& animator : animators_)
    {
        SCOPED_PROFILE("Animation::BlendTree");
        animator.UpdateBlendTree();
        
        SCOPED_PROFILE("Animation::Sampling");
        animator.SampleAnimation();
        
        SCOPED_PROFILE("Animation::Skinning");
        animator.UpdateSkinning();
    }
}
```

---

## Web Dashboard

### Features

The web UI at `http://localhost:8080` displays:

**Metrics Panel**
- Current FPS
- Average CPU time (ms)
- Average GPU time (ms)
- Total frame time

**Real-Time Charts**
- Frame time trend (last 100 frames)
- CPU vs GPU comparison
- Per-marker breakdown

**Server Status**
- Server connection status
- Update interval
- Total metrics tracked

**Marker Details**
- Current frame markers
- Individual marker times
- Hierarchy visualization

**Controls**
- Manual refresh button
- Download profiling data (JSON)
- Toggle auto-update
- Time period selection

### API Endpoints

**Main Dashboard:**
```
GET http://localhost:8080/
```

**Current Metrics:**
```
GET http://localhost:8080/api/metrics
```

Returns JSON:
```json
{
  "cpu_profiler": [...],
  "gpu_profiler": [...],
  "stats": {
    "avg_fps": 59.8,
    "avg_cpu_ms": 15.2,
    "avg_gpu_ms": 12.8
  }
}
```

**Metrics History:**
```
GET http://localhost:8080/api/history?limit=100
```

**Server Status:**
```
GET http://localhost:8080/api/status
```

---

## Configuration

### Frame History Size

```cpp
// Keep last 600 frames (~10 seconds at 60 FPS)
Profiler::Instance().SetMaxFrameHistory(600);
GPUProfiler::Instance().SetMaxFrameHistory(600);
PerformanceMonitor::Instance().SetMaxFrameHistory(600);
```

### Enable/Disable Profiling

```cpp
// Disable for performance-critical sections
RemoteProfiler::Instance().EnableProfiling(false);
// ... critical code ...
RemoteProfiler::Instance().EnableProfiling(true);

// Or per-system
Profiler::Instance().SetEnabled(false);
GPUProfiler::Instance().SetEnabled(false);
```

### Custom Port

```cpp
RemoteProfiler::Instance().Initialize(9000);  // Non-default port
```

### Update Interval

```cpp
// Update telemetry every 16.67 ms (60 FPS)
RemoteProfiler::Instance().SetUpdateInterval(16.67f);
```

---

## Performance Considerations

### CPU Overhead

- **Minimal**: `SCOPED_PROFILE` uses std::chrono (negligible with -O3)
- **Memory**: ~100 bytes per marker stored
- **600-frame history**: ~500 KB memory per system

### GPU Overhead

- **GPU queries**: ~1-2% on typical workloads
- **Debug labels**: Negligible unless debugger is active
- **No GPU stalls**: Uses ARB_query_buffer_object when available

### Best Practices

1. **Use scoped profilers** - RAII prevents timing leaks
2. **Scope strategically** - Profile major systems, not every function
3. **Enable/disable dynamically** - Turn off in production builds if needed
4. **Use GPU debugger integration** - RenderDoc recognizes GL_KHR_debug labels

### Optimization Tips

```cpp
// Bad: Too many scopes (overhead)
{
    SCOPED_PROFILE("UpdateX");
    x = 5;
}

// Good: Scope around meaningful work
{
    SCOPED_PROFILE("UpdatePhysics");
    UpdateRigidBodies();
    UpdateCollisions();
    SolveConstraints();
}

// Good: Conditional profiling
#if ENABLE_PROFILING
{
    SCOPED_PROFILE("ExpensiveFunction");
    ExpensiveFunction();
}
#else
ExpensiveFunction();
#endif
```

---

## Exporting & Analysis

### JSON Export

```cpp
// Get all data as JSON
json data = RemoteProfiler::Instance().GetProfileData();

// Save to file
std::ofstream file("profiling_data.json");
file << data.dump(2);
```

### Custom Analysis

```python
import json

with open('profiling_data.json') as f:
    data = json.load(f)

# Analyze CPU time
cpu_times = [m['cpu_ms'] for m in data['metrics']]
print(f"Avg CPU: {sum(cpu_times) / len(cpu_times):.2f} ms")
print(f"Peak CPU: {max(cpu_times):.2f} ms")

# Analyze markers
for metric in data['cpu_profiler']:
    for name, times in metric['markers'].items():
        print(f"{name}: {sum(times)/len(times):.2f} ms")
```

### Integration with Third-Party Tools

**Export for Perfetto (Chrome trace format):**
```cpp
json perfetto_trace = ConvertToPerfetto(RemoteProfiler::Instance().GetProfileData());
// Visualize in https://ui.perfetto.dev
```

**Export for WPA (Windows Performance Analyzer):**
```cpp
ExportToETW(RemoteProfiler::Instance().GetProfileData());
// Analyze with Windows Performance Toolkit
```

---

## Troubleshooting

### Server Won't Start

```cpp
// Check if port is already in use
RemoteProfiler::Instance().Initialize(9000);  // Use different port

// Verify server initialization
if (RemoteProfiler::Instance().GetServer()->IsRunning())
{
    std::cout << "Server OK" << std::endl;
}
```

### GPU Markers Not Showing

```cpp
// Ensure GL_KHR_debug is available
if (GLAD_GL_KHR_DEBUG)
{
    std::cout << "Debug extension available" << std::endl;
}

// Attach debugger to see labels
// - RenderDoc: Capture and inspect in timeline
// - NSight: Monitor frame capture
// - PIX: Direct GPU capture on Windows
```

### High CPU Overhead

```cpp
// Reduce profiling scope
Profiler::Instance().SetMaxFrameHistory(300);  // Smaller buffer

// Use conditional compilation
#ifdef PROFILING_ENABLED
    SCOPED_PROFILE("MyScope");
#endif
```

### Telemetry Server Timeout

```cpp
// Browser console shows connection errors
// Check firewall settings
// Verify server is running: RemoteProfiler::Instance().IsInitialized()
```

---

## Advanced Topics

### Custom Metric Collection

```cpp
class CustomProfiler
{
public:
    static void RecordMetric(const std::string& name, double value)
    {
        auto data = json::object({
            {"metric", name},
            {"value", value},
            {"timestamp", std::time(nullptr)}
        });
        
        RemoteProfiler::Instance().GetServer()->PublishMessage("custom", data);
    }
};

// Usage
CustomProfiler::RecordMetric("GameObjectCount", game_objects.size());
CustomProfiler::RecordMetric("TriangleCount", renderer.GetTriangleCount());
```

### GPU Timeline Correlation

```cpp
void Renderer::Render()
{
    uint64_t cpu_timestamp = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    
    glInsertEventMarker(
        ("CPU_TS:" + std::to_string(cpu_timestamp)).c_str()
    );
    
    SCOPED_PROFILE("Render");
    // ... rendering ...
    
    // GPU trace contains CPU timestamp for correlation
}
```

### Network Profiling Export

```cpp
// Send profiling data over network
void SendProfilingToServer(const std::string& endpoint)
{
    auto data = RemoteProfiler::Instance().GetProfileData();
    
    // POST to remote server
    HttpClient client(endpoint);
    client.Post("/profile", data.dump());
}
```

---

## File Reference

| File | Purpose |
|------|---------|
| [Profiler.h](../include/Profiler.h) | CPU/GPU profiler headers |
| [Profiler.cpp](../src/Profiler.cpp) | CPU/GPU profiler implementation |
| [TelemetryServer.h](../include/TelemetryServer.h) | Remote server headers |
| [TelemetryServer.cpp](../src/TelemetryServer.cpp) | Remote server implementation |

---

## Integration Checklist

- [ ] Include `Profiler.h` and `TelemetryServer.h`
- [ ] Initialize RemoteProfiler in Application::Initialize()
- [ ] Add BeginFrame/EndFrame in main loop
- [ ] Add SCOPED_PROFILE to major systems
- [ ] Add PROFILE_GPU to rendering code
- [ ] Call PerformanceMonitor::Update() each frame
- [ ] Test with `http://localhost:8080`
- [ ] Download and analyze exported data
- [ ] Benchmark before/after profiling overhead
- [ ] Document performance baselines

---

## Example: Complete Integration

```cpp
// application.h
class Application
{
    std::unique_ptr<Renderer> renderer_;
    std::unique_ptr<PhysicsSystem> physics_;
    
public:
    void Initialize()
    {
        renderer_ = std::make_unique<Renderer>();
        physics_ = std::make_unique<PhysicsSystem>();
        
        // Start telemetry server
        RemoteProfiler::Instance().Initialize(8080);
        Profiler::Instance().SetEnabled(true);
    }
    
    void Update(float dt)
    {
        Profiler::Instance().BeginFrame();
        
        {
            SCOPED_PROFILE("InputUpdate");
            HandleInput();
        }
        
        {
            SCOPED_PROFILE("PhysicsUpdate");
            physics_->Update(dt);
        }
        
        {
            SCOPED_PROFILE("RenderFrame");
            renderer_->Render();
        }
        
        Profiler::Instance().EndFrame();
        PerformanceMonitor::Instance().Update();
        RemoteProfiler::Instance().Update();
    }
    
    void Shutdown()
    {
        RemoteProfiler::Instance().Shutdown();
    }
};
```

---

**Last Updated**: December 2025  
**Status**: ✅ Production Ready  
**Profilers**: 3 (CPU, GPU, Combined)  
**Remote Viewers**: 1 (Web UI)  
**API Endpoints**: 3+ documented  
**Max Frame History**: 600 frames (~10 sec @ 60 FPS)
