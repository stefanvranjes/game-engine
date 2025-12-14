# Profiling Quick Reference

## 30-Second Setup

```cpp
#include "Profiler.h"
#include "TelemetryServer.h"

// In Application::Initialize()
RemoteProfiler::Instance().Initialize(8080);

// In game loop
Profiler::Instance().BeginFrame();
// ... frame code ...
Profiler::Instance().EndFrame();

RemoteProfiler::Instance().Update();

// In browser: http://localhost:8080
```

---

## Common Patterns

### Profile a Function

```cpp
void MyFunction()
{
    SCOPED_PROFILE("MyFunction");
    // ... code ...
}  // Automatically ends scope
```

### Profile a Code Block

```cpp
{
    SCOPED_PROFILE("BlockName");
    // ... code to profile ...
}
```

### Profile GPU Work

```cpp
{
    PROFILE_GPU("ShadowMap");
    // ... GPU code ...
}
```

### Profile with Color (for debugger)

```cpp
{
    PROFILE_GPU_COLOR("GeometryPass", glm::vec4(1, 0, 0, 1));  // Red
    // ... GPU code ...
}
```

### Manual Markers

```cpp
Profiler::Instance().BeginScope("LongOperation");
// ... code ...
Profiler::Instance().EndScope();
```

---

## Viewing Results

| Method | URL | Info |
|--------|-----|------|
| **Web Dashboard** | `http://localhost:8080` | Real-time charts, stats, export |
| **JSON API** | `http://localhost:8080/api/metrics` | Raw metrics data |
| **History API** | `http://localhost:8080/api/history?limit=100` | Last N frames |

---

## Getting Statistics

```cpp
// Frame timing
double avg_frame_ms = Profiler::Instance().GetAverageFrameTime();
double fps = 1000.0 / avg_frame_ms;

// Marker timing
double avg_marker = Profiler::Instance().GetAverageMarkerTime("RenderPass");
double max_marker = Profiler::Instance().GetMaxMarkerTime("RenderPass");

// GPU timing
double gpu_ms = GPUProfiler::Instance().GetAverageGPUTime();

// Combined
double combined_fps = PerformanceMonitor::Instance().GetAverageFPS();
```

---

## Common Profiling Points

### Renderer

```cpp
void Renderer::Render()
{
    SCOPED_PROFILE("Renderer::Render");
    
    {
        PROFILE_GPU("OcclusionPass");
        // occlusion rendering
    }
    
    {
        PROFILE_GPU("GeometryPass");
        // G-Buffer rendering
    }
    
    {
        PROFILE_GPU("LightingPass");
        // Lighting computation
    }
    
    {
        PROFILE_GPU("PostProcessing");
        // SSAO, SSR, TAA, bloom
    }
}
```

### Physics

```cpp
void PhysicsSystem::Update(float dt)
{
    SCOPED_PROFILE("Physics");
    
    {
        SCOPED_PROFILE("BroadPhase");
        UpdateBroadPhase();
    }
    
    {
        SCOPED_PROFILE("NarrowPhase");
        UpdateNarrowPhase();
    }
    
    {
        SCOPED_PROFILE("SolveConstraints");
        SolveConstraints();
    }
}
```

### Animation

```cpp
void Animator::Update(float dt)
{
    SCOPED_PROFILE("Animation");
    
    {
        SCOPED_PROFILE("BlendTree");
        UpdateBlendTree();
    }
    
    {
        SCOPED_PROFILE("Sampling");
        SampleAnimation();
    }
    
    {
        SCOPED_PROFILE("Skinning");
        UpdateSkinning();
    }
}
```

### Main Loop

```cpp
void Application::Update(float dt)
{
    Profiler::Instance().BeginFrame();
    
    {
        SCOPED_PROFILE("Input");
        HandleInput();
    }
    
    {
        SCOPED_PROFILE("Update");
        gameLogic_.Update(dt);
    }
    
    {
        SCOPED_PROFILE("Render");
        renderer_.Render();
    }
    
    Profiler::Instance().EndFrame();
    PerformanceMonitor::Instance().Update();
    RemoteProfiler::Instance().Update();
}
```

---

## Macros Reference

```cpp
// CPU Profiling
SCOPED_PROFILE("name")           // Automatic CPU scope
PROFILE_SCOPE("name")            // Alias for SCOPED_PROFILE

// GPU Profiling
PROFILE_GPU("name")              // Automatic GPU scope
PROFILE_GPU_COLOR("name", color) // GPU scope with color
SCOPED_GPU_PROFILE("name")       // Alias for PROFILE_GPU

// Manual
Profiler::Instance().BeginScope("name");
Profiler::Instance().EndScope();

GPUProfiler::Instance().BeginMarker("name", color);
GPUProfiler::Instance().EndMarker();
GPUProfiler::Instance().InsertMarker("name");
```

---

## Configuration

```cpp
// Enable/disable profiling
Profiler::Instance().SetEnabled(true);
RemoteProfiler::Instance().EnableProfiling(true);

// Change history size (frames to keep)
Profiler::Instance().SetMaxFrameHistory(300);

// Change server port
RemoteProfiler::Instance().Initialize(9000);

// Clear history
Profiler::Instance().Clear();
PerformanceMonitor::Instance().Clear();
```

---

## Exporting Data

```cpp
// Get as JSON
json data = RemoteProfiler::Instance().GetProfileData();

// Save to file
std::ofstream file("profile.json");
file << data.dump(2);

// Download from web UI
// Dashboard → Download Data button
```

---

## Performance Tips

✅ **DO:**
- Use SCOPED_PROFILE for automatic cleanup
- Profile major systems (not every function)
- Use GPU profiler in rendering code
- Enable telemetry server for analysis

❌ **DON'T:**
- Profile tiny functions (too much overhead)
- Nest scopes excessively (clutters data)
- Leave profiling on in shipping game (optional for telemetry)
- Profile in inner loops (cache locality issues)

---

## Debugging GPU Profiler

If GPU markers aren't showing:

1. **Check extension support:**
   ```cpp
   if (GLAD_GL_KHR_DEBUG) {
       // Debug labels supported
   }
   ```

2. **Attach debugger:**
   - RenderDoc: Capture frame
   - NSight: GPU profiling mode
   - PIX: Direct GPU capture (Windows)

3. **Use debug messages:**
   ```cpp
   GPUProfiler::Instance().InsertMarker("DebugPoint");
   ```

---

## Benchmarking

Before profiling:
```cpp
RemoteProfiler::Instance().EnableProfiling(false);
RunBenchmark();
```

With profiling:
```cpp
RemoteProfiler::Instance().EnableProfiling(true);
RemoteProfiler::Instance().Initialize(8080);
RunBenchmark();
// Compare results
```

---

## Web Dashboard Features

| Feature | Location | Use |
|---------|----------|-----|
| FPS Graph | Main dashboard | Monitor framerate |
| CPU/GPU Time | Metrics panel | Compare load |
| Per-Frame Markers | Marker table | Find bottlenecks |
| Historical Data | Charts | Trend analysis |
| Download | Control bar | External analysis |

---

## Integration Checklist

- [ ] Include headers: `#include "Profiler.h"`, `#include "TelemetryServer.h"`
- [ ] Initialize: `RemoteProfiler::Instance().Initialize(8080);`
- [ ] Frame calls: `BeginFrame()` / `EndFrame()`
- [ ] Major scopes: Add `SCOPED_PROFILE()` to systems
- [ ] GPU work: Add `PROFILE_GPU()` to rendering
- [ ] Update: Call `RemoteProfiler::Instance().Update();`
- [ ] Test: Open `http://localhost:8080` in browser
- [ ] Analysis: Download JSON and analyze

---

## Common Issues

**Q: Server won't start**
- A: Try different port: `Initialize(9000)`

**Q: GPU markers not visible**
- A: Attach RenderDoc or NSight debugger

**Q: High memory usage**
- A: Reduce frame history: `SetMaxFrameHistory(300)`

**Q: Browser shows "Cannot connect"**
- A: Check firewall, ensure server started, verify port

**Q: Profiling overhead too high**
- A: Disable in production, use conditional: `#ifdef PROFILING`

---

## File Locations

- **Headers**: `include/Profiler.h`, `include/TelemetryServer.h`
- **Implementation**: `src/Profiler.cpp`, `src/TelemetryServer.cpp`
- **Web UI**: Built-in to server (auto-generated HTML)
- **Guide**: `docs/PROFILING_TELEMETRY_GUIDE.md` (full reference)

---

**Status**: ✅ Ready to use  
**Platforms**: Windows, Linux, macOS  
**C++ Standard**: C++20  
**Dependencies**: nlohmann/json, OpenGL 3.3+  
**Max History**: 600 frames @ 60 FPS = ~10 seconds
