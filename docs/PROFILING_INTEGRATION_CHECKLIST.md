# Profiler Integration Checklist

Use this checklist when integrating profiling into your code.

## Setup Phase

- [ ] Include headers in Application.h
  ```cpp
  #include "Profiler.h"
  #include "TelemetryServer.h"
  ```

- [ ] Initialize remote profiler in `Application::Initialize()`
  ```cpp
  RemoteProfiler::Instance().Initialize(8080);
  ```

- [ ] Add frame management in game loop `Application::Update()`
  ```cpp
  Profiler::Instance().BeginFrame();
  // ... frame code ...
  Profiler::Instance().EndFrame();
  RemoteProfiler::Instance().Update();
  ```

- [ ] Add shutdown in `Application::Shutdown()`
  ```cpp
  RemoteProfiler::Instance().Shutdown();
  ```

- [ ] Test access to dashboard
  ```
  http://localhost:8080
  ```

## Integration Phase

### Major Systems

- [ ] **Renderer**
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

- [ ] **Physics System**
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

- [ ] **Animation System**
  ```cpp
  void AnimationSystem::Update(float dt)
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

- [ ] **Audio System**
  ```cpp
  void AudioSystem::Update()
  {
      SCOPED_PROFILE("Audio");
      // Update listeners, sources
  }
  ```

- [ ] **Particle System**
  ```cpp
  void ParticleSystem::Update(float dt)
  {
      SCOPED_PROFILE("Particles");
      
      {
          SCOPED_PROFILE("Emission");
          UpdateEmitters();
      }
      
      {
          SCOPED_PROFILE("Physics");
          UpdateParticlePhysics();
      }
  }
  ```

- [ ] **Game Logic**
  ```cpp
  void GameLogic::Update(float dt)
  {
      SCOPED_PROFILE("GameLogic");
      // Update game systems
  }
  ```

### Rendering Pipeline Details

- [ ] Occlusion culling
  ```cpp
  {
      PROFILE_GPU("OcclusionPass");
      // occlusion rendering
  }
  ```

- [ ] Geometry pass (G-Buffer)
  ```cpp
  {
      PROFILE_GPU_COLOR("GeometryPass", glm::vec4(0, 1, 0, 1));
      // G-buffer rendering
  }
  ```

- [ ] Lighting pass
  ```cpp
  {
      PROFILE_GPU_COLOR("LightingPass", glm::vec4(1, 1, 0, 1));
      // Lighting computation
  }
  ```

- [ ] Shadow map rendering
  ```cpp
  {
      PROFILE_GPU_COLOR("ShadowMaps", glm::vec4(1, 0, 1, 1));
      // Directional + point light shadows
  }
  ```

- [ ] SSAO effect
  ```cpp
  {
      PROFILE_GPU("SSAO");
      // Screen-space ambient occlusion
  }
  ```

- [ ] SSR effect
  ```cpp
  {
      PROFILE_GPU("SSR");
      // Screen-space reflections
  }
  ```

- [ ] TAA effect
  ```cpp
  {
      PROFILE_GPU("TAA");
      // Temporal anti-aliasing
  }
  ```

- [ ] Bloom effect
  ```cpp
  {
      PROFILE_GPU("Bloom");
      // Bloom extraction and blending
  }
  ```

- [ ] Volumetric fog
  ```cpp
  {
      PROFILE_GPU("VolumetricFog");
      // Volumetric fog rendering
  }
  ```

- [ ] Forward rendering (particles, transparent)
  ```cpp
  {
      PROFILE_GPU("ForwardPass");
      // Particle and transparent rendering
  }
  ```

## Verification Phase

- [ ] Build succeeds
  ```bash
  cmake --preset windows-msvc-release
  cmake --build --preset windows-msvc-release
  ```

- [ ] No compilation warnings
  ```bash
  # Check build output
  ```

- [ ] Application starts
  ```bash
  ./build/Debug/GameEngine.exe
  ```

- [ ] Dashboard accessible
  ```
  http://localhost:8080
  # Should show the profiler dashboard
  ```

- [ ] Metrics displayed
  - [ ] FPS gauge shows
  - [ ] CPU/GPU times shown
  - [ ] Frame time chart updates
  - [ ] Marker list populated

- [ ] Performance reasonable
  - [ ] FPS within expected range
  - [ ] CPU time < 16.67ms (60 FPS target)
  - [ ] GPU time < 16.67ms
  - [ ] Profiling overhead < 5%

- [ ] Data export works
  - [ ] Download button functional
  - [ ] JSON file valid format
  - [ ] Contains all expected fields

## Testing Phase

- [ ] CPU measurements accurate
  ```cpp
  auto start = chrono::high_resolution_clock::now();
  {
      SCOPED_PROFILE("Test");
      std::this_thread::sleep_for(chrono::milliseconds(100));
  }
  auto elapsed = chrono::high_resolution_clock::now() - start;
  // Verify SCOPED_PROFILE shows ~100ms
  ```

- [ ] GPU markers appear in debugger
  - [ ] Attach RenderDoc
  - [ ] Capture frame
  - [ ] See marker labels in timeline

- [ ] Thread safety verified
  - [ ] Multi-threaded update (async physics, animation)
  - [ ] No crashes or data corruption
  - [ ] Profiler output consistent

- [ ] History buffer working
  - [ ] Dashboard shows 100+ frames
  - [ ] Chart smooth and animated
  - [ ] Data persists after new frames

- [ ] Remote access works
  - [ ] Access from another machine on network
  - [ ] Metrics update in real-time
  - [ ] Download works from remote

## Performance Baseline

Document baseline metrics before optimization:

- [ ] Measure baseline FPS
  ```
  Average FPS: _____
  Min FPS: _____
  Max FPS: _____
  ```

- [ ] Measure baseline CPU time
  ```
  Render time: _____ ms
  Physics time: _____ ms
  Animation time: _____ ms
  Total CPU: _____ ms
  ```

- [ ] Measure baseline GPU time
  ```
  GPU total: _____ ms
  Geometry Pass: _____ ms
  Lighting Pass: _____ ms
  Post-Processing: _____ ms
  ```

- [ ] Document profiling overhead
  ```
  Without profiling: _____ FPS
  With profiling: _____ FPS
  Overhead: _____% 
  ```

## Production Checklist

- [ ] Profiling disabled by default in shipped builds
  ```cpp
  #ifdef DEVELOPMENT_BUILD
      RemoteProfiler::Instance().Initialize(8080);
  #endif
  ```

- [ ] Console commands for profiling control
  ```cpp
  if (command == "profile enable")
      RemoteProfiler::Instance().EnableProfiling(true);
  if (command == "profile disable")
      RemoteProfiler::Instance().EnableProfiling(false);
  ```

- [ ] Configuration file for telemetry
  ```json
  {
      "telemetry": {
          "enabled": false,
          "port": 8080,
          "max_frame_history": 600
      }
  }
  ```

- [ ] Telemetry security (if exposed)
  - [ ] Authentication mechanism
  - [ ] Rate limiting
  - [ ] Data encryption (HTTPS)
  - [ ] Access logging

- [ ] Documentation updated
  - [ ] README mentions profiling features
  - [ ] Wiki/docs linked to PROFILING_QUICK_REFERENCE.md
  - [ ] Performance tuning guide updated
  - [ ] FAQ includes profiling Q&A

## Custom Metrics (Optional)

- [ ] Add custom metric collection
  ```cpp
  class CustomMetrics {
  public:
      static void Record(const string& name, double value) {
          auto data = json::object({
              {"metric", name},
              {"value", value},
              {"timestamp", time(nullptr)}
          });
          RemoteProfiler::Instance().GetServer()
              ->PublishMessage("custom", data);
      }
  };
  ```

- [ ] Track game-specific data
  ```cpp
  CustomMetrics::Record("GameObjectCount", game_objects.size());
  CustomMetrics::Record("TriangleCount", renderer.GetTriangleCount());
  CustomMetrics::Record("ShadowMapSize", shadow_map_resolution);
  ```

- [ ] Display on custom dashboard
  - [ ] Extend HTML dashboard
  - [ ] Add custom chart type
  - [ ] Real-time updates

## Documentation

- [ ] Add profiling section to README
- [ ] Link to PROFILING_QUICK_REFERENCE.md
- [ ] Include example dashboard screenshots
- [ ] Document custom metrics if used
- [ ] Add troubleshooting section

## Sign-Off

- [ ] All checks passed
- [ ] Baseline metrics documented
- [ ] Code review completed
- [ ] Performance acceptable
- [ ] Documentation complete
- [ ] Ready for testing phase

---

**Completion Date**: _________  
**Developer**: _________  
**Notes**: 

```


```

---

See [PROFILING_QUICK_REFERENCE.md](PROFILING_QUICK_REFERENCE.md) for quick lookup.  
See [PROFILING_TELEMETRY_GUIDE.md](PROFILING_TELEMETRY_GUIDE.md) for complete documentation.
