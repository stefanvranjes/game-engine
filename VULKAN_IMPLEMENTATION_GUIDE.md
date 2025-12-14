# Vulkan Backend & Multi-GPU Implementation Guide

## Quick Start

### Phase 1: OpenGL Backend Integration (Current Status - COMPLETE)

The abstraction layer has been created with the following files:

**Core Headers:**
- `include/RenderBackend.h` - Abstract graphics API interface (600+ lines)
- `include/OpenGLBackend.h` - OpenGL 3.3+ implementation
- `include/GPUScheduler.h` - GPU detection and load balancing
- `include/EngineConfig.h` - Configuration management
- `include/VulkanBackend.h` - Vulkan 1.3+ stub (ready for implementation)

**Core Implementation:**
- `src/RenderBackend.cpp` - Factory function
- `src/OpenGLBackend.cpp` - OpenGL wrapper (~700 lines)
- `src/GPUScheduler.cpp` - GPU scheduling logic (~400 lines)
- `src/EngineConfig.cpp` - Configuration loading

### Phase 2: Integration with Renderer (NEXT STEP)

Modify `Renderer.h` and `Renderer.cpp` to use the `RenderBackend` interface:

#### Changes to Renderer.h:

```cpp
class Renderer {
    // ... existing members ...

private:
    std::unique_ptr<RenderBackend> m_Backend;
    std::unique_ptr<GPUScheduler> m_Scheduler;
    std::unique_ptr<RenderGraph> m_RenderGraph;
    
    // Helper to select/switch backend at runtime
    void SelectRenderBackend(const EngineConfig& config);
};
```

#### Changes to Renderer.cpp initialization:

```cpp
bool Renderer::Init() {
    // Load configuration
    g_EngineConfig = EngineConfig::LoadFromEnvironment();
    
    // Select backend
    if (g_EngineConfig.preferredGraphicsAPI == GraphicsAPI::Vulkan) {
        m_Backend = std::make_unique<VulkanBackend>();
    } else {
        m_Backend = std::make_unique<OpenGLBackend>();
    }
    
    if (!m_Backend->Init(m_Width, m_Height, m_Window->GetNativeHandle())) {
        SPDLOG_ERROR("Failed to initialize render backend");
        return false;
    }
    
    // Initialize scheduler for multi-GPU support
    m_Scheduler = std::make_unique<GPUScheduler>(m_Backend.get());
    m_Scheduler->Init();
    
    // Initialize render graph
    m_RenderGraph = m_Backend->GetRenderGraph();
    
    SPDLOG_INFO("Renderer initialized with {} backend",
                m_Backend->GetAPIName());
    
    return true;
}
```

### Phase 3: CMakeLists.txt Configuration

Add to `CMakeLists.txt`:

```cmake
# Render Backend Options
option(ENABLE_VULKAN "Enable Vulkan backend support" OFF)
option(ENABLE_MULTI_GPU "Enable multi-GPU support" ON)
option(FORCE_VULKAN "Use Vulkan instead of OpenGL" OFF)

if(ENABLE_VULKAN)
    find_package(Vulkan REQUIRED)
    add_compile_definitions(ENABLE_VULKAN)
    
    # Fetch VMA (Vulkan Memory Allocator)
    FetchContent_Declare(
        vma
        GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git
        GIT_TAG v3.0.1
    )
    FetchContent_MakeAvailable(vma)
    
    # Fetch glslang (GLSL to SPIR-V compiler)
    FetchContent_Declare(
        glslang
        GIT_REPOSITORY https://github.com/KhronosGroup/glslang.git
        GIT_TAG 13.1.1
    )
    FetchContent_MakeAvailable(glslang)
endif()

if(FORCE_VULKAN)
    add_compile_definitions(FORCE_VULKAN_BACKEND)
endif()

# Add new source files to GameEngine target
target_sources(GameEngine PRIVATE
    src/RenderBackend.cpp
    src/OpenGLBackend.cpp
    src/GPUScheduler.cpp
    src/EngineConfig.cpp
    ${VULKAN_SOURCES} # Only compiled if ENABLE_VULKAN
)

# Link Vulkan if enabled
if(ENABLE_VULKAN)
    target_link_libraries(GameEngine PRIVATE
        Vulkan::Vulkan
        VulkanMemoryAllocator
        glslang::glslang
        glslang::SPIRV
    )
    target_sources(GameEngine PRIVATE
        src/VulkanBackend.cpp
        src/VulkanShaderCompiler.cpp
        src/VulkanDebugUtils.cpp
    )
endif()
```

## Implementation Roadmap

### Week 1-2: OpenGL Backend Integration
- [ ] Modify Renderer to use RenderBackend interface
- [ ] Integrate EngineConfig loading
- [ ] Test OpenGL path with existing shaders
- [ ] Implement GPU scheduling for single GPU
- [ ] Benchmark: Compare legacy OpenGL vs abstracted OpenGL

**Success Criteria:**
- Game renders identically with legacy OpenGL and new abstracted backend
- No performance regression
- All existing features work (deferred rendering, post-processing, etc.)

### Week 3: Vulkan Backend Foundation
- [ ] Implement VulkanBackend initialization
  - [ ] Instance creation with validation layers
  - [ ] Physical device enumeration
  - [ ] Logical device creation with queues
  - [ ] Surface and swapchain setup
  - [ ] Command pool/buffer management
  
- [ ] Implement basic resource creation
  - [ ] Buffer creation (with VMA)
  - [ ] Texture creation and loading
  - [ ] Framebuffer/RenderPass creation
  
- [ ] Implement basic rendering
  - [ ] BeginRenderPass/EndRenderPass
  - [ ] Draw/DrawIndexed commands
  - [ ] Pipeline binding

**Success Criteria:**
- Render simple triangle with Vulkan
- No validation errors
- Frame times comparable to OpenGL

### Week 4: Vulkan Shader Compilation
- [ ] Integrate glslang for GLSL→SPIR-V compilation
- [ ] Create shader caching system
- [ ] Implement pipeline creation from shaders
- [ ] Handle push constants and descriptors

**Success Criteria:**
- All existing GLSL shaders compile to SPIR-V
- Shader hot-reload works
- PBR shaders render correctly

### Week 5: Vulkan Deferred Rendering
- [ ] Implement G-Buffer passes
- [ ] Implement lighting pass with input attachments
- [ ] Port post-processing effects
- [ ] Test feature parity with OpenGL

**Success Criteria:**
- G-Buffer output matches OpenGL
- Lighting calculations identical
- Post-processing effects work

### Week 6: Vulkan Multi-GPU Support
- [ ] Implement device group queries
- [ ] Add split-frame rendering support
- [ ] Implement alternate-frame rendering
- [ ] Handle GPU synchronization

**Success Criteria:**
- 2 GPU scaling: >1.7x performance
- Zero synchronization errors
- Render output identical to single GPU

### Week 7: Testing & Optimization
- [ ] Comprehensive benchmarking
- [ ] Memory usage profiling
- [ ] Validation layer cleanup
- [ ] Performance tuning
- [ ] Documentation

## Key Files Needing Implementation

### Vulkan Backend Core (Priority: HIGH)
```
src/VulkanBackend.cpp          ~1500 lines
src/VulkanDevice.cpp           ~1000 lines
src/VulkanShaderCompiler.cpp   ~500 lines
src/VulkanDebugUtils.cpp       ~300 lines
```

### Vulkan Device Abstraction
```cpp
// include/VulkanDevice.h
class VulkanDevice {
public:
    VulkanInstance* GetInstance() const;
    VkPhysicalDevice GetPhysicalDevice() const;
    VkDevice GetLogicalDevice() const;
    VkQueue GetGraphicsQueue() const;
    VkQueue GetComputeQueue() const;
    VkCommandPool GetCommandPool() const;
    
    // Memory management
    std::shared_ptr<RenderResource> AllocateBuffer(
        size_t size,
        VkBufferUsageFlags usage,
        VkMemoryPropertyFlags properties);
    
    // Synchronization
    void WaitForIdle();
    VkFence CreateFence();
    VkSemaphore CreateSemaphore();
    
private:
    VkPhysicalDevice m_PhysicalDevice;
    VkDevice m_LogicalDevice;
    VkQueue m_GraphicsQueue;
    VkQueue m_ComputeQueue;
    std::unique_ptr<VulkanMemoryAllocator> m_Allocator;
};
```

## Environment Variables for Testing

```bash
# Force Vulkan backend
set GE_GRAPHICS_API=vulkan

# Enable multi-GPU with 2 GPUs
set GE_MULTI_GPU=true
set GE_GPU_COUNT=2

# Enable Vulkan validation
set GE_VULKAN_VALIDATION=true

# Run with split-frame rendering
# (Requires Vulkan backend)
set GE_MULTI_GPU_STRATEGY=split-frame
```

## Build Instructions

### With OpenGL Only (Default)
```bash
cmake -B build
cmake --build build --config Release
```

### With Vulkan Support
```bash
cmake -B build -DENABLE_VULKAN=ON
cmake --build build --config Release
```

### Force Vulkan (Requires Vulkan SDK)
```bash
cmake -B build -DENABLE_VULKAN=ON -DFORCE_VULKAN=ON
cmake --build build --config Release
```

## Testing Strategy

### Unit Tests
```cpp
// tests/test_render_backend.cpp
TEST(RenderBackendTest, OpenGLCreation) {
    auto backend = std::make_unique<OpenGLBackend>();
    ASSERT_TRUE(backend->Init(800, 600, nullptr));
    EXPECT_EQ(backend->GetAPI(), RenderBackend::API::OpenGL);
}

TEST(RenderBackendTest, VulkanCreation) {
    if (!IsVulkanAvailable()) SKIP("Vulkan not available");
    auto backend = std::make_unique<VulkanBackend>();
    ASSERT_TRUE(backend->Init(800, 600, nullptr));
    EXPECT_EQ(backend->GetAPI(), RenderBackend::API::Vulkan);
}

TEST(GPUSchedulerTest, MultiGPUDetection) {
    auto backend = std::make_unique<OpenGLBackend>();
    backend->Init(800, 600, nullptr);
    
    GPUScheduler scheduler(backend.get());
    scheduler.Init();
    
    EXPECT_GE(scheduler.GetGPUCount(), 1);
}
```

### Integration Tests
```bash
# Compare OpenGL vs Vulkan output
./test_rendering --compare-backends --output-path results/

# Benchmark multi-GPU scaling
./benchmark_multi_gpu --gpu-count 2 --duration 60
```

### Visual Regression Tests
```bash
# Capture reference frame from OpenGL
./game_engine --backend opengl --screenshot opengl_ref.ppm

# Capture test frame from Vulkan
./game_engine --backend vulkan --screenshot vulkan_test.ppm

# Compare (pixel difference < 1%)
./image_compare opengl_ref.ppm vulkan_test.ppm
```

## Performance Profiling

### With Vulkan Validation
```cpp
// In VulkanBackend::Init()
VkValidationFeatureEnableEXT enables[] = {
    VK_VALIDATION_FEATURE_ENABLE_GPU_ASSISTED_EXT
};
VkValidationFeaturesEXT features{};
features.enabledValidationFeatureCount = 1;
features.pEnabledValidationFeatures = enables;
```

### GPU Timeline Analysis
```cpp
// Timestamped commands
backend->BeginGPUQuery("geometry_pass");
// ... render geometry ...
double elapsed = backend->EndGPUQuery("geometry_pass");
SPDLOG_INFO("Geometry pass: {:.2f}ms", elapsed);
```

### Multi-GPU Load Monitoring
```cpp
auto loads = scheduler->GetGPUUtilizations();
for (size_t i = 0; i < loads.size(); i++) {
    SPDLOG_INFO("GPU {}: {:.1f}%", i, loads[i]);
}
```

## Troubleshooting

### "Vulkan not found"
- Install Vulkan SDK from https://www.lunarg.com/vulkan-sdk/
- Set `VULKAN_SDK` environment variable
- Restart Visual Studio

### "SPIR-V compilation failed"
- Check glslang version compatibility
- Verify shader syntax with `glslangValidator.exe shader.vert`

### Multi-GPU not scaling
- Check `scheduler->GetGPUUtilizations()` to see load distribution
- Verify linked GPU support with `backend->SupportsLinkedGPUs()`
- Profile with RenderDoc to see GPU utilization

### Performance regression
- Compare frame times: `backend->EndGPUQuery("frame_total")`
- Profile with PIX or RenderDoc
- Check CPU-GPU stall points

## References

- [Vulkan Specification](https://www.khronos.org/vulkan/)
- [VMA Documentation](https://gpuopen.com/guidelines/naming-vulkan-memory-allocator/)
- [glslang GitHub](https://github.com/KhronosGroup/glslang)
- [RenderDoc Tutorial](https://renderdoc.org/docs/quick_start/index.html)

## Next Steps

1. **Review** this architecture with the team
2. **Decide** on Vulkan priority (optional vs core)
3. **Assign** Week 1-2 work for OpenGL integration
4. **Setup** CI/CD for both backends
5. **Schedule** Vulkan implementation phases

