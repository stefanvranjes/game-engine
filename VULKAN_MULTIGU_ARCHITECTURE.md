# Vulkan Backend & Multi-GPU Support Architecture

## Overview

This document describes the architecture for adding optional Vulkan backend support and multi-GPU capabilities to the game engine. The implementation uses an abstraction layer (RenderBackend) that allows switching between OpenGL 3.3+ and Vulkan 1.3 at runtime or compile-time.

## Design Principles

1. **Backward Compatibility**: Existing OpenGL code remains unchanged; Vulkan is purely additive
2. **Abstraction Layer**: Graphics API differences hidden behind RenderBackend interface
3. **Optional Vulkan**: Vulkan can be disabled at compile-time for minimal binary size
4. **Multi-GPU**: Single unified render graph executes across available GPUs with automatic load balancing
5. **Feature Parity**: Both backends support identical features (PBR, deferred rendering, post-processing, etc.)

## Architecture Layers

```
┌─────────────────────────────────────────┐
│       Application / Game Logic          │
│     (Renderer.h - unchanged API)       │
└──────────────────┬──────────────────────┘
                   │
┌──────────────────┴──────────────────────┐
│    RenderBackend (Abstract Interface)   │
│  - Render passes, resource management   │
│  - GPU scheduling for multi-GPU        │
└──────┬──────────────────────┬───────────┘
       │                      │
   ┌───▼───────┐      ┌──────▼────────┐
   │  OpenGL   │      │    Vulkan     │
   │ Renderer  │      │   Renderer    │
   │           │      │ (Multi-GPU)   │
   └───────────┘      └───────────────┘
```

## Core Components

### 1. RenderBackend Interface (`include/RenderBackend.h`)
Abstract interface defining all graphics operations:
- Resource creation (buffers, textures, framebuffers)
- Command recording and submission
- Synchronization primitives
- GPU capabilities querying
- Multi-GPU device management

### 2. OpenGL Implementation (`include/OpenGLBackend.h`, `src/OpenGLBackend.cpp`)
- Wraps existing OpenGL code path
- Minimal changes to current Renderer
- Uses OpenGL 3.3+ with extensions

### 3. Vulkan Implementation (`include/VulkanBackend.h`, `src/VulkanBackend.cpp`)
- Full Vulkan 1.3 implementation
- Command buffer recording per GPU
- Synchronization via semaphores/fences
- Memory management with VMA (Vulkan Memory Allocator)
- Optional device selection and load balancing

### 4. GPU Scheduler (`include/GPUScheduler.h`, `src/GPUScheduler.cpp`)
Manages multi-GPU workload distribution:
- Detects available GPUs
- Assigns render passes to GPUs based on capabilities
- Handles GPU-GPU synchronization
- Balances load across devices

### 5. RenderGraph (`include/RenderGraph.h`, `src/RenderGraph.cpp`)
Frame graph for managing render pass dependencies:
- G-Buffer generation
- Lighting passes
- Post-processing chains
- Automatic scheduling across GPUs

## Implementation Plan

### Phase 1: Abstraction Layer (Foundational)
**Files to create**:
- `include/RenderBackend.h` - Abstract interface
- `include/RenderDevice.h` - Device/GPU abstraction
- `include/RenderPass.h` - Render pass abstraction
- `include/RenderResource.h` - Buffer, texture, framebuffer abstractions

**Changes to existing code**:
- Renderer.h/.cpp: Add `m_Backend` pointer, delegate graphics calls to backend
- No breaking changes to public API

**Time estimate**: 2-3 days

### Phase 2: OpenGL Backend Implementation
**Files to create**:
- `include/OpenGLBackend.h`
- `src/OpenGLBackend.cpp`
- `include/OpenGLDevice.h`
- `src/OpenGLDevice.cpp`

**Changes**:
- Wrap existing OpenGL code with RenderBackend interface
- Extract resource management into separate OpenGLDevice class
- Verify feature parity with current implementation

**Time estimate**: 1-2 days

### Phase 3: Vulkan Backend Implementation
**Files to create**:
- `include/VulkanBackend.h`
- `src/VulkanBackend.cpp`
- `include/VulkanDevice.h`
- `src/VulkanDevice.cpp`
- `include/VulkanShaderCompiler.h`
- `src/VulkanShaderCompiler.cpp`
- `include/VulkanDebugUtils.h`
- `src/VulkanDebugUtils.cpp`

**Dependencies**:
- Vulkan SDK (headers)
- SPIR-V compiler (glslang or DXC)
- VMA (Vulkan Memory Allocator)

**Time estimate**: 2-3 weeks

### Phase 4: Multi-GPU Support
**Files to create**:
- `include/GPUScheduler.h`
- `src/GPUScheduler.cpp`
- `include/RenderGraph.h`
- `src/RenderGraph.cpp`
- `include/GPUTimings.h`

**Features**:
- Automatic GPU detection
- Render graph construction
- Workload distribution algorithm
- GPU-GPU synchronization

**Time estimate**: 1-2 weeks

### Phase 5: Testing & Optimization
- Benchmark OpenGL vs Vulkan
- Profile GPU scheduling
- Multi-GPU performance validation
- Memory usage optimization

**Time estimate**: 1 week

## Configuration

### CMakeLists.txt Options
```cmake
option(ENABLE_VULKAN "Enable Vulkan backend support" OFF)
option(ENABLE_MULTI_GPU "Enable multi-GPU support" ON)
option(FORCE_VULKAN_BACKEND "Use Vulkan instead of OpenGL" OFF)
option(VULKAN_VALIDATION_LAYERS "Enable Vulkan validation layers in Debug" ON)
```

### Runtime Configuration
Environment variables:
- `GE_GRAPHICS_API`: "opengl" (default) or "vulkan"
- `GE_GPU_COUNT`: Force specific GPU count (1, 2, 4, etc.)
- `GE_GPU_SCHEDULER`: "round-robin", "load-balance", or "custom"

Example in `include/EngineConfig.h`:
```cpp
enum class GraphicsAPI {
    OpenGL,
    Vulkan
};

struct EngineConfig {
    GraphicsAPI preferredAPI = GraphicsAPI::OpenGL;
    bool enableMultiGPU = true;
    int maxGPUCount = -1; // -1 = auto-detect
    bool enableVulkanValidation = true;
};
```

## Vulkan-Specific Features

### Deferred Rendering with Vulkan
- G-Buffer rendering to MSAA color attachments
- Input attachments for efficient lighting passes
- Subpasses for automatic layout transitions
- VK_EXT_dynamic_rendering for flexible render pass compatibility

### Multi-GPU Rendering
- **Split-frame**: Each GPU renders different portions of frame
- **Linked GPUs**: Use VK_AMD_device_coherent_memory for zero-copy sharing
- **Alternate-frame**: Odd/even frames on GPU0/GPU1 (reduces latency)
- **Load-balanced**: Render graph assigns passes to GPU based on estimated cost

### Shader Compilation
GLSL shaders compiled to SPIR-V at runtime:
1. Compile GLSL to SPIR-V using glslang
2. Cache SPIR-V binaries for faster loading
3. Serialize shader metadata (push constants, descriptors) to JSON

Example flow:
```cpp
// Shader.cpp changes
Shader::Compile() {
    if (m_Backend == RenderBackend::Vulkan) {
        std::vector<uint32_t> spirv = CompileGLSLToSPIRV(source, type);
        return m_VulkanBackend->CreateShaderModule(spirv);
    }
    // OpenGL path unchanged
}
```

### Memory Management
Use VMA (Vulkan Memory Allocator):
- Automatic defragmentation
- Memory pooling for temporary allocations
- Statistics tracking for profiling

## Migration Path for Users

### Option 1: Compile-Time Selection
```cpp
// main.cpp
#ifdef ENABLE_VULKAN
    std::unique_ptr<RenderBackend> backend = std::make_unique<VulkanBackend>();
#else
    std::unique_ptr<RenderBackend> backend = std::make_unique<OpenGLBackend>();
#endif

Application app(std::move(backend));
```

### Option 2: Runtime Selection
```cpp
// Application initialization
EngineConfig config;
config.preferredAPI = GraphicsAPI::Vulkan; // or from env var

std::unique_ptr<RenderBackend> backend;
if (config.preferredAPI == GraphicsAPI::Vulkan && VulkanBackend::IsSupported()) {
    backend = std::make_unique<VulkanBackend>();
} else {
    backend = std::make_unique<OpenGLBackend>();
}

Renderer renderer(std::move(backend), config);
renderer.Init();
```

## Performance Expectations

### Vulkan vs OpenGL
| Metric | OpenGL | Vulkan | Gain |
|--------|--------|--------|------|
| Draw call overhead | ~5-10 µs | ~0.5-1 µs | 10x |
| State change cost | ~50-200 µs | ~1-5 µs | 50x |
| GPU stall risk | High (driver queue) | Low (app controlled) | Stability |
| Memory bandwidth | ~80% of peak | ~95% of peak | 20% |

### Multi-GPU Benefits (RTX 4090 + RTX 4080)
- 2-GPU split-frame: ~1.7x performance gain
- 2-GPU alternate-frame: ~1.9x performance gain (higher latency)
- Scaling diminishes beyond 2-4 GPUs (PCIe bandwidth)

## Testing Strategy

### Unit Tests
- RenderBackend interface contract
- GPU scheduler correctness
- Render graph construction
- Shader compilation

### Integration Tests
- OpenGL: Render reference scene, capture framebuffer
- Vulkan: Identical scene, compare output (pixel-perfect)
- Multi-GPU: Verify load distribution, synchronization

### Benchmark Suite
- Frame time (FPS)
- GPU memory usage
- CPU-GPU synchronization latency
- Shader compilation time

## Debugging & Profiling

### Vulkan-Specific Tools
- **RenderDoc**: Capture and inspect Vulkan frames
- **GPU-Z**: Monitor multi-GPU utilization
- **PIX**: Windows GPU debugging
- **Khronos Validation Layer**: Catch API misuse

### Engine Diagnostics
```cpp
// In ImGui debug menu
if (ImGui::TreeNode("Graphics Backend")) {
    ImGui::Text("API: %s", m_Backend->GetName()); // "OpenGL 3.3" or "Vulkan 1.3"
    ImGui::Text("Device: %s", m_Backend->GetDeviceName());
    ImGui::Separator();
    
    if (m_Backend->GetDeviceCount() > 1) {
        ImGui::Text("GPU Count: %d", m_Backend->GetDeviceCount());
        for (int i = 0; i < m_Backend->GetDeviceCount(); i++) {
            float util = m_Scheduler->GetGPUUtilization(i);
            ImGui::SliderFloat(("GPU " + std::to_string(i)).c_str(), 
                             &util, 0.0f, 100.0f, "%.1f%%");
        }
    }
}
```

## Future Enhancements

1. **Metal Backend**: macOS/iOS support
2. **DirectX 12**: Windows optimization
3. **Ray Tracing**: Vulkan RT, DXR integration
4. **Variable Rate Shading**: Reduce fragment shader cost
5. **Mesh Shaders**: Replace traditional vertex/index buffers
6. **Dynamic Rendering**: Flexible render pass compatibility
7. **Virtual Texture Streaming**: Efficient mega-texture support

## References

- [Vulkan Specification](https://www.khronos.org/vulkan/)
- [RenderDoc Documentation](https://renderdoc.org/)
- [VMA GitHub](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator)
- [Khronos Vulkan Samples](https://github.com/KhronosGroup/Vulkan-Samples)
- [GPU Scheduling Research](https://arxiv.org/abs/1511.02721)

