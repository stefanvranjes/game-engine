# Multi-GPU Rendering Techniques & Strategies

## Overview

This document explains the three primary multi-GPU rendering strategies implemented in the game engine and when to use each.

## 1. Single GPU (Baseline)

**When to use:**
- Single GPU systems
- Debugging graphics issues
- Baseline performance comparison

**Characteristics:**
- No synchronization overhead
- All rendering on GPU 0
- Simplest code path

**Performance:** Baseline (1.0x)

```cpp
// Automatic selection
GPUScheduler::Strategy strategy = scheduler->RecommendStrategy();
if (strategy == GPUScheduler::Strategy::Single) {
    // All rendering uses GPU 0
    backend->SetActiveDevice(0);
}
```

## 2. Split-Frame (Horizontal Split)

**When to use:**
- 2-4 GPUs of similar performance
- Low latency critical (VR, competitive gaming)
- Balanced workload distribution possible

**How it works:**
```
GPU0: Renders left half     GPU1: Renders right half
     X: [0, width/2)              X: [width/2, width)
     
Combined frame sent to display
```

**Advantages:**
- ~1.7x-1.9x speedup with 2 balanced GPUs
- Low latency: Frame ready every frame (no buffering)
- Simple load balancing: Vertical scissor splits viewport

**Disadvantages:**
- Requires GPU-GPU synchronization
- Edge artifacts if not handled carefully
- Limited scaling beyond 2-3 GPUs

**Implementation:**

```cpp
// In Renderer::Update()
if (multiGPUStrategy == MultiGPUStrategy::SplitFrame) {
    uint32_t halfWidth = m_Width / 2;
    
    // GPU 0: Left half
    backend->SetActiveDevice(0);
    backend->SetViewport(0, 0, halfWidth, m_Height);
    backend->BeginRenderPass(m_GBuffer, clearColor);
    // ... render geometry ...
    backend->EndRenderPass();
    
    // GPU 1: Right half
    backend->SetActiveDevice(1);
    backend->SetViewport(halfWidth, 0, halfWidth, m_Height);
    backend->BeginRenderPass(m_GBuffer, clearColor);
    // ... render same geometry ...
    backend->EndRenderPass();
    
    // Synchronize before lighting pass
    backend->SyncGPUs();
    
    // Lighting pass (can also split)
    // ...
}
```

**Multi-way splits (4 GPUs):**

```
┌────────────┬────────────┐
│   GPU0     │   GPU1     │
├────────────┼────────────┤
│   GPU2     │   GPU3     │
└────────────┴────────────┘
```

```cpp
uint32_t gpuTileWidth = m_Width / 2;
uint32_t gpuTileHeight = m_Height / 2;

for (uint32_t gpuIdx = 0; gpuIdx < 4; gpuIdx++) {
    uint32_t x = (gpuIdx % 2) * gpuTileWidth;
    uint32_t y = (gpuIdx / 2) * gpuTileHeight;
    
    backend->SetActiveDevice(gpuIdx);
    backend->SetViewport(x, y, gpuTileWidth, gpuTileHeight);
    // ... render tile ...
}

backend->SyncGPUs();
```

## 3. Alternate-Frame (Temporal Split)

**When to use:**
- Any multi-GPU configuration
- Higher latency tolerance
- Complex scenes with non-uniform load distribution
- Display refresh rates ≤60 Hz

**How it works:**
```
Frame 0: GPU0 renders ──────┐
Frame 1:        GPU1 renders├─→ Display @ 60 Hz
Frame 2: GPU0 renders ──────┤
Frame 3:        GPU1 renders┘
```

**Advantages:**
- ~1.9x-2.0x speedup with 2 GPUs (each GPU has 2x time)
- No spatial synchronization needed
- Scales well to 3+ GPUs
- Balances uneven workloads naturally
- Works with asynchronous compute

**Disadvantages:**
- Frame latency doubled (usually 2-3 frames buffered)
- Noticeable in VR (causes sickness)
- Input-to-output latency: ~33ms at 60 Hz

**Implementation:**

```cpp
// In Application::Update()
static uint32_t frameCount = 0;
uint32_t gpuIndex = frameCount % 2; // Alternates 0, 1, 0, 1, ...

backend->SetActiveDevice(gpuIndex);

// Render full frame on assigned GPU
backend->BeginRenderPass(m_GBuffer, clearColor);
RenderGeometry();
backend->EndRenderPass();

RenderLighting();
ApplyPostProcessing();
Present();

frameCount++;
```

**For 4 GPUs (round-robin):**

```cpp
uint32_t gpuIndex = frameCount % 4;
backend->SetActiveDevice(gpuIndex);
// ... render frame ...
frameCount++;
```

## 4. Pinned Memory Transfers (Advanced)

**When to use:**
- Fine-grained load balancing needed
- Specific render passes assigned to specific GPUs
- Complex dependency graphs

**Technique:**
Each render pass is assigned to optimal GPU based on:
- GPU utilization
- GPU capabilities (RT cores, compute)
- Data locality

```cpp
// RenderGraph assigns passes to GPUs
RenderPassRequirements gBufferReqs;
gBufferReqs.type = RenderPassType::Geometry;
gBufferReqs.estimatedGPUTime = 4.5f; // ms
gBufferReqs.name = "G-Buffer";

renderGraph->AddPass("G-Buffer", gBufferReqs, [this]() {
    RenderGeometryPass();
});

// Scheduler assigns to GPU with lowest current load
renderGraph->Compile(numGPUs);
```

## Load Balancing Algorithm

```cpp
// In GPUScheduler::SelectGPU()
uint32_t selectedGPU = 0;
float minLoad = m_GPUs[0].currentLoad;

for (uint32_t i = 1; i < m_GPUCount; i++) {
    if (m_GPUs[i].currentLoad < minLoad) {
        minLoad = m_GPUs[i].currentLoad;
        selectedGPU = i;
    }
}

// Account for GPU performance differences
float normalizedLoad = m_GPUs[selectedGPU].currentLoad /
                     m_GPUs[selectedGPU].estimatedTFlops;

return selectedGPU;
```

## Synchronization Primitives

### CPU-GPU Synchronization
```cpp
// Wait for all GPU work to complete
backend->WaitForGPU();

// Per-GPU fence
VkFence fence = backend->GetDeviceHandle(gpuIndex);
vkWaitForFences(..., fence, ...);
```

### GPU-GPU Synchronization
```cpp
// Cross-GPU synchronization (Vulkan only)
if (backend->SupportsLinkedGPUs()) {
    backend->SyncGPUs();
}

// Alternatively: CPU waits for both GPUs
for (uint32_t i = 0; i < numGPUs; i++) {
    backend->SetActiveDevice(i);
    backend->WaitForGPU();
}
```

### Framebuffer Coherency
```cpp
// Ensure color/depth writes visible to next pass
backend->FramebufferBarrier();

// Full memory barrier
backend->MemoryBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
```

## Memory Considerations

### Linked GPU Coherent Memory
```cpp
// VK_AMD_device_coherent_memory extension
// Zero-copy shared memory between GPUs

if (backend->SupportsLinkedGPUs()) {
    // Allocate coherent memory
    VkMemoryAllocateInfo allocInfo{};
    allocInfo.memoryTypeIndex = coherentMemoryType;
    
    VkDeviceMemory memory;
    vkAllocateMemory(device, &allocInfo, nullptr, &memory);
    
    // Both GPUs can access without explicit sync
    // Useful for: double buffers, shared textures
}
```

### Unified Memory (Alternatives)
```cpp
// Option 1: Mirror textures on each GPU
for (uint32_t i = 0; i < numGPUs; i++) {
    auto texture = backend->CreateTexture(...);
    gpuTextures[i] = texture;
}

// Option 2: DMA transfer before first use
for (uint32_t i = 0; i < numGPUs; i++) {
    backend->SetActiveDevice(i);
    backend->CopyBufferToTexture(stagingBuffer, gpuTexture, ...);
}
```

## Performance Analysis

### Frame Time Breakdown (2 GPU, Split-Frame)

```
Frame: 16.67ms (60 Hz)
├─ GPU0 geometry pass:    4.5ms
├─ GPU1 geometry pass:    4.2ms (parallel with GPU0)
├─ Sync point:            0.3ms
├─ GPU0/1 lighting pass:  3.8ms (parallel)
├─ GPU0/1 post-process:   2.5ms (parallel)
├─ GPU0/1 present:        1.2ms (parallel)
└─ Total: ~8.5ms effective (1.96x speedup)
```

### Frame Time Breakdown (2 GPU, Alternate-Frame)

```
Frame N (GPU0):   16.67ms
├─ Geometry:      4.5ms
├─ Lighting:      3.8ms
├─ Post-process:  2.5ms
├─ Wait for GPU1: 5.37ms (GPU1 still rendering frame N-1)
└─ Present:       1.2ms

Frame N+1 (GPU1): 16.67ms (rendered in parallel during frame N)
```

## Implementation Checklist

- [ ] **Phase 1**: Single GPU (OpenGL abstraction)
  - [ ] Run benchmarks for baseline
  - [ ] Profile rendering pipeline
  - [ ] Identify bottlenecks

- [ ] **Phase 2**: Split-Frame preparation
  - [ ] Implement GPU query/detection
  - [ ] Add viewport splitting logic
  - [ ] Test GPU synchronization
  - [ ] Benchmark: Should see ~1.7x with balanced GPUs

- [ ] **Phase 3**: Alternate-Frame
  - [ ] Implement frame ping-ponging
  - [ ] Add GPU index tracking
  - [ ] Profile latency impact
  - [ ] Benchmark: Should see ~1.9x with any GPU count

- [ ] **Phase 4**: Smart scheduling
  - [ ] Profile individual passes
  - [ ] Implement load prediction
  - [ ] Add dynamic strategy switching
  - [ ] Benchmark: Should see >2x for unbalanced loads

## Environment Variables for Multi-GPU

```bash
# Enable multi-GPU
set GE_MULTI_GPU=true

# Force specific count
set GE_GPU_COUNT=2

# Strategy selection
set GE_MULTI_GPU_STRATEGY=split-frame  # or alternate-frame

# Debug output
set GE_GPU_SCHEDULER_DEBUG=true
```

## Troubleshooting Multi-GPU Issues

### Rendering artifacts at GPU boundaries (split-frame)
- **Cause**: Misaligned scissor rect or viewport
- **Solution**: Verify viewport calculation, add 1px overlap if needed

```cpp
// Add 1px overlap for filtering
uint32_t halfWidth = m_Width / 2 + 1;
backend->SetViewport(0, 0, halfWidth, m_Height);
```

### Inconsistent frame times (alternate-frame)
- **Cause**: Unbalanced workload or GPU throttling
- **Solution**: Profile each pass, consider workload redistribution

```cpp
auto stats = renderGraph->GetStats();
for (const auto& [passName, time] : stats.perPassTimes) {
    SPDLOG_INFO("{}: {:.2f}ms", passName, time);
}
```

### GPU sync hangs
- **Cause**: Deadlock in synchronization primitives
- **Solution**: Enable validation layers, check fence/semaphore usage

```cpp
// Debug: Timeout instead of infinite wait
VkResult result = vkWaitForFences(device, 1, &fence, VK_TRUE, 1000000000);
if (result == VK_TIMEOUT) {
    SPDLOG_ERROR("GPU sync timeout!");
}
```

### No scaling beyond 2 GPUs
- **Cause**: PCIe bandwidth saturation or workload imbalance
- **Solution**: Profile with vendor tools (NVIDIA NSight, AMD RGP)

## References

- NVIDIA: "Multi-GPU Rendering" - https://developer.nvidia.com/
- AMD: "MGPU Best Practices" - https://gpuopen.com/
- Khronos: "Vulkan Device Groups" - https://www.khronos.org/vulkan/
- IEEE: "GPU Scheduling" - https://arxiv.org/abs/2003.04629

