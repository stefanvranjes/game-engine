# Deferred + Forward Hybrid Rendering with GPU-Driven Culling

## Overview

This document describes the new **Deferred + Forward Hybrid Renderer** with **GPU-driven culling** using compute shaders and an **SRP-like (Scriptable Render Pipeline) architecture**.

## Architecture Components

### 1. GPU-Driven Culling System (`GPUCullingSystem`)

**Purpose**: Replace CPU frustum culling with GPU compute shaders to offload visibility determination to the GPU.

**Key Features**:
- **Frustum Culling**: Tests bounding boxes/spheres against camera frustum using compute shaders
- **Occlusion Culling**: Tests visibility against depth hierarchy (mipmap pyramid) from previous frame
- **LOD Selection**: Computes appropriate LOD level based on distance and screen coverage
- **Indirect Draw Buffers**: Results stored in GPU buffers for direct GPU-driven rendering

**Usage Flow**:
```cpp
GPUCullingSystem cullingSystem;
cullingSystem.Initialize();

// Per-frame:
cullingSystem.SetupCulling(viewMat, projMat, cameraPos, instanceCount);
cullingSystem.SetCullData(instanceData);
cullingSystem.ExecuteAll(depthTextureFromPreviousFrame);

auto results = cullingSystem.GetResults();  // {visibleIndices, lodLevels, counts}
```

**GPU Shaders**:
- `gpu_cull_frustum.comp`: Tests instances against frustum planes (6 planes extracted from view-projection matrix)
- `gpu_cull_occlusion.comp`: Tests visibility using depth pyramid sampling with automatic mip selection
- Both use shared data structure `CullData` containing model matrix, bounding volume, and LOD info

**Compute Configuration**:
- **Workgroup Size**: 32x1x1 (one instance per thread for coalesced memory access)
- **Output**: Visibility flags and LOD levels stored in SSBOs for downstream rendering

### 2. SRP-Like Render Pipeline (`RenderPass` + `RenderPipeline`)

**Purpose**: Provide modular, extensible render pipeline following Unity's Scriptable Render Pipeline design.

**Class Hierarchy**:
```
RenderPass (abstract)
├── ShadowPass
├── GeometryPass (G-Buffer fill)
├── LightingPass (deferred lighting)
├── TransparentPass (forward)
├── PostProcessingPass
└── CompositePass

RenderPipeline
├── Contains ordered list of RenderPass instances
├── Executes passes in sequence
└── Allows dynamic enabling/disabling of passes
```

**Benefits**:
- **Modularity**: Each pass is self-contained and reusable
- **Extensibility**: Easy to add custom passes or reorder existing ones
- **Flexibility**: Individual passes can be swapped, disabled, or replaced
- **Debugging**: Each pass can have independent debug modes

**Example Usage**:
```cpp
auto pipeline = std::make_unique<RenderPipeline>("MainPipeline");

auto geometryPass = std::make_unique<GeometryPass>("GeometryPass");
geometryPass->Initialize();
pipeline->AddPass(std::move(geometryPass));

RenderPass::PassContext ctx = {camera, width, height, deltaTime, ...};
pipeline->Execute(ctx);
```

### 3. Hybrid Renderer (`HybridRenderer`)

**Purpose**: Orchestrate the complete rendering pipeline combining deferred and forward rendering.

**Rendering Modes**:
1. **DeferredOnly**: Pure deferred rendering (all lights in screen-space)
2. **ForwardOnly**: Pure forward rendering (traditional approach)
3. **HybridOptimized** (default):
   - Opaque objects: Deferred (write G-Buffer)
   - Transparent objects: Forward (blend on top)
   - Provides best performance for mixed scenes
4. **HybridDebug**: Hybrid with culling visualization overlays

**Lighting Modes**:
- **Deferred**: All lights computed in screen-space (limited by max simultaneous lights, typically 32)
- **ForwardPlus**: Clustered forward+ (tiles compute light lists for each tile)
- **TiledDeferred**: Tiled deferred (tiles compute light lists for screen-space lighting)

**Pipeline Stages**:
```
1. GPU Culling
   ├─ Frustum culling (compute)
   ├─ Occlusion culling (compute)
   └─ LOD selection (compute)
   
2. Geometry Pass (Deferred)
   ├─ G-Buffer output: Position, Normal, Albedo+Spec, Emissive
   └─ Depth buffer filled for occlusion queries
   
3. Lighting Pass
   ├─ Screen-space deferred lighting (compute shader)
   ├─ PBR evaluation per pixel
   └─ Shadow map sampling
   
4. Transparent Pass (Forward)
   ├─ Blend transparent objects on top
   ├─ Particles
   └─ Billboards
   
5. Post-Processing
   ├─ SSAO
   ├─ SSR
   ├─ TAA
   ├─ Volumetric fog
   └─ Bloom
   
6. Composite
   └─ Final output to screen
```

## GPU Memory Layout

### G-Buffer Format (4 targets, 32 bits each)
| Target | R | G | B | A |
|--------|---|---|---|---|
| 0 (Position) | X | Y | Z | Depth |
| 1 (Normal) | X | Y | Z | Roughness |
| 2 (Albedo) | R | G | B | Metallic |
| 3 (Emissive) | R | G | B | AO |

### Culling Data Structure
```glsl
struct CullData {
    mat4 modelMatrix;        // Model-to-world transform
    vec4 boundingSphere;     // xyz=center, w=radius
    vec4 aabbMin;           // Bounding box minimum
    vec4 aabbMax;           // Bounding box maximum
    uint meshletCount;      // For potential meshlet culling
    uint lodLevel;          // Current LOD (populated by compute)
    uint isVisible;         // Visibility flag (populated by compute)
    uint screenCoverage;    // Approx % of screen occupied
};
```

### Light Data Buffer
```glsl
struct Light {
    vec4 position;          // xyz=pos, w=type (0=dir, 1=point, 2=spot)
    vec4 direction;         // xyz=direction, w=unused
    vec4 colorIntensity;    // rgb=color, a=intensity
    vec4 params;            // x=range, y=spotAngle, z=attenuation, w=bias
};
```

## Frustum Culling Algorithm

**GPU Implementation** (`gpu_cull_frustum.comp`):

1. **Extract Frustum Planes**: 6 planes from view-projection matrix
   ```glsl
   mat4 viewProj = projection * view;
   // Right plane: (viewProj[0][3] - viewProj[0][0], ...)
   // Left plane: (viewProj[0][3] + viewProj[0][0], ...)
   // ... (similar for top, bottom, near, far)
   ```

2. **Test Sphere Against Planes** (per instance):
   ```glsl
   bool testSphereFrustum(vec3 center, float radius) {
       for(int i = 0; i < 6; i++) {
           float dist = dot(center, plane[i].xyz) - plane[i].w;
           if(dist < -radius) return false;  // Outside
       }
       return true;
   }
   ```

3. **Conservative AABB Test**: Test bounding box corners against each plane

4. **LOD Selection**: Based on distance to camera
   ```glsl
   float distance = length(sphereCenter - cameraPosition);
   uint lod = (distance > 50.0) ? 3 : 
             (distance > 25.0) ? 2 : 
             (distance > 10.0) ? 1 : 0;
   ```

## Occlusion Culling Algorithm

**GPU Implementation** (`gpu_cull_occlusion.comp`):

1. **Project Sphere to Screen Space**:
   ```glsl
   vec4 projPos = projection * vec4(sphereCenter, 1.0);
   projPos /= projPos.w;
   vec2 screenUV = projPos.xy * 0.5 + 0.5;
   ```

2. **Estimate Projected Radius**:
   - Used to select appropriate mipmap level of depth pyramid
   - Higher mips (coarser) for large screen coverage, lower mips for small objects

3. **Sample Depth Pyramid**:
   ```glsl
   int mip = int(ceil(log2(projRadius_pixels)));
   float sampledDepth = textureLod(depthPyramid, screenUV, mip);
   ```

4. **Occlusion Test**:
   - If sphere depth > sampled depth: occluded (cull)
   - Small bias added for numerical stability

## Deferred Lighting Computation

**Compute Shader** (`deferred_lighting.comp`):

**Tiling Strategy**: 16x16 pixel tiles compute shared light lists
- Reduces redundant light testing
- Improves memory coherence

**PBR Lighting Model**:
```glsl
// Fresnel (Schlick approximation)
F = F0 + (1-F0) * (1-cos(theta))^5

// Normal Distribution (GGX)
D = a^2 / (π * ((N·H)^2 * (a^2-1) + 1)^2)

// Geometry (Schlick-GGX)
G = (N·V) / ((N·V) * (1-k) + k)

// Final: kD * albedo/π + D*F*G/(4*(N·V)*(N·L))
```

**Light Types**:
- **Directional**: Single direction, no attenuation
- **Point**: Quadratic distance attenuation
- **Spot**: Directional with cone angle falloff

## Performance Characteristics

| Feature | CPU Culling | GPU Culling | Benefit |
|---------|------------|-------------|---------|
| Frustum Tests | ~1000 objects/ms | ~100k objects/ms | 100x faster |
| Occlusion Tests | Can't do per-frame | Full framebuffer | Automatic |
| LOD Selection | Manual/heuristic | Distance-based | Automated |
| Memory Overhead | Visibility arrays | GPU SSBOs (32 bits/obj) | Minimal |

**Scalability**:
- Frustum culling: Linear O(n) with instance count
- Occlusion culling: O(1) per-pixel for screen-space operations
- Suitable for 10k-100k+ dynamic objects with GPU culling

## Integration with Existing Renderer

The hybrid renderer can coexist with the legacy deferred renderer:

```cpp
// Option 1: Use new hybrid renderer exclusively
auto renderer = std::make_unique<HybridRenderer>();
renderer->Initialize();
renderer->Render();

// Option 2: Migrate gradually - keep legacy as fallback
class Application {
    std::unique_ptr<HybridRenderer> m_HybridRenderer;
    std::unique_ptr<Renderer> m_LegacyRenderer;
    bool m_UseHybrid = true;
};
```

## Debug Visualization

The hybrid renderer supports several debug modes:

```cpp
renderer->SetShowCullingBounds(true);   // Render bounding boxes
renderer->SetShowGBuffer(true);          // Visualize G-Buffer channels
renderer->SetShowLightHeatmap(true);     // Show light coverage
renderer->SetRenderMode(RenderMode::HybridDebug);
```

## Shader Compilation

Compute shaders require GLSL 4.6+:
```cpp
#version 460 core
layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;
```

Compile with:
```bash
glCompileShader(shader);  // Uses GL_COMPUTE_SHADER type
glLinkProgram(program);   // No vertex/fragment stages
glDispatchCompute(groupsX, groupsY, groupsZ);
```

## Next Steps / TODO

1. **Complete RenderPass Implementations**:
   - ShadowPass (cascaded shadowmaps)
   - GeometryPass (G-Buffer fill from scene)
   - TransparentPass (forward-rendered objects)
   - PostProcessingPass (effects pipeline)

2. **Optimize Lighting**:
   - Implement ForwardPlus clustering
   - Implement TiledDeferred approach
   - Add shadow map sampling to lighting compute

3. **LOD System Integration**:
   - Connect GPU-computed LOD levels to mesh selection
   - Implement mesh streaming based on LOD

4. **Advanced Culling**:
   - Meshlet-level culling (requires hardware support)
   - Variable rate shading integration
   - Depth pre-pass optimization

5. **Performance Profiling**:
   - GPU query metrics for each stage
   - CPU-GPU sync point analysis
   - Memory bandwidth optimization

## References

- [Unity Scriptable Render Pipeline](https://docs.unity3d.com/Manual/srp-introduction.html)
- [GPU-Driven Rendering Pipelines (Jendrik Illner, Chaos Group)](https://www.nvidia.com/en-us/on-demand/session/gtcspring21-s32673/)
- [Frustum Culling (Real-Time Rendering)](https://www.gamedev.net/tutorials/programming/graphics/frustum-culling-in-the-wild-r4262/)
- [Depth Pyramid for Occlusion Culling](https://www.youtube.com/watch?v=1lLN_lFvnlc)
