# Hybrid Renderer Quick Reference

## Quick Start

### Basic Setup

```cpp
#include "HybridRenderer.h"

// In Application::Init()
auto renderer = std::make_unique<HybridRenderer>();
renderer->Initialize();
renderer->SetCamera(m_Camera.get());
renderer->SetSceneRoot(m_SceneRoot);

// In Application::Update(float dt)
renderer->Update(dt);

// In Application::Render()
renderer->Render();
```

### Adding Lights

```cpp
Light directionalLight;
directionalLight.type = LightType::Directional;
directionalLight.direction = glm::vec3(0, -1, 0);
directionalLight.color = glm::vec3(1, 1, 1);
directionalLight.intensity = 1.5f;

renderer->AddLight(directionalLight);
```

### Controlling Rendering Modes

```cpp
// Change rendering mode at runtime
renderer->SetRenderMode(HybridRenderer::RenderMode::HybridOptimized);

// Change lighting strategy
renderer->SetLightingMode(HybridRenderer::LightingMode::Deferred);

// Toggle GPU culling
renderer->SetGPUCullingEnabled(true);
```

## GPU Culling System

### Direct Usage (Standalone)

```cpp
#include "GPUCullingSystem.h"

GPUCullingSystem culling;
culling.Initialize();

// Per frame:
culling.SetupCulling(viewMat, projMat, cameraPos, instanceCount);

// Populate cull data
std::vector<GPUCullingSystem::CullData> data;
for(auto& obj : objects) {
    GPUCullingSystem::CullData cd;
    cd.modelMatrix = obj->GetWorldMatrix();
    cd.boundingSphere = glm::vec4(center, radius);
    cd.aabbMin = glm::vec4(min, 0);
    cd.aabbMax = glm::vec4(max, 0);
    data.push_back(cd);
}

culling.SetCullData(data);

// Run culling pipeline
culling.ExecuteAll(depthTextureHandle);

// Get results
auto results = culling.GetResults();
for(uint32_t visibleIdx : results.visibleIndices) {
    // Render object at visibleIdx
}
```

## Render Pipeline (SRP-like)

### Creating Custom Passes

```cpp
#include "RenderPass.h"

class MyCustomPass : public RenderPass {
public:
    MyCustomPass() : RenderPass("MyPass", PassType::Custom) {}
    
    bool Initialize() override {
        // Load shaders, setup resources
        return true;
    }
    
    void Execute(const PassContext& ctx) override {
        // Cast ctx.userData to get pass-specific data
        Shader* shader = GetShader();
        if(!shader) return;
        
        shader->Use();
        shader->SetMat4("u_View", ctx.viewMatrix);
        shader->SetMat4("u_Projection", ctx.projectionMatrix);
        
        // Render your content
    }
    
    void Shutdown() override {
        // Cleanup
    }
};

// Add to pipeline
auto pipeline = renderer->GetPipeline();
auto pass = std::make_unique<MyCustomPass>();
pass->Initialize();
pipeline->AddPass(std::move(pass));
```

### Reordering Passes

```cpp
// Get pipeline
auto pipeline = renderer->GetPipeline();

// Remove a pass
pipeline->RemovePass("LightingPass");

// Add custom pass in its place
auto customLighting = std::make_unique<CustomLightingPass>();
pipeline->AddPass(std::move(customLighting));
```

## Debug Visualization

```cpp
// Show bounding boxes of culled objects
renderer->SetShowCullingBounds(true);

// Visualize G-Buffer channels
renderer->SetShowGBuffer(true);

// Show light coverage heatmap
renderer->SetShowLightHeatmap(true);

// Use debug rendering mode
renderer->SetRenderMode(HybridRenderer::RenderMode::HybridDebug);
```

## Performance Profiling

### GPU Timing

```cpp
// Get culling system timing
float computeTime = renderer->GetCullingSystem()->GetLastComputeTime();
uint32_t visibleCount = renderer->GetCullingSystem()->GetVisibleInstanceCount();

printf("Culled to %u visible objects in %.2f ms\n", visibleCount, computeTime);
```

### Frame Statistics

```cpp
// Check what's being rendered
auto cullingResults = renderer->GetCullingSystem()->GetResults();
printf("Culled: %u / Visible: %u\n", 
       cullingResults.culledCount, 
       cullingResults.visibleCount);
```

## Common Configurations

### Mobile/Low-End (CPU Bound)
```cpp
renderer->SetRenderMode(HybridRenderer::RenderMode::ForwardOnly);
renderer->SetGPUCullingEnabled(false);  // Use CPU culling instead
renderer->GetPostProcessing()->SetEnabled(false);
```

### Desktop (GPU Bound)
```cpp
renderer->SetRenderMode(HybridRenderer::RenderMode::HybridOptimized);
renderer->SetGPUCullingEnabled(true);
renderer->SetLightingMode(HybridRenderer::LightingMode::Deferred);
```

### High-End (Many Lights)
```cpp
renderer->SetRenderMode(HybridRenderer::RenderMode::HybridOptimized);
renderer->SetLightingMode(HybridRenderer::LightingMode::ForwardPlus);  // Clustered
renderer->SetGPUCullingEnabled(true);
```

## Shader Integration

### G-Buffer Sampling in Custom Shaders

```glsl
// In custom fragment/compute shader
layout(binding = 0) uniform sampler2D u_GBufferPosition;
layout(binding = 1) uniform sampler2D u_GBufferNormal;
layout(binding = 2) uniform sampler2D u_GBufferAlbedo;

void main() {
    vec3 pos = texture(u_GBufferPosition, uv).rgb;
    vec3 normal = normalize(texture(u_GBufferNormal, uv).rgb);
    vec3 albedo = texture(u_GBufferAlbedo, uv).rgb;
    
    // Custom lighting computation
    vec3 lit = DoMyCustomLighting(pos, normal, albedo);
    imageStore(output, pixelCoord, vec4(lit, 1.0));
}
```

### Light Buffer Access

```glsl
struct Light {
    vec4 position;       // xyz=pos, w=type
    vec4 direction;      // xyz=dir
    vec4 colorIntensity; // rgb=color, a=intensity
    vec4 params;         // x=range, y=spotAngle, z=attenuation, w=bias
};

layout(std140, binding = 0) uniform LightBuffer {
    Light lights[MAX_LIGHTS];
    int lightCount;
};

void main() {
    vec3 lit = vec3(0);
    for(int i = 0; i < lightCount; i++) {
        lit += ComputeLightContribution(lights[i], ...);
    }
}
```

## Troubleshooting

### GPU Culling Not Working
- Check compute shader compilation: `glGetShaderInfoLog()`
- Verify buffer bindings: `glGetInteger(GL_SHADER_STORAGE_BUFFER_BINDING)`
- Enable debug mode: `cullingSystem->SetDebugMode(true)`

### Black Screen / No Rendering
- Verify G-Buffer is properly bound: `m_GBuffer->BindForWriting()`
- Check lighting shader compilation
- Ensure camera is set: `renderer->SetCamera(camera)`

### Performance Issues
- Profile GPU with `GetVisibleInstanceCount()` and `GetLastComputeTime()`
- Check if CPU-GPU sync is bottleneck (read results too early)
- Consider disabling occlusion culling if depth pyramid creation is slow

### Artifacts / Incorrect Rendering
- Verify frustum plane extraction from view-projection matrix
- Check depth pyramid mipmap generation
- Test with simpler scenes first (single sphere)

## API Comparison: Legacy vs Hybrid

| Feature | Legacy Renderer | Hybrid Renderer |
|---------|-----------------|-----------------|
| Culling | CPU frustum | GPU compute (frustum + occlusion) |
| Lighting | Screen-space deferred | Screen-space + forward (hybrid) |
| LOD | Manual | GPU-computed |
| Post-FX | Fixed pipeline | Modular passes |
| Extensibility | Limited | SRP-like modularity |
| Transparency | Forward blend | Forward pass |
| Max Lights | 32 | Unlimited (with clustering) |

## Migration from Legacy Renderer

```cpp
// Old way
m_Renderer->SetCamera(camera);
m_Renderer->Render();

// New way (compatible)
m_HybridRenderer->SetCamera(camera);
m_HybridRenderer->SetSceneRoot(sceneRoot);
m_HybridRenderer->Render();

// Gradually migrate passes:
// 1. Keep legacy renderer, use hybrid for new features
// 2. Replace one pass at a time
// 3. Test thoroughly between each replacement
```

## References

- [HYBRID_RENDERING_GUIDE.md](HYBRID_RENDERING_GUIDE.md) - Full architecture documentation
- [RenderPass.h](include/RenderPass.h) - SRP API
- [GPUCullingSystem.h](include/GPUCullingSystem.h) - Culling API
- [HybridRenderer.h](include/HybridRenderer.h) - Main renderer API
