# Next: Phase 2 - Renderer Integration

## Overview

Phase 1 (Foundation) is complete! Phase 2 focuses on integrating NVRHI into the core rendering pipeline.

## What to Do First

### 1. Create Window/Application Integration
**File**: Create `include/NVRHIRenderer.h` and `src/NVRHIRenderer.cpp`

This is the bridge between your Application and NVRHI:

```cpp
class NVRHIRenderer {
public:
    // Initialize NVRHI on startup
    bool Initialize(uint32_t width, uint32_t height, void* windowHandle,
                   Graphics::GraphicsBackend backend = Graphics::GraphicsBackend::D3D12);
    
    void Shutdown();
    
    // Core rendering
    void BeginFrame();
    void EndFrame();
    void Present();
    
    // Resource access
    Graphics::DevicePtr GetDevice() { return m_Device; }
    Graphics::CommandListPtr GetCommandList() { return m_CommandList; }
    
private:
    Graphics::DevicePtr m_Device;
    Graphics::CommandListPtr m_CommandList;
    Graphics::TexturePtr m_BackBuffer;
    Graphics::TexturePtr m_DepthBuffer;
};
```

### 2. Update Application Class
**File**: `src/Application.cpp`

In `Application::Init()`:
```cpp
bool Application::Init() {
    // ... existing code ...
    
    // Initialize NVRHI instead of OpenGL
    m_NVRHIRenderer = std::make_unique<NVRHIRenderer>();
    
    if (!m_NVRHIRenderer->Initialize(
        m_Window->GetWidth(),
        m_Window->GetHeight(),
        m_Window->GetNativeHandle(),
        Graphics::GraphicsBackend::D3D12)) {
        
        std::cerr << "Failed to initialize NVRHI renderer" << std::endl;
        return false;
    }
    
    // ... rest of initialization ...
    return true;
}
```

### 3. Update Renderer Class
**File**: `include/Renderer.h` and `src/Renderer.cpp`

Replace OpenGL initialization with NVRHI:

```cpp
bool Renderer::Init() {
    auto device = Graphics::GetDevice();
    if (!device) {
        std::cerr << "Graphics device not initialized" << std::endl;
        return false;
    }
    
    // Create G-Buffer textures using NVRHI
    Graphics::TextureDesc gbufferDesc;
    gbufferDesc.width = 1920;  // Get from device
    gbufferDesc.height = 1080;
    
    // Position buffer
    gbufferDesc.format = Graphics::TextureFormat::R32G32B32A32_FLOAT;
    m_GBufferPositions = device->CreateTexture(gbufferDesc);
    
    // Normal buffer
    gbufferDesc.format = Graphics::TextureFormat::R16G16B16A16_FLOAT;
    m_GBufferNormals = device->CreateTexture(gbufferDesc);
    
    // Albedo buffer
    gbufferDesc.format = Graphics::TextureFormat::R8G8B8A8_UNORM;
    m_GBufferAlbedo = device->CreateTexture(gbufferDesc);
    
    // ... create other buffers ...
    
    return true;
}
```

### 4. Update Shader System
**File**: `src/Shader.cpp`

For now, compile shaders to bytecode and load them:

```cpp
bool Shader::Compile(const std::string& source, ShaderType type) {
    // Compile HLSL to bytecode using DXC
    std::vector<uint8_t> bytecode = CompileHLSL(source, type);
    
    if (bytecode.empty()) {
        return false;
    }
    
    // Create NVRHI shader
    auto device = Graphics::GetDevice();
    Graphics::ShaderDesc shaderDesc;
    shaderDesc.type = ConvertShaderType(type);
    shaderDesc.language = Graphics::ShaderLanguage::HLSL;
    shaderDesc.bytecode = bytecode;
    
    m_NVRHIShader = device->CreateShader(shaderDesc);
    return m_NVRHIShader != nullptr;
}
```

### 5. Update GBuffer Rendering
**File**: `src/GBuffer.cpp` and `include/GBuffer.h`

Replace OpenGL framebuffer with NVRHI:

```cpp
void GBuffer::BindForWriting() {
    auto device = Graphics::GetDevice();
    auto cmdList = device->CreateCommandList();
    
    cmdList->Begin();
    cmdList->SetRenderTargets(
        {m_Positions, m_Normals, m_Albedo, m_Emissive},
        m_DepthBuffer
    );
    cmdList->ClearRenderTarget(m_Positions, {0, 0, 0, 0});
    cmdList->ClearRenderTarget(m_Normals, {0.5f, 0.5f, 1.0f, 0});
    cmdList->ClearRenderTarget(m_Albedo, {0, 0, 0, 1});
    cmdList->ClearDepthTarget(m_DepthBuffer, 1.0f);
}

void GBuffer::BindForReading() {
    auto device = Graphics::GetDevice();
    
    // Set as textures for lighting pass
    device->SetTexture(0, m_Positions);
    device->SetTexture(1, m_Normals);
    device->SetTexture(2, m_Albedo);
    device->SetTexture(3, m_Emissive);
}
```

## Integration Steps

### Step 1: Create NVRHIRenderer (1-2 hours)
1. Create the wrapper class
2. Implement Initialize/Shutdown
3. Handle swapchain management
4. Test basic creation

### Step 2: Update Application (1-2 hours)
1. Add NVRHIRenderer member
2. Initialize on startup
3. Hook up to render loop
4. Test application start

### Step 3: Port Texture Creation (2-3 hours)
1. Replace OpenGL glCreateTexture calls
2. Use Graphics::CreateTexture
3. Update all texture creation paths
4. Test texture loading

### Step 4: Port Shader System (3-4 hours)
1. Integrate DXC compiler
2. Update Shader class
3. Handle shader compilation
4. Test shader loading

### Step 5: Port GBuffer (4-5 hours)
1. Replace framebuffer with NVRHI
2. Update render target binding
3. Update lighting pass
4. Test G-Buffer rendering

### Step 6: Port Shadow Rendering (3-4 hours)
1. Update ShadowMap class
2. Update CascadedShadowMap
3. Update CubemapShadow
4. Test shadow rendering

### Step 7: Port Post-Processing (4-5 hours)
1. Update SSAO
2. Update SSR
3. Update TAA
4. Update Bloom
5. Test all effects

## Files to Create/Modify

### New Files
- `include/NVRHIRenderer.h` - Renderer wrapper
- `src/NVRHIRenderer.cpp` - Implementation
- `include/ShaderCompiler.h` - DXC integration
- `src/ShaderCompiler.cpp` - Implementation

### Files to Modify
- `src/Application.cpp` - Initialize NVRHI
- `include/Renderer.h` - Port to NVRHI
- `src/Renderer.cpp` - Port implementation
- `src/Shader.cpp` - Compile to bytecode
- `include/GBuffer.h` - NVRHI render targets
- `src/GBuffer.cpp` - NVRHI implementation
- `src/ShadowMap.cpp` - NVRHI render targets
- `src/PostProcessing.cpp` - NVRHI implementation

## Recommended Order

1. **Week 1**: NVRHIRenderer + Application integration
2. **Week 2**: Shader system + basic rendering
3. **Week 3**: GBuffer and shadow rendering
4. **Week 4**: Post-processing effects

## Key Considerations

### Backward Compatibility
- Keep OpenGL code initially
- Add preprocessor guards:
  ```cpp
  #ifdef USE_NVRHI
  // NVRHI code
  #else
  // OpenGL code
  #endif
  ```

### Performance
- Profile regularly with GPU debuggers (NSight, RenderDoc)
- Use GPU timestamps for performance monitoring
- Monitor GPU memory usage

### Testing
- Create unit tests for each system
- Test on multiple backends if possible
- Validate rendering results visually

## Common Pitfalls to Avoid

1. **Forgetting to call Begin()/End()** on command lists
2. **Mixing OpenGL and NVRHI** in same code
3. **Not handling format conversions** properly
4. **Forgetting to submit command lists**
5. **Not managing resource lifetimes** correctly

## Help & Resources

- Check `include/Graphics/GraphicsCompat.h` for helper functions
- Review NVRHI samples in official repository
- See `NVRHI_INTEGRATION_GUIDE.md` for detailed information
- Refer to `NVRHI_IMPLEMENTATION_CHECKLIST.md` for progress tracking

## Success Criteria

Phase 2 will be complete when:
- ✅ NVRHI renderer initializes successfully
- ✅ Shaders compile and load
- ✅ Textures render correctly
- ✅ G-Buffer captures scene data
- ✅ Lighting pass works
- ✅ All post-processing effects work
- ✅ Performance is equivalent or better than OpenGL version

---

**Estimated Duration**: 2-4 weeks (depending on complexity)
**Current Status**: Ready to begin
**Next Step**: Create NVRHIRenderer class
