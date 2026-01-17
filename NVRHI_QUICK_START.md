# NVRHI Integration - Quick Start Guide

## 5-Minute Overview

NVRHI (NVIDIA Rendering Hardware Interface) has been integrated into GameEngine. It's a modern graphics abstraction layer supporting D3D12, D3D11, and Vulkan from a single codebase.

### What You Need to Know

1. **No Breaking Changes Yet** - The engine still uses OpenGL
2. **Foundation Ready** - NVRHI is available and can be used
3. **Gradual Migration** - Integrate one system at a time
4. **Easy to Use** - Simple API very similar to OpenGL

## Getting Started

### 1. Build with NVRHI Support

```bash
# Default (D3D12 on Windows)
cmake -B build -DGRAPHICS_BACKEND=D3D12

# Or Vulkan for cross-platform
cmake -B build -DGRAPHICS_BACKEND=VULKAN

# Then build
cmake --build build --config Debug
```

### 2. Minimal Example

Create a simple graphics program:

```cpp
#include "Graphics/GraphicsInit.h"

int main() {
    // Initialize NVRHI
    auto device = Graphics::InitializeGraphics(
        Graphics::GraphicsBackend::D3D12,
        1920, 1080,
        windowHandle  // Your HWND or equivalent
    );
    
    if (!device) {
        std::cerr << "Failed to initialize graphics" << std::endl;
        return 1;
    }

    // Create a simple texture
    Graphics::TextureDesc textureDesc;
    textureDesc.width = 512;
    textureDesc.height = 512;
    textureDesc.format = Graphics::TextureFormat::R8G8B8A8_UNORM;
    
    auto texture = device->CreateTexture(textureDesc);

    // Create a constant buffer
    auto constantBuffer = device->CreateBuffer(
        Graphics::BufferDesc{
            .size = 256,
            .usage = Graphics::BufferUsage::ConstantBuffer,
            .cpuAccess = true
        }
    );

    // Record some commands
    auto cmdList = device->CreateCommandList();
    cmdList->Begin();
    cmdList->SetViewport({0, 0, 1920, 1080});
    cmdList->SetConstantBuffer(0, constantBuffer);
    cmdList->End();
    cmdList->Submit();

    // Cleanup
    Graphics::ShutdownGraphics();
    
    return 0;
}
```

### 3. Using Migration Helpers

For faster integration, use the compatibility helpers:

```cpp
#include "Graphics/GraphicsCompat.h"

// Instead of learning NVRHI details, use familiar patterns:
auto vertexBuffer = Graphics::GLToNVRHIAdapter::CreateVertexBuffer(
    vertexData.size(),
    vertexData.data()
);

auto texture = Graphics::GLToNVRHIAdapter::CreateTexture2D(
    width, height,
    Graphics::TextureFormat::R8G8B8A8_UNORM,
    pixelData
);

// Or use convenience macros:
auto constantBuffer = CREATE_CONSTANT_BUFFER(256, nullptr);
```

## File Structure

```
include/Graphics/
├── GraphicsCommon.h       ← Types, enums, descriptors
├── GraphicsResource.h     ← Abstract resource classes
├── GraphicsDevice.h       ← Abstract device class
├── NVRHIBackend.h         ← NVRHI wrapper classes
├── GraphicsInit.h         ← Easy initialization
└── GraphicsCompat.h       ← OpenGL→NVRHI helpers

src/
└── NVRHIBackend.cpp       ← Full implementation

Documentation/
├── NVRHI_INTEGRATION_GUIDE.md       ← Detailed guide
├── NVRHI_IMPLEMENTATION_CHECKLIST.md ← Phase breakdown
├── NVRHI_PHASE1_SUMMARY.md          ← What's done
└── NVRHI_QUICK_START.md             ← This file
```

## Key APIs at a Glance

### Initialization
```cpp
auto device = Graphics::InitializeGraphics(backend, width, height, windowHandle);
auto device = Graphics::GetGraphicsDevice();  // Get active device
Graphics::ShutdownGraphics();                 // Cleanup
```

### Resource Creation
```cpp
auto texture = device->CreateTexture(desc);
auto buffer = device->CreateBuffer(desc);
auto shader = device->CreateShader(desc);
auto pipeline = device->CreatePipeline();
```

### Command Recording
```cpp
auto cmdList = device->CreateCommandList();
cmdList->Begin();
cmdList->SetViewport(viewport);
cmdList->SetRenderTargets({...}, depthTarget);
cmdList->SetPipeline(pipeline);
cmdList->DrawIndexed(indexCount);
cmdList->End();
cmdList->Submit();
```

### Resource Updates
```cpp
buffer->UpdateData(data, size, offset);
texture->SetData(pixelData, dataSize);
texture->GenerateMipMaps();
```

## Common Patterns

### Create a Color Render Target
```cpp
auto colorTarget = Graphics::GLToNVRHIAdapter::CreateRenderTarget(
    1920, 1080,
    Graphics::TextureFormat::R8G8B8A8_UNORM
);
```

### Create a Depth Target
```cpp
auto depthTarget = Graphics::GLToNVRHIAdapter::CreateDepthTarget(1920, 1080);
```

### Create Vertex Buffer
```cpp
std::vector<Vertex> vertices = {...};
auto vb = Graphics::GLToNVRHIAdapter::CreateVertexBuffer(
    vertices.size() * sizeof(Vertex),
    vertices.data()
);
```

### Create Dynamic Constant Buffer
```cpp
auto cb = Graphics::GLToNVRHIAdapter::CreateConstantBuffer(256, nullptr);

// Update per-frame
void* mapped = Graphics::GLToNVRHIAdapter::MapBuffer(cb);
std::memcpy(mapped, &cbData, sizeof(cbData));
Graphics::GLToNVRHIAdapter::UnmapBuffer(cb);
```

## Supported Graphics Backends

### D3D12 (Recommended for Windows)
- ✅ Highest performance on Windows 10/11
- ✅ Modern API design
- ✅ Best driver support
- ✅ Supports latest NVIDIA features

### D3D11 (Legacy Windows)
- ✅ Works on Windows 7+
- ✅ Fallback for older systems
- ⚠️ Slightly lower performance

### Vulkan (Cross-platform)
- ✅ Linux/Mac/Windows support
- ✅ Open standard
- ✅ Good performance
- ⚠️ More verbose API

## System Requirements

### Windows
- Windows 7 or later
- GPU with DirectX 11 or 12 support (D3D11 or D3D12 backend)
- Latest graphics driver

### Linux/Mac
- Vulkan 1.2 or later
- GPU with Vulkan support
- Latest graphics driver

## Choosing a Backend

**Use D3D12 if:**
- Targeting Windows 10/11 only
- Want maximum performance
- Need NVIDIA-specific features

**Use D3D11 if:**
- Need Windows 7 support
- Targeting older hardware
- Simplicity is more important than performance

**Use Vulkan if:**
- Need cross-platform support
- Targeting Linux/Mac
- Want open standard compatibility

## Next Steps for Integration

### Short-term (This week)
1. ✅ Review this quick start
2. ✅ Check out `NVRHI_INTEGRATION_GUIDE.md` for details
3. 📋 Begin integrating into Application/Window classes

### Mid-term (Next 1-2 weeks)
1. Port TextureManager to NVRHI
2. Update Shader system for NVRHI
3. Port Renderer to use NVRHI

### Long-term (Next month)
1. Complete deferred rendering pipeline
2. Port all post-processing effects
3. Optimize and profile
4. Add runtime backend selection UI

## Troubleshooting

### "Failed to create graphics device"
- Ensure GPU supports the selected backend
- Check drivers are up to date
- Try a different backend (Vulkan as fallback)

### Build fails with NVRHI not found
- NVRHI fetches from GitHub - check internet connection
- Try deleting `_deps/nvrhi*` and rebuilding
- Ensure CMake is recent enough

### Low performance
- Profile with GPU debugger (NSight, RenderDoc)
- Check GPU is being used (not CPU fallback)
- Verify correct backend is loaded

## Additional Resources

- **NVRHI GitHub**: https://github.com/NVIDIA-Omniverse/nvrhi
- **NVRHI Samples**: Check NVRHI repo for sample code
- **DirectX Docs**: https://docs.microsoft.com/en-us/windows/win32/direct3d12/
- **Vulkan Spec**: https://www.khronos.org/vulkan/

## Questions?

Refer to:
1. `NVRHI_INTEGRATION_GUIDE.md` - Comprehensive guide
2. `NVRHI_IMPLEMENTATION_CHECKLIST.md` - Phase-by-phase plan
3. Header file documentation - Inline API docs
4. NVRHI official repository - Official examples

---

**Ready to use NVRHI! Start with the examples above and gradually integrate into your systems.** 🚀
