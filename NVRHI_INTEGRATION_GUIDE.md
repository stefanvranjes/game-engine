# NVRHI Integration Guide

## Overview

This document describes the NVRHI (NVIDIA Rendering Hardware Interface) integration into the GameEngine. NVRHI provides a modern, abstracted graphics API that supports multiple backends (D3D12, D3D11, Vulkan) from a single codebase.

## Architecture

### Graphics Abstraction Layer

The integration follows a three-level abstraction:

```
User Code (Renderer.h, TextureManager, etc.)
    ↓
Graphics Abstraction (Graphics::Device, Graphics::CommandList, Graphics::Resource)
    ↓
NVRHI Backend (NVRHIDevice, NVRHICommandList, NVRHIBuffer, etc.)
    ↓
Graphics API (D3D12, D3D11, Vulkan)
```

### Key Components

#### 1. **GraphicsCommon.h**
Defines common types, enums, and data structures:
- `GraphicsBackend` enum (D3D12, D3D11, Vulkan)
- `BufferUsage`, `TextureFormat`, `ResourceAccess` enums
- `BufferDesc`, `TextureDesc`, `ShaderDesc` structures
- Smart pointer type aliases

#### 2. **GraphicsResource.h**
Abstract base classes for graphics resources:
- `Buffer` - GPU buffer abstraction
- `Texture` - GPU texture abstraction
- `Shader` - Shader program abstraction
- `Pipeline` - Graphics/compute pipeline
- `CommandList` - Command recording interface

#### 3. **GraphicsDevice.h**
Abstract device interface:
- Device initialization and management
- Resource creation and destruction
- Swapchain management
- Sampler creation
- Performance monitoring
- GPU memory queries

#### 4. **NVRHIBackend.h & NVRHIBackend.cpp**
NVRHI implementation of the abstract interfaces:
- `NVRHIBuffer` - Wraps `nvrhi::IBuffer`
- `NVRHITexture` - Wraps `nvrhi::ITexture`
- `NVRHIShader` - Wraps `nvrhi::IShader`
- `NVRHIPipeline` - Wraps `nvrhi::GraphicsPipelineHandle`
- `NVRHICommandList` - Wraps `nvrhi::ICommandList`
- `NVRHIDevice` - Wraps `nvrhi::IDevice`

#### 5. **GraphicsInit.h**
Convenience functions for initialization:
- `InitializeGraphics()` - Setup NVRHI with specified backend
- `ShutdownGraphics()` - Cleanup
- `GetGraphicsDevice()` - Access current device
- `GetBackendName()` - Human-readable backend names

## Usage Example

```cpp
#include "Graphics/GraphicsInit.h"

// Initialize
auto device = Graphics::InitializeGraphics(
    Graphics::GraphicsBackend::D3D12,
    1920, 1080,
    hwndWindowHandle
);

if (!device) {
    // Handle error
    return false;
}

// Create a texture
Graphics::TextureDesc textureDesc;
textureDesc.width = 512;
textureDesc.height = 512;
textureDesc.format = Graphics::TextureFormat::R8G8B8A8_UNORM;

auto texture = device->CreateTexture(textureDesc);

// Create a buffer
Graphics::BufferDesc bufferDesc;
bufferDesc.size = 256 * 1024;
bufferDesc.usage = Graphics::BufferUsage::ConstantBuffer;
bufferDesc.cpuAccess = true;

auto constantBuffer = device->CreateBuffer(bufferDesc);

// Create a command list
auto cmdList = device->CreateCommandList();

// Record commands
cmdList->Begin();
cmdList->SetViewport({0, 0, 1920, 1080});
cmdList->SetConstantBuffer(0, constantBuffer);
cmdList->Draw(3); // Draw 3 vertices
cmdList->End();
cmdList->Submit();

// Cleanup
Graphics::ShutdownGraphics();
```

## Building

The NVRHI library is automatically fetched from GitHub during the CMake configuration:

```bash
# Configure (Windows)
cmake -B build -DGRAPHICS_BACKEND=D3D12

# Build
cmake --build build --config Release

# Or use the provided build scripts
build.bat
```

### Graphics Backend Selection

Set the `GRAPHICS_BACKEND` option during CMake configuration:

```bash
# D3D12 (default, Windows only)
cmake -DGRAPHICS_BACKEND=D3D12

# D3D11 (Windows only)
cmake -DGRAPHICS_BACKEND=D3D11

# Vulkan (cross-platform)
cmake -DGRAPHICS_BACKEND=VULKAN
```

## Integration with Existing Engine Systems

### Renderer Integration (TODO)

The `Renderer` class needs to be refactored to use `Graphics::Device` instead of direct OpenGL calls:

```cpp
// Old (OpenGL)
glCreateTexture(GL_TEXTURE_2D, &texture);
glBindTexture(GL_TEXTURE_2D, texture);

// New (NVRHI)
auto device = Graphics::GetGraphicsDevice();
auto texture = device->CreateTexture(textureDesc);
```

### TextureManager Integration (TODO)

Update `TextureManager` to use the graphics abstraction:

```cpp
// In TextureManager::LoadTexture()
auto device = Graphics::GetGraphicsDevice();
auto texture = device->CreateTexture(desc);
// ... load image data ...
texture->SetData(pixelData, dataSize);
```

### Shader System Integration (TODO)

Update shader compilation and management:

```cpp
// Compile HLSL to DXIL for D3D12, or to SPIR-V for Vulkan
Graphics::ShaderDesc shaderDesc;
shaderDesc.type = Graphics::ShaderType::Vertex;
shaderDesc.language = Graphics::ShaderLanguage::HLSL;
shaderDesc.entryPoint = "main";
shaderDesc.bytecode = compiledBytecode;

auto shader = device->CreateShader(shaderDesc);
```

## Shader Compilation

NVRHI supports multiple shader compilation paths:

### HLSL (Recommended for D3D backends)
- Compile directly using FXC or DXC
- Output: DXIL bytecode for D3D12/D3D11
- Tools: `dxc.exe` (installed with Windows SDK)

### GLSL → SPIR-V (For Vulkan)
- Compile GLSL using `glslang` or `DXC`
- Output: SPIR-V binary
- Cross-compilation: HLSL → SPIR-V via DXC

### Setup in NVRHIBackend.cpp

```cpp
// In NVRHIDevice::CreateShader()
switch (desc.language) {
    case ShaderLanguage::HLSL:
        // Use DXC or FXC to compile
        // Output DXIL for D3D, SPIR-V for Vulkan
        break;
    case ShaderLanguage::GLSL:
        // Use glslang to compile to SPIR-V
        break;
    case ShaderLanguage::SPIR_V:
        // Direct SPIR-V bytecode
        break;
}
```

## Performance Considerations

### Multi-Backend Performance
- **D3D12**: Best performance on Windows, modern API
- **D3D11**: Fallback for older Windows versions
- **Vulkan**: Cross-platform, excellent Linux support

### Memory Management
- Use `GetGPUMemoryUsed()` and `GetGPUMemoryAvailable()` for profiling
- Implement resource pooling for frequently created/destroyed resources
- Consider resident set management for large texture sets

### Command Recording
- Pool command lists to avoid allocation overhead
- Use `CommandList::Begin()` and `End()` for recording
- Submit in batches for efficiency

## Future Enhancements

### NVIDIA-Specific Features
1. **DLSS Integration** - Deep Learning Super Sampling
2. **OptiX Support** - Ray tracing
3. **NVIDIA FXAA** - Fast approximate anti-aliasing
4. **NVIDIA GFXBench** - Performance benchmarking

### Additional Features
1. **Mesh Shaders** - Modern GPU geometry processing
2. **Variable Rate Shading** - Efficiency improvements
3. **Sampler Feedback** - Streaming optimization
4. **GPU Debugging** - Built-in profiling and debugging

## File Structure

```
include/
  Graphics/
    GraphicsCommon.h      - Type definitions and enums
    GraphicsResource.h    - Abstract resource interfaces
    GraphicsDevice.h      - Abstract device interface
    NVRHIBackend.h        - NVRHI implementation declarations
    GraphicsInit.h        - Convenience initialization functions

src/
  NVRHIBackend.cpp        - NVRHI implementation definitions

CMakeLists.txt            - NVRHI fetch and linking configuration
```

## Troubleshooting

### Build Errors

**"Could not find nvrhi"**
- NVRHI may not have finished fetching
- Run `cmake --build build` again
- Check internet connection for GitHub access

### Runtime Errors

**"Failed to create graphics device"**
- Ensure graphics backend is supported by hardware
- Check GPU driver is up to date
- Verify DXGI/Vulkan runtime is installed

**"Failed to create shader"**
- Check shader bytecode is valid for the backend
- Ensure shader entry point matches (`main` for GLSL, configurable for HLSL)
- Verify shader type matches declaration

## Related Documentation

- [NVRHI GitHub Repository](https://github.com/NVIDIA-Omniverse/nvrhi)
- [NVRHI API Documentation](https://github.com/NVIDIA-Omniverse/nvrhi/wiki)
- [DirectX 12 Documentation](https://docs.microsoft.com/en-us/windows/win32/direct3d12/direct3d-12-graphics)
- [Vulkan Specification](https://www.khronos.org/vulkan/)

## Authors & Contributors

- NVRHI: NVIDIA Omniverse Team
- Integration: GameEngine Development Team
