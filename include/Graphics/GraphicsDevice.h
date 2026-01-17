#pragma once

#include "GraphicsCommon.h"
#include "GraphicsResource.h"
#include <glm/glm.hpp>
#include <functional>

namespace Graphics {

class Device {
public:
    virtual ~Device() = default;

    // Device initialization and management
    virtual bool Initialize(GraphicsBackend backend, uint32_t width, uint32_t height, void* windowHandle) = 0;
    virtual void Shutdown() = 0;
    virtual GraphicsBackend GetBackend() const = 0;
    virtual bool IsValid() const = 0;

    // Display management
    virtual uint32_t GetScreenWidth() const = 0;
    virtual uint32_t GetScreenHeight() const = 0;
    virtual void Present(bool vsync = true) = 0;
    virtual void Resize(uint32_t width, uint32_t height) = 0;

    // Resource creation
    virtual BufferPtr CreateBuffer(const BufferDesc& desc) = 0;
    virtual TexturePtr CreateTexture(const TextureDesc& desc) = 0;
    virtual ShaderPtr CreateShader(const ShaderDesc& desc) = 0;
    virtual PipelinePtr CreatePipeline() = 0;
    virtual CommandListPtr CreateCommandList() = 0;

    // Resource destruction
    virtual void DestroyBuffer(BufferPtr& buffer) = 0;
    virtual void DestroyTexture(TexturePtr& texture) = 0;
    virtual void DestroyShader(ShaderPtr& shader) = 0;
    virtual void DestroyPipeline(PipelinePtr& pipeline) = 0;

    // Swapchain management
    virtual TexturePtr GetBackBuffer() = 0;
    virtual TexturePtr GetDepthBuffer() = 0;

    // Sampler management
    virtual uint32_t CreateSampler(bool linearFilter = true, bool clampUVW = false) = 0;
    virtual void DestroySampler(uint32_t samplerHandle) = 0;

    // Synchronization
    virtual void WaitForIdle() = 0;
    virtual void Flush() = 0;

    // Debugging
    virtual void SetDebugName(BufferPtr buffer, const std::string& name) = 0;
    virtual void SetDebugName(TexturePtr texture, const std::string& name) = 0;
    virtual void SetDebugName(ShaderPtr shader, const std::string& name) = 0;
    virtual void SetDebugName(PipelinePtr pipeline, const std::string& name) = 0;
    
    // Performance monitoring
    virtual uint64_t GetGPUMemoryUsed() const = 0;
    virtual uint64_t GetGPUMemoryAvailable() const = 0;
    virtual float GetGPUTemperature() const = 0;

    // Native handle (for advanced usage)
    virtual void* GetNativeDevice() = 0;
    virtual void* GetNativeContext() = 0;
};

// Global device accessor
DevicePtr GetDevice();
void SetDevice(DevicePtr device);

// Factory function
DevicePtr CreateDevice(GraphicsBackend backend = GraphicsBackend::D3D12);

} // namespace Graphics
