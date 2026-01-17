#pragma once

#include "GraphicsCommon.h"
#include <vector>

namespace Graphics {

class Buffer {
public:
    virtual ~Buffer() = default;

    virtual size_t GetSize() const = 0;
    virtual BufferUsage GetUsage() const = 0;
    virtual void* Map() = 0;
    virtual void Unmap() = 0;
    virtual void UpdateData(const void* data, size_t size, size_t offset = 0) = 0;
    virtual void* GetNativeHandle() = 0;
};

class Texture {
public:
    virtual ~Texture() = default;

    virtual uint32_t GetWidth() const = 0;
    virtual uint32_t GetHeight() const = 0;
    virtual uint32_t GetDepth() const = 0;
    virtual TextureFormat GetFormat() const = 0;
    virtual void SetData(const void* data, size_t size, uint32_t mipLevel = 0, uint32_t arrayIndex = 0) = 0;
    virtual void* GetData(uint32_t mipLevel = 0, uint32_t arrayIndex = 0) = 0;
    virtual void GenerateMipMaps() = 0;
    virtual void* GetNativeHandle() = 0;
};

class Shader {
public:
    virtual ~Shader() = default;

    virtual ShaderType GetType() const = 0;
    virtual ShaderLanguage GetLanguage() const = 0;
    virtual const std::vector<uint8_t>& GetBytecode() const = 0;
    virtual const std::string& GetSource() const = 0;
    virtual void* GetNativeHandle() = 0;
};

class Pipeline {
public:
    virtual ~Pipeline() = default;

    virtual void SetVertexShader(ShaderPtr shader) = 0;
    virtual void SetFragmentShader(ShaderPtr shader) = 0;
    virtual void SetComputeShader(ShaderPtr shader) = 0;
    virtual ShaderPtr GetVertexShader() const = 0;
    virtual ShaderPtr GetFragmentShader() const = 0;
    virtual ShaderPtr GetComputeShader() const = 0;
    virtual void* GetNativeHandle() = 0;
};

class CommandList {
public:
    virtual ~CommandList() = default;

    // Command recording
    virtual void Begin() = 0;
    virtual void End() = 0;
    virtual void Submit() = 0;

    // Rendering commands
    virtual void SetViewport(const ViewportRect& viewport) = 0;
    virtual void SetScissorRect(uint32_t x, uint32_t y, uint32_t width, uint32_t height) = 0;
    
    // Render target management
    virtual void SetRenderTargets(const std::vector<TexturePtr>& renderTargets, TexturePtr depthTarget) = 0;
    virtual void ClearRenderTarget(TexturePtr target, const ClearColor& color) = 0;
    virtual void ClearDepthTarget(TexturePtr target, float depth) = 0;

    // Pipeline and resources
    virtual void SetPipeline(PipelinePtr pipeline) = 0;
    virtual void SetConstantBuffer(uint32_t slot, BufferPtr buffer) = 0;
    virtual void SetVertexBuffer(uint32_t slot, BufferPtr buffer, uint32_t stride) = 0;
    virtual void SetIndexBuffer(BufferPtr buffer, bool is32Bit) = 0;
    virtual void SetTexture(uint32_t slot, TexturePtr texture) = 0;
    virtual void SetSampler(uint32_t slot, uint32_t samplerHandle) = 0;

    // Draw calls
    virtual void DrawIndexed(uint32_t indexCount, uint32_t startIndexLocation = 0, int32_t baseVertexLocation = 0) = 0;
    virtual void Draw(uint32_t vertexCount, uint32_t startVertexLocation = 0) = 0;
    virtual void DrawInstanced(uint32_t vertexCountPerInstance, uint32_t instanceCount, uint32_t startVertexLocation = 0, uint32_t startInstanceLocation = 0) = 0;
    virtual void DrawIndexedInstanced(uint32_t indexCountPerInstance, uint32_t instanceCount, uint32_t startIndexLocation = 0, int32_t baseVertexLocation = 0, uint32_t startInstanceLocation = 0) = 0;

    // Compute dispatch
    virtual void Dispatch(uint32_t threadGroupCountX, uint32_t threadGroupCountY, uint32_t threadGroupCountZ) = 0;

    // Resource transitions
    virtual void TransitionResource(TexturePtr resource, ResourceAccess oldAccess, ResourceAccess newAccess) = 0;
    virtual void TransitionBuffer(BufferPtr resource, ResourceAccess oldAccess, ResourceAccess newAccess) = 0;

    // Copy operations
    virtual void CopyBuffer(BufferPtr src, BufferPtr dst, size_t size, size_t srcOffset = 0, size_t dstOffset = 0) = 0;
    virtual void CopyTexture(TexturePtr src, TexturePtr dst) = 0;

    // Timestamps and queries
    virtual uint32_t BeginTimestamp() = 0;
    virtual void EndTimestamp(uint32_t queryId) = 0;
    virtual double GetTimestamp(uint32_t queryId) = 0;
};

} // namespace Graphics
