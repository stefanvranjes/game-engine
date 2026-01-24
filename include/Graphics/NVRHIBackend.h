#pragma once

#include "Graphics/GraphicsDevice.h"
#include <nvrhi/nvrhi.h>
#include <memory>
#include <unordered_map>

namespace Graphics {

class NVRHIBuffer : public Buffer {
public:
    NVRHIBuffer(nvrhi::IBuffer* nvrhi_buffer, nvrhi::IDevice* device, const BufferDesc& desc);
    ~NVRHIBuffer() override;

    size_t GetSize() const override { return m_Desc.size; }
    BufferUsage GetUsage() const override { return m_Desc.usage; }
    void* Map() override;
    void Unmap() override;
    void UpdateData(const void* data, size_t size, size_t offset = 0) override;
    void* GetNativeHandle() override { return m_NVRHIBuffer; }

private:
    nvrhi::IBuffer* m_NVRHIBuffer;
    nvrhi::IDevice* m_Device;
    BufferDesc m_Desc;
    void* m_MappedPtr = nullptr;
};

class NVRHITexture : public Texture {
public:
    NVRHITexture(nvrhi::ITexture* nvrhi_texture, const TextureDesc& desc);
    ~NVRHITexture() override;

    uint32_t GetWidth() const override { return m_Desc.width; }
    uint32_t GetHeight() const override { return m_Desc.height; }
    uint32_t GetDepth() const override { return m_Desc.depth; }
    TextureFormat GetFormat() const override { return m_Desc.format; }
    void SetData(const void* data, size_t size, uint32_t mipLevel = 0, uint32_t arrayIndex = 0) override;
    void* GetData(uint32_t mipLevel = 0, uint32_t arrayIndex = 0) override;
    void GenerateMipMaps() override;
    void* GetNativeHandle() override { return m_NVRHITexture; }
    nvrhi::ITexture* GetNVRHITexture() const { return m_NVRHITexture; }

private:
    nvrhi::ITexture* m_NVRHITexture;
    TextureDesc m_Desc;
};

class NVRHIShader : public Shader {
public:
    NVRHIShader(nvrhi::IShader* nvrhi_shader, const ShaderDesc& desc);
    ~NVRHIShader() override;

    ShaderType GetType() const override { return m_Desc.type; }
    ShaderLanguage GetLanguage() const override { return m_Desc.language; }
    const std::vector<uint8_t>& GetBytecode() const override { return m_Desc.bytecode; }
    const std::string& GetSource() const override { return m_Desc.source; }
    void* GetNativeHandle() override { return m_NVRHIShader; }

private:
    nvrhi::IShader* m_NVRHIShader;
    ShaderDesc m_Desc;
};

class NVRHIPipeline : public Pipeline {
public:
    NVRHIPipeline(nvrhi::GraphicsPipelineHandle pipeline);
    ~NVRHIPipeline() override;

    void SetVertexShader(ShaderPtr shader) override;
    void SetFragmentShader(ShaderPtr shader) override;
    void SetComputeShader(ShaderPtr shader) override;
    ShaderPtr GetVertexShader() const override { return m_VertexShader; }
    ShaderPtr GetFragmentShader() const override { return m_FragmentShader; }
    ShaderPtr GetComputeShader() const override { return m_ComputeShader; }
    void* GetNativeHandle() override { return &m_Pipeline; }

    nvrhi::GraphicsPipelineHandle GetNVRHIPipeline() const { return m_Pipeline; }

private:
    nvrhi::GraphicsPipelineHandle m_Pipeline;
    ShaderPtr m_VertexShader;
    ShaderPtr m_FragmentShader;
    ShaderPtr m_ComputeShader;
};

class NVRHICommandList : public CommandList {
public:
    NVRHICommandList(nvrhi::ICommandList* cmdList, nvrhi::IDevice* device);
    ~NVRHICommandList() override;

    void Begin() override;
    void End() override;
    void Submit() override;

    void SetViewport(const ViewportRect& viewport) override;
    void SetScissorRect(uint32_t x, uint32_t y, uint32_t width, uint32_t height) override;
    
    void SetRenderTargets(const std::vector<TexturePtr>& renderTargets, TexturePtr depthTarget) override;
    void ClearRenderTarget(TexturePtr target, const ClearColor& color) override;
    void ClearDepthTarget(TexturePtr target, float depth) override;

    void SetPipeline(PipelinePtr pipeline) override;
    void SetConstantBuffer(uint32_t slot, BufferPtr buffer) override;
    void SetVertexBuffer(uint32_t slot, BufferPtr buffer, uint32_t stride) override;
    void SetIndexBuffer(BufferPtr buffer, bool is32Bit) override;
    void SetTexture(uint32_t slot, TexturePtr texture) override;
    void SetSampler(uint32_t slot, uint32_t samplerHandle) override;

    void DrawIndexed(uint32_t indexCount, uint32_t startIndexLocation = 0, int32_t baseVertexLocation = 0) override;
    void Draw(uint32_t vertexCount, uint32_t startVertexLocation = 0) override;
    void DrawInstanced(uint32_t vertexCountPerInstance, uint32_t instanceCount, uint32_t startVertexLocation = 0, uint32_t startInstanceLocation = 0) override;
    void DrawIndexedInstanced(uint32_t indexCountPerInstance, uint32_t instanceCount, uint32_t startIndexLocation = 0, int32_t baseVertexLocation = 0, uint32_t startInstanceLocation = 0) override;

    void Dispatch(uint32_t threadGroupCountX, uint32_t threadGroupCountY, uint32_t threadGroupCountZ) override;

    void TransitionResource(TexturePtr resource, ResourceAccess oldAccess, ResourceAccess newAccess) override;
    void TransitionBuffer(BufferPtr resource, ResourceAccess oldAccess, ResourceAccess newAccess) override;

    void CopyBuffer(BufferPtr src, BufferPtr dst, size_t size, size_t srcOffset = 0, size_t dstOffset = 0) override;
    void CopyTexture(TexturePtr src, TexturePtr dst) override;

    uint32_t BeginTimestamp() override;
    void EndTimestamp(uint32_t queryId) override;
    double GetTimestamp(uint32_t queryId) override;

    nvrhi::ICommandList* GetNVRHICommandList() const { return m_CommandList; }

private:
    nvrhi::ICommandList* m_CommandList;
    nvrhi::IDevice* m_Device;
    nvrhi::GraphicsState m_GraphicsState;
    nvrhi::ComputeState m_ComputeState;
    uint32_t m_TimestampCounter = 0;
    std::unordered_map<uint32_t, uint64_t> m_Timestamps;
};

class NVRHIDevice : public Device {
public:
    NVRHIDevice();
    ~NVRHIDevice() override;

    bool Initialize(GraphicsBackend backend, uint32_t width, uint32_t height, void* windowHandle) override;
    void Shutdown() override;
    GraphicsBackend GetBackend() const override { return m_Backend; }
    bool IsValid() const override { return m_Device != nullptr; }

    uint32_t GetScreenWidth() const override { return m_ScreenWidth; }
    uint32_t GetScreenHeight() const override { return m_ScreenHeight; }
    void Present(bool vsync = true) override;
    void Resize(uint32_t width, uint32_t height) override;

    BufferPtr CreateBuffer(const BufferDesc& desc) override;
    TexturePtr CreateTexture(const TextureDesc& desc) override;
    ShaderPtr CreateShader(const ShaderDesc& desc) override;
    PipelinePtr CreatePipeline() override;
    CommandListPtr CreateCommandList() override;

    void DestroyBuffer(BufferPtr& buffer) override;
    void DestroyTexture(TexturePtr& texture) override;
    void DestroyShader(ShaderPtr& shader) override;
    void DestroyPipeline(PipelinePtr& pipeline) override;

    TexturePtr GetBackBuffer() override;
    TexturePtr GetDepthBuffer() override;

    uint32_t CreateSampler(bool linearFilter = true, bool clampUVW = false) override;
    void DestroySampler(uint32_t samplerHandle) override;

    void WaitForIdle() override;
    void Flush() override;

    void SetDebugName(BufferPtr buffer, const std::string& name) override;
    void SetDebugName(TexturePtr texture, const std::string& name) override;
    void SetDebugName(ShaderPtr shader, const std::string& name) override;
    void SetDebugName(PipelinePtr pipeline, const std::string& name) override;
    
    uint64_t GetGPUMemoryUsed() const override;
    uint64_t GetGPUMemoryAvailable() const override;
    float GetGPUTemperature() const override;

    void* GetNativeDevice() override;
    void* GetNativeContext() override;

    nvrhi::IDevice* GetNVRHIDevice() const { return m_Device; }

private:
    nvrhi::IDevice* m_Device = nullptr;
    nvrhi::IFramebuffer* m_Framebuffer = nullptr;
    TexturePtr m_BackBuffer;
    TexturePtr m_DepthBuffer;
    GraphicsBackend m_Backend = GraphicsBackend::D3D12;
    uint32_t m_ScreenWidth = 1920;
    uint32_t m_ScreenHeight = 1080;
    void* m_WindowHandle = nullptr;

    nvrhi::Format ConvertFormat(TextureFormat format) const;
    TextureFormat ConvertFormatFromNVRHI(nvrhi::Format format) const;
    nvrhi::ShaderType ConvertShaderType(ShaderType type) const;
    nvrhi::GraphicsPipelineDesc CreateGraphicsPipelineDesc() const;
};

} // namespace Graphics
