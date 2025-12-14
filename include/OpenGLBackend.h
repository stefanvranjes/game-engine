#pragma once

#include "RenderBackend.h"
#include <map>
#include <queue>

/**
 * @brief OpenGL 3.3+ backend implementation
 * 
 * Wraps existing OpenGL rendering code with the RenderBackend interface.
 * Maintains compatibility with current engine implementation.
 */
class OpenGLBackend : public RenderBackend {
public:
    OpenGLBackend();
    ~OpenGLBackend() override;

    // Initialization
    bool Init(uint32_t width, uint32_t height, void* windowHandle) override;
    void Shutdown() override;

    API GetAPI() const override { return API::OpenGL; }
    std::string GetAPIName() const override;

    // GPU Device Management (OpenGL always single GPU)
    uint32_t GetDeviceCount() const override { return 1; }
    GPUDeviceInfo GetDeviceInfo(uint32_t deviceIndex) const override;
    void SetActiveDevice(uint32_t deviceIndex) override {} // No-op
    uint32_t GetActiveDevice() const override { return 0; }
    bool SupportsLinkedGPUs() const override { return false; }

    // Capability Queries
    bool SupportsFeature(const std::string& featureName) const override;
    bool SupportsRayTracing() const override { return false; }
    bool SupportsMeshShaders() const override { return false; }
    uint32_t GetMaxLights() const override { return 32; }
    uint32_t GetMaxTextureSize() const override;
    glm::uvec2 GetMaxFramebufferSize() const override;

    // Resource Creation
    std::shared_ptr<RenderResource> CreateBuffer(
        size_t size,
        const void* data = nullptr,
        uint32_t usageFlags = 0) override;

    std::shared_ptr<RenderResource> CreateTexture(
        uint32_t width,
        uint32_t height,
        const std::string& format,
        const void* data = nullptr,
        bool generateMips = true) override;

    std::shared_ptr<RenderResource> CreateTexture3D(
        uint32_t width,
        uint32_t height,
        uint32_t depth,
        const std::string& format,
        const void* data = nullptr) override;

    std::shared_ptr<RenderResource> CreateCubemap(
        uint32_t size,
        const std::string& format,
        const std::vector<const void*>& faceImages = {}) override;

    std::shared_ptr<RenderResource> CreateFramebuffer(
        uint32_t width,
        uint32_t height,
        const std::vector<std::string>& colorFormats,
        const std::string& depthFormat = "") override;

    std::shared_ptr<RenderResource> CreateShader(
        const std::string& source,
        ShaderType type) override;

    std::shared_ptr<RenderResource> CreatePipeline(
        const void* config) override;

    // Resource Updates
    void UpdateBuffer(
        const std::shared_ptr<RenderResource>& buffer,
        size_t offset,
        size_t size,
        const void* data) override;

    void UpdateTexture(
        const std::shared_ptr<RenderResource>& texture,
        uint32_t x,
        uint32_t y,
        uint32_t width,
        uint32_t height,
        const void* data) override;

    void CopyBuffer(
        const std::shared_ptr<RenderResource>& src,
        const std::shared_ptr<RenderResource>& dst,
        size_t size,
        size_t srcOffset = 0,
        size_t dstOffset = 0) override;

    void CopyBufferToTexture(
        const std::shared_ptr<RenderResource>& buffer,
        const std::shared_ptr<RenderResource>& texture,
        uint32_t width,
        uint32_t height) override;

    // Resource Cleanup
    void DestroyResource(const std::shared_ptr<RenderResource>& resource) override;
    void WaitForGPU() override;

    // Rendering Commands
    void BeginCommandBuffer() override;
    void EndCommandBuffer() override;
    void BeginRenderPass(
        const std::shared_ptr<RenderResource>& framebuffer,
        const glm::vec4& clearColor = glm::vec4(0),
        float clearDepth = 1.0f) override;
    void EndRenderPass() override;
    void SetViewport(uint32_t x, uint32_t y, uint32_t width, uint32_t height) override;
    void BindPipeline(const std::shared_ptr<RenderResource>& pipeline) override;
    void BindVertexBuffer(
        const std::shared_ptr<RenderResource>& buffer,
        size_t offset = 0) override;
    void BindIndexBuffer(
        const std::shared_ptr<RenderResource>& buffer,
        size_t offset = 0) override;
    void BindTexture(
        uint32_t slot,
        const std::shared_ptr<RenderResource>& texture,
        const std::shared_ptr<RenderResource>& sampler = nullptr) override;
    void BindStorageBuffer(
        uint32_t slot,
        const std::shared_ptr<RenderResource>& buffer) override;
    void SetPushConstants(
        const void* data,
        size_t size,
        size_t offset = 0) override;

    // Drawing Commands
    void Draw(
        uint32_t vertexCount,
        uint32_t instanceCount = 1,
        uint32_t firstVertex = 0,
        uint32_t firstInstance = 0) override;

    void DrawIndexed(
        uint32_t indexCount,
        uint32_t instanceCount = 1,
        uint32_t firstIndex = 0,
        int32_t vertexOffset = 0,
        uint32_t firstInstance = 0) override;

    void DrawIndirect(
        const std::shared_ptr<RenderResource>& indirectBuffer,
        size_t offset,
        uint32_t drawCount) override;

    void Dispatch(
        uint32_t groupCountX,
        uint32_t groupCountY,
        uint32_t groupCountZ) override;

    void DispatchIndirect(
        const std::shared_ptr<RenderResource>& indirectBuffer,
        size_t offset) override;

    // Synchronization
    void MemoryBarrier(uint32_t barrierType) override;
    void FramebufferBarrier() override;

    // Multi-GPU (no-op for OpenGL)
    void SyncGPUs() override {}
    uint32_t GetFrameIndex() const override;
    uint64_t GetDeviceHandle(uint32_t deviceIndex) const override;

    // Performance Monitoring
    void BeginGPUQuery(const std::string& label) override;
    double EndGPUQuery(const std::string& label) override;
    float GetGPUUtilization(uint32_t deviceIndex = 0) const override;
    uint64_t GetGPUMemoryUsage(uint32_t deviceIndex = 0) const override;
    uint64_t GetGPUMemoryTotal(uint32_t deviceIndex = 0) const override;

private:
    uint32_t m_Width = 0;
    uint32_t m_Height = 0;
    uint32_t m_FrameIndex = 0;

    // OpenGL state tracking
    uint32_t m_CurrentFramebuffer = 0;
    uint32_t m_CurrentProgram = 0;
    uint32_t m_CurrentVAO = 0;

    // GPU query tracking
    std::map<std::string, uint32_t> m_QueryObjects;
    std::map<std::string, std::chrono::high_resolution_clock::time_point> m_QueryStartTimes;

    // Helper functions
    uint32_t GetGLFormat(const std::string& format) const;
    uint32_t GetGLUsageFlags(uint32_t usageFlags) const;
};

