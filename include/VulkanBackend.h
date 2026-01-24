#pragma once

#include "RenderBackend.h"
#include <map>
#include <vulkan/vulkan.h>

// Forward declarations
class VulkanDevice;
class VulkanMemoryAllocator;

/**
 * @brief Vulkan 1.3+ backend implementation with multi-GPU support
 * 
 * Full implementation of deferred rendering pipeline using Vulkan.
 * Supports linked GPU rendering, split-frame, and alternate-frame techniques
 * for multi-GPU scaling.
 * 
 * Features:
 * - Command pools per thread for lock-free recording
 * - VMA for automatic memory management and defragmentation
 * - Input attachments for efficient deferred lighting
 * - Subpasses for automatic layout transitions
 * - Dynamic rendering (VK_KHR_dynamic_rendering) for flexibility
 * - Multi-GPU synchronization with device groups
 */
class VulkanBackend : public RenderBackend {
public:
    VulkanBackend();
    ~VulkanBackend() override;

    // Initialization
    bool Init(uint32_t width, uint32_t height, void* windowHandle) override;
    void Shutdown() override;

    API GetAPI() const override { return API::Vulkan; }
    std::string GetAPIName() const override;

    // Static utility
    static bool IsAvailable();

    // GPU Device Management
    uint32_t GetDeviceCount() const override;
    GPUDeviceInfo GetDeviceInfo(uint32_t deviceIndex) const override;
    void SetActiveDevice(uint32_t deviceIndex) override;
    uint32_t GetActiveDevice() const override { return m_ActiveDevice; }
    bool SupportsLinkedGPUs() const override;

    // Capability Queries
    bool SupportsFeature(const std::string& featureName) const override;
    bool SupportsRayTracing() const override;
    bool SupportsMeshShaders() const override;
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
    void GPUMemoryBarrier(uint32_t barrierType) override;
    void FramebufferBarrier() override;

    // Multi-GPU Support
    void SyncGPUs() override;
    uint32_t GetFrameIndex() const override { return m_FrameIndex; }
    uint64_t GetDeviceHandle(uint32_t deviceIndex) const override;

    // Performance Monitoring
    void BeginGPUQuery(const std::string& label) override;
    double EndGPUQuery(const std::string& label) override;
    float GetGPUUtilization(uint32_t deviceIndex = 0) const override;
    uint64_t GetGPUMemoryUsage(uint32_t deviceIndex = 0) const override;
    uint64_t GetGPUMemoryTotal(uint32_t deviceIndex = 0) const override;

    // Vulkan-specific helpers
    VkInstance GetVkInstance() const { return m_Instance; }
    VkPhysicalDevice GetVkPhysicalDevice(uint32_t deviceIndex = 0) const;
    VkDevice GetVkDevice() const;
    VkQueue GetVkQueue(uint32_t deviceIndex = 0) const;
    VkCommandBuffer GetVkCommandBuffer() const { return m_CurrentCommandBuffer; }

    // Shader compilation from GLSL to SPIR-V
    std::vector<uint32_t> CompileGLSLToSPIRV(
        const std::string& glslSource,
        ShaderType shaderType);

private:
    // Vulkan instance and devices
    VkInstance m_Instance = VK_NULL_HANDLE;
    VkSurfaceKHR m_Surface = VK_NULL_HANDLE;
    VkSwapchainKHR m_Swapchain = VK_NULL_HANDLE;

    std::vector<VkPhysicalDevice> m_PhysicalDevices;
    std::vector<std::unique_ptr<VulkanDevice>> m_Devices;
    uint32_t m_ActiveDevice = 0;

    // Swap chain images
    std::vector<VkImage> m_SwapchainImages;
    std::vector<VkImageView> m_SwapchainImageViews;
    VkFormat m_SwapchainFormat = VK_FORMAT_UNDEFINED;
    VkExtent2D m_SwapchainExtent = {0, 0};

    // Command recording
    VkCommandBuffer m_CurrentCommandBuffer = VK_NULL_HANDLE;
    VkRenderPass m_CurrentRenderPass = VK_NULL_HANDLE;

    // Multi-GPU state
    uint32_t m_FrameIndex = 0;
    bool m_LinkedGPUsSupported = false;
    VkDeviceGroupBindSparseDeviceIndexInfoKHR m_GroupBindInfo = {};

    // Timestamp queries
    struct TimestampQuery {
        VkQueryPool queryPool = VK_NULL_HANDLE;
        uint32_t startIndex = 0;
        uint32_t endIndex = 1;
    };
    std::map<std::string, TimestampQuery> m_TimestampQueries;

    // Memory allocator (VMA)
    std::unique_ptr<VulkanMemoryAllocator> m_MemoryAllocator;

    // Debug utilities
    VkDebugUtilsMessengerEXT m_DebugMessenger = VK_NULL_HANDLE;

    // Initialization helpers
    void CreateInstance();
    void SelectPhysicalDevices();
    void CreateLogicalDevices();
    void CreateSurface(void* windowHandle);
    void CreateSwapchain(uint32_t width, uint32_t height);
    void SetupDebugMessenger();

    // Rendering helpers
    void TransitionImageLayout(
        VkImage image,
        VkFormat format,
        VkImageLayout oldLayout,
        VkImageLayout newLayout);

    void CopyBufferToBuffer(
        VkBuffer src,
        VkBuffer dst,
        VkDeviceSize size,
        VkDeviceSize srcOffset,
        VkDeviceSize dstOffset);

    void CopyBufferToImage(
        VkBuffer buffer,
        VkImage image,
        uint32_t width,
        uint32_t height);

    // Format conversions
    VkFormat GetVkFormat(const std::string& formatStr) const;
    VkBufferUsageFlags GetVkBufferUsageFlags(uint32_t usageFlags) const;
};

