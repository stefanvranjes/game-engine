#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include "glm/glm.hpp"

// Forward declarations
class Texture;
class Mesh;
class Shader;

/**
 * @brief Abstract render resource (buffer, texture, framebuffer)
 */
class RenderResource {
public:
    enum class Type {
        Buffer,
        Texture,
        Framebuffer,
        Sampler,
        DescriptorSet,
        Pipeline,
        RenderPass
    };

    Type type;
    std::string name;
    uint64_t nativeHandle = 0; // Cast to VkBuffer, GLuint, etc.

    RenderResource(Type t, const std::string& n = "") : type(t), name(n) {}
    virtual ~RenderResource() = default;
};

/**
 * @brief Information about a GPU device
 */
struct GPUDeviceInfo {
    uint32_t id = 0;
    std::string name;
    std::string driverVersion;
    uint64_t dedicatedMemory = 0;
    uint64_t sharedMemory = 0;
    bool supportsRayTracing = false;
    bool supportsMeshShaders = false;
    uint32_t computeQueueCount = 1;
    float estimatedPeakTFlops = 0.0f; // For scheduling decisions
};

/**
 * @brief Render backend abstraction interface
 * 
 * This interface defines all graphics operations that can be performed
 * in a graphics-API-agnostic way. Implementations exist for OpenGL and Vulkan.
 */
class RenderBackend {
public:
    enum class API {
        OpenGL,
        Vulkan
    };

    enum class ShaderType {
        Vertex,
        Fragment,
        Compute,
        Geometry,
        TessControl,
        TessEval,
        MeshTask,
        MeshShader
    };

    virtual ~RenderBackend() = default;

    // ============= Initialization =============
    /**
     * @brief Initialize the render backend for a given window
     * @param width Window width
     * @param height Window height
     * @param windowHandle Platform-specific window handle
     * @return true if initialization successful
     */
    virtual bool Init(uint32_t width, uint32_t height, void* windowHandle) = 0;

    /**
     * @brief Shutdown the backend and release all resources
     */
    virtual void Shutdown() = 0;

    /**
     * @brief Get the graphics API type
     */
    virtual API GetAPI() const = 0;

    /**
     * @brief Get human-readable API name (e.g., "Vulkan 1.3", "OpenGL 4.6")
     */
    virtual std::string GetAPIName() const = 0;

    // ============= GPU Device Management =============
    /**
     * @brief Get number of available GPU devices
     */
    virtual uint32_t GetDeviceCount() const = 0;

    /**
     * @brief Get information about a specific GPU
     */
    virtual GPUDeviceInfo GetDeviceInfo(uint32_t deviceIndex) const = 0;

    /**
     * @brief Select which GPU to use (Vulkan only, multi-GPU)
     * @param deviceIndex Index of GPU to use for subsequent operations
     */
    virtual void SetActiveDevice(uint32_t deviceIndex) = 0;

    /**
     * @brief Get currently active GPU index
     */
    virtual uint32_t GetActiveDevice() const = 0;

    /**
     * @brief Query if linked GPUs are available (VK_AMD_device_group equivalent)
     */
    virtual bool SupportsLinkedGPUs() const = 0;

    // ============= Capability Queries =============
    /**
     * @brief Check if a specific feature is supported
     */
    virtual bool SupportsFeature(const std::string& featureName) const = 0;

    /**
     * @brief Check if ray tracing is supported
     */
    virtual bool SupportsRayTracing() const = 0;

    /**
     * @brief Check if mesh shaders are supported
     */
    virtual bool SupportsMeshShaders() const = 0;

    /**
     * @brief Get maximum number of active lights
     */
    virtual uint32_t GetMaxLights() const = 0;

    /**
     * @brief Get maximum texture size
     */
    virtual uint32_t GetMaxTextureSize() const = 0;

    /**
     * @brief Get maximum frame buffer width/height
     */
    virtual glm::uvec2 GetMaxFramebufferSize() const = 0;

    // ============= Resource Creation =============
    /**
     * @brief Create a buffer (uniform, vertex, index, storage)
     * @param size Buffer size in bytes
     * @param data Initial data pointer (nullptr for uninitialized)
     * @param usageFlags API-specific usage flags
     * @return Resource handle
     */
    virtual std::shared_ptr<RenderResource> CreateBuffer(
        size_t size,
        const void* data = nullptr,
        uint32_t usageFlags = 0) = 0;

    /**
     * @brief Create a texture from image data
     * @param width Texture width
     * @param height Texture height
     * @param format Pixel format (e.g., "RGBA8", "RGB32F", "D32F")
     * @param data Pixel data pointer
     * @param generateMips Generate mipmaps
     * @return Resource handle
     */
    virtual std::shared_ptr<RenderResource> CreateTexture(
        uint32_t width,
        uint32_t height,
        const std::string& format,
        const void* data = nullptr,
        bool generateMips = true) = 0;

    /**
     * @brief Create a 3D texture
     */
    virtual std::shared_ptr<RenderResource> CreateTexture3D(
        uint32_t width,
        uint32_t height,
        uint32_t depth,
        const std::string& format,
        const void* data = nullptr) = 0;

    /**
     * @brief Create a cubemap texture
     * @param faceImages Array of 6 images (right, left, top, bottom, front, back)
     * @return Resource handle
     */
    virtual std::shared_ptr<RenderResource> CreateCubemap(
        uint32_t size,
        const std::string& format,
        const std::vector<const void*>& faceImages = {}) = 0;

    /**
     * @brief Create a framebuffer with attachments
     * @param width Framebuffer width
     * @param height Framebuffer height
     * @param colorFormats List of color attachment formats
     * @param depthFormat Depth attachment format (empty string = no depth)
     * @return Resource handle
     */
    virtual std::shared_ptr<RenderResource> CreateFramebuffer(
        uint32_t width,
        uint32_t height,
        const std::vector<std::string>& colorFormats,
        const std::string& depthFormat = "") = 0;

    /**
     * @brief Create a shader module from source code
     * @param source GLSL shader source code
     * @param type Shader type (vertex, fragment, etc.)
     * @return Resource handle
     */
    virtual std::shared_ptr<RenderResource> CreateShader(
        const std::string& source,
        ShaderType type) = 0;

    /**
     * @brief Create a pipeline state object
     * @param config Pipeline configuration (API-specific)
     * @return Resource handle
     */
    virtual std::shared_ptr<RenderResource> CreatePipeline(
        const void* config) = 0;

    // ============= Resource Updates =============
    /**
     * @brief Update buffer contents
     * @param buffer Resource from CreateBuffer
     * @param offset Offset into buffer
     * @param size Size of data to update
     * @param data Data pointer
     */
    virtual void UpdateBuffer(
        const std::shared_ptr<RenderResource>& buffer,
        size_t offset,
        size_t size,
        const void* data) = 0;

    /**
     * @brief Update texture contents
     */
    virtual void UpdateTexture(
        const std::shared_ptr<RenderResource>& texture,
        uint32_t x,
        uint32_t y,
        uint32_t width,
        uint32_t height,
        const void* data) = 0;

    /**
     * @brief Copy buffer to buffer
     */
    virtual void CopyBuffer(
        const std::shared_ptr<RenderResource>& src,
        const std::shared_ptr<RenderResource>& dst,
        size_t size,
        size_t srcOffset = 0,
        size_t dstOffset = 0) = 0;

    /**
     * @brief Copy buffer to texture
     */
    virtual void CopyBufferToTexture(
        const std::shared_ptr<RenderResource>& buffer,
        const std::shared_ptr<RenderResource>& texture,
        uint32_t width,
        uint32_t height) = 0;

    // ============= Resource Cleanup =============
    /**
     * @brief Destroy a resource (optional - shared_ptr handles it)
     */
    virtual void DestroyResource(const std::shared_ptr<RenderResource>& resource) = 0;

    /**
     * @brief Wait for all pending GPU operations to complete
     */
    virtual void WaitForGPU() = 0;

    // ============= Rendering Commands =============
    /**
     * @brief Begin recording a command buffer
     */
    virtual void BeginCommandBuffer() = 0;

    /**
     * @brief End recording a command buffer and prepare for submission
     */
    virtual void EndCommandBuffer() = 0;

    /**
     * @brief Begin a render pass
     * @param framebuffer Target framebuffer
     * @param clearColor Clear color (if applicable)
     * @param clearDepth Clear depth value
     */
    virtual void BeginRenderPass(
        const std::shared_ptr<RenderResource>& framebuffer,
        const glm::vec4& clearColor = glm::vec4(0),
        float clearDepth = 1.0f) = 0;

    /**
     * @brief End the current render pass
     */
    virtual void EndRenderPass() = 0;

    /**
     * @brief Set viewport and scissor rect
     */
    virtual void SetViewport(
        uint32_t x,
        uint32_t y,
        uint32_t width,
        uint32_t height) = 0;

    /**
     * @brief Bind a pipeline
     */
    virtual void BindPipeline(const std::shared_ptr<RenderResource>& pipeline) = 0;

    /**
     * @brief Bind vertex buffer
     */
    virtual void BindVertexBuffer(
        const std::shared_ptr<RenderResource>& buffer,
        size_t offset = 0) = 0;

    /**
     * @brief Bind index buffer
     */
    virtual void BindIndexBuffer(
        const std::shared_ptr<RenderResource>& buffer,
        size_t offset = 0) = 0;

    /**
     * @brief Bind texture for sampling
     * @param slot Texture slot/unit
     * @param texture Texture resource
     * @param sampler Sampler resource (or nullptr for default)
     */
    virtual void BindTexture(
        uint32_t slot,
        const std::shared_ptr<RenderResource>& texture,
        const std::shared_ptr<RenderResource>& sampler = nullptr) = 0;

    /**
     * @brief Bind storage buffer for compute
     */
    virtual void BindStorageBuffer(
        uint32_t slot,
        const std::shared_ptr<RenderResource>& buffer) = 0;

    /**
     * @brief Set push constants (small amounts of data)
     * @param data Pointer to push constant data
     * @param size Size of data
     * @param offset Offset into push constant space
     */
    virtual void SetPushConstants(
        const void* data,
        size_t size,
        size_t offset = 0) = 0;

    /**
     * @brief Draw primitives (vertices)
     * @param vertexCount Number of vertices to draw
     * @param instanceCount Number of instances
     * @param firstVertex First vertex index
     * @param firstInstance First instance index
     */
    virtual void Draw(
        uint32_t vertexCount,
        uint32_t instanceCount = 1,
        uint32_t firstVertex = 0,
        uint32_t firstInstance = 0) = 0;

    /**
     * @brief Draw indexed primitives
     */
    virtual void DrawIndexed(
        uint32_t indexCount,
        uint32_t instanceCount = 1,
        uint32_t firstIndex = 0,
        int32_t vertexOffset = 0,
        uint32_t firstInstance = 0) = 0;

    /**
     * @brief Draw indirect (parameters from GPU buffer)
     */
    virtual void DrawIndirect(
        const std::shared_ptr<RenderResource>& indirectBuffer,
        size_t offset,
        uint32_t drawCount) = 0;

    /**
     * @brief Dispatch compute shader
     */
    virtual void Dispatch(
        uint32_t groupCountX,
        uint32_t groupCountY,
        uint32_t groupCountZ) = 0;

    /**
     * @brief Dispatch compute shader indirectly
     */
    virtual void DispatchIndirect(
        const std::shared_ptr<RenderResource>& indirectBuffer,
        size_t offset) = 0;

    // ============= Synchronization =============
    /**
     * @brief Memory barrier for GPU synchronization
     * @param barrierType Type of barrier (shader-to-framebuffer, etc.)
     */
    virtual void MemoryBarrier(uint32_t barrierType) = 0;

    /**
     * @brief Framebuffer barrier for input attachments
     */
    virtual void FramebufferBarrier() = 0;

    // ============= Multi-GPU Support =============
    /**
     * @brief Synchronize between GPUs before frame submission
     * Used for linked GPU rendering or split-frame techniques
     */
    virtual void SyncGPUs() = 0;

    /**
     * @brief Get current frame index (for alternating GPU rendering)
     */
    virtual uint32_t GetFrameIndex() const = 0;

    /**
     * @brief Request GPU resource on specific device (multi-GPU)
     * @return Device-specific handle
     */
    virtual uint64_t GetDeviceHandle(uint32_t deviceIndex) const = 0;

    // ============= Performance Monitoring =============
    /**
     * @brief Begin a GPU timestamp query
     */
    virtual void BeginGPUQuery(const std::string& label) = 0;

    /**
     * @brief End a GPU timestamp query and retrieve results
     * @return Elapsed GPU time in milliseconds (0 if not ready)
     */
    virtual double EndGPUQuery(const std::string& label) = 0;

    /**
     * @brief Get GPU utilization percentage for a device
     */
    virtual float GetGPUUtilization(uint32_t deviceIndex = 0) const = 0;

    /**
     * @brief Get total GPU memory usage in bytes
     */
    virtual uint64_t GetGPUMemoryUsage(uint32_t deviceIndex = 0) const = 0;

    /**
     * @brief Get peak GPU memory available
     */
    virtual uint64_t GetGPUMemoryTotal(uint32_t deviceIndex = 0) const = 0;
};

// ============= Inline Helpers =============

/**
 * @brief Helper to create a render backend for the current platform
 * @param preferApi Preferred API (OpenGL or Vulkan if available)
 * @return Unique pointer to backend instance
 */
std::unique_ptr<RenderBackend> CreateRenderBackend(
    RenderBackend::API preferApi = RenderBackend::API::OpenGL);

/**
 * @brief Check if Vulkan backend is available on this system
 */
bool IsVulkanAvailable();

