#include "VulkanBackend.h"
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cstring>

VulkanBackend::VulkanBackend() = default;

VulkanBackend::~VulkanBackend() {
    Shutdown();
}

bool VulkanBackend::Init(uint32_t width, uint32_t height, void* windowHandle) {
    m_SwapchainExtent.width = width;
    m_SwapchainExtent.height = height;

    try {
        CreateInstance();
        SetupDebugMessenger();
        CreateSurface(windowHandle);
        SelectPhysicalDevices();
        CreateLogicalDevices();
        CreateSwapchain(width, height);

        SPDLOG_INFO("Vulkan backend initialized: {}x{}, {} GPU(s)",
                    width, height, m_Devices.size());
        return true;
    } catch (const std::exception& e) {
        SPDLOG_ERROR("Vulkan initialization failed: {}", e.what());
        return false;
    }
}

void VulkanBackend::Shutdown() {
    if (m_Devices.empty()) return;

    WaitForGPU();

    // Destroy swapchain
    if (m_Swapchain != VK_NULL_HANDLE) {
        for (auto imageView : m_SwapchainImageViews) {
            vkDestroyImageView(GetVkDevice(), imageView, nullptr);
        }
        vkDestroySwapchainKHR(GetVkDevice(), m_Swapchain, nullptr);
        m_Swapchain = VK_NULL_HANDLE;
    }

    // Destroy surface
    if (m_Surface != VK_NULL_HANDLE && m_Instance != VK_NULL_HANDLE) {
        vkDestroySurfaceKHR(m_Instance, m_Surface, nullptr);
        m_Surface = VK_NULL_HANDLE;
    }

    // Clear devices (VulkanDevice destructors handle cleanup)
    m_Devices.clear();
    m_PhysicalDevices.clear();

    // Destroy debug messenger
    if (m_DebugMessenger != VK_NULL_HANDLE && m_Instance != VK_NULL_HANDLE) {
        PFN_vkDestroyDebugUtilsMessengerEXT func =
            (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
                m_Instance, "vkDestroyDebugUtilsMessengerEXT");
        if (func != nullptr) {
            func(m_Instance, m_DebugMessenger, nullptr);
        }
    }

    // Destroy instance
    if (m_Instance != VK_NULL_HANDLE) {
        vkDestroyInstance(m_Instance, nullptr);
        m_Instance = VK_NULL_HANDLE;
    }

    SPDLOG_INFO("Vulkan backend shutdown complete");
}

std::string VulkanBackend::GetAPIName() const {
    return "Vulkan 1.3";
}

bool VulkanBackend::IsAvailable() {
    // Try to load Vulkan library
    #ifdef _WIN32
        HMODULE vulkanLib = LoadLibraryA("vulkan-1.dll");
        if (vulkanLib) {
            FreeLibrary(vulkanLib);
            return true;
        }
    #elif defined(__linux__)
        void* vulkanLib = dlopen("libvulkan.so.1", RTLD_LAZY);
        if (vulkanLib) {
            dlclose(vulkanLib);
            return true;
        }
    #endif
    return false;
}

uint32_t VulkanBackend::GetDeviceCount() const {
    return m_Devices.size();
}

GPUDeviceInfo VulkanBackend::GetDeviceInfo(uint32_t deviceIndex) const {
    if (deviceIndex >= m_PhysicalDevices.size()) {
        return GPUDeviceInfo{};
    }

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(m_PhysicalDevices[deviceIndex], &props);

    GPUDeviceInfo info;
    info.id = deviceIndex;
    info.name = props.deviceName;

    // Query device memory
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(m_PhysicalDevices[deviceIndex], &memProps);

    for (uint32_t i = 0; i < memProps.memoryHeapCount; i++) {
        if (memProps.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
            info.dedicatedMemory = memProps.memoryHeaps[i].size;
            break;
        }
    }

    // Estimate compute power (TFlops)
    // This is a rough estimate; real benchmarking would be more accurate
    if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
        info.estimatedPeakTFlops = 50.0f; // Conservative estimate for modern GPUs
    } else {
        info.estimatedPeakTFlops = 10.0f;
    }

    return info;
}

void VulkanBackend::SetActiveDevice(uint32_t deviceIndex) {
    if (deviceIndex < m_Devices.size()) {
        m_ActiveDevice = deviceIndex;
    }
}

bool VulkanBackend::SupportsLinkedGPUs() const {
    return m_LinkedGPUsSupported;
}

bool VulkanBackend::SupportsFeature(const std::string& featureName) const {
    // TODO: Check VkPhysicalDeviceFeatures for requested feature
    return false;
}

bool VulkanBackend::SupportsRayTracing() const {
    // TODO: Check for VK_KHR_ray_tracing_pipeline extension
    return false;
}

bool VulkanBackend::SupportsMeshShaders() const {
    // TODO: Check for VK_EXT_mesh_shader extension
    return false;
}

uint32_t VulkanBackend::GetMaxTextureSize() const {
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(m_PhysicalDevices[0], &props);
    return props.limits.maxImageDimension2D;
}

glm::uvec2 VulkanBackend::GetMaxFramebufferSize() const {
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(m_PhysicalDevices[0], &props);
    return glm::uvec2(
        props.limits.maxFramebufferWidth,
        props.limits.maxFramebufferHeight);
}

std::shared_ptr<RenderResource> VulkanBackend::CreateBuffer(
    size_t size,
    const void* data,
    uint32_t usageFlags)
{
    // TODO: Implement with VMA
    auto resource = std::make_shared<RenderResource>(RenderResource::Type::Buffer);
    return resource;
}

std::shared_ptr<RenderResource> VulkanBackend::CreateTexture(
    uint32_t width,
    uint32_t height,
    const std::string& format,
    const void* data,
    bool generateMips)
{
    // TODO: Implement
    auto resource = std::make_shared<RenderResource>(RenderResource::Type::Texture);
    return resource;
}

std::shared_ptr<RenderResource> VulkanBackend::CreateTexture3D(
    uint32_t width,
    uint32_t height,
    uint32_t depth,
    const std::string& format,
    const void* data)
{
    // TODO: Implement
    auto resource = std::make_shared<RenderResource>(RenderResource::Type::Texture);
    return resource;
}

std::shared_ptr<RenderResource> VulkanBackend::CreateCubemap(
    uint32_t size,
    const std::string& format,
    const std::vector<const void*>& faceImages)
{
    // TODO: Implement
    auto resource = std::make_shared<RenderResource>(RenderResource::Type::Texture);
    return resource;
}

std::shared_ptr<RenderResource> VulkanBackend::CreateFramebuffer(
    uint32_t width,
    uint32_t height,
    const std::vector<std::string>& colorFormats,
    const std::string& depthFormat)
{
    // TODO: Implement
    auto resource = std::make_shared<RenderResource>(RenderResource::Type::Framebuffer);
    return resource;
}

std::shared_ptr<RenderResource> VulkanBackend::CreateShader(
    const std::string& source,
    ShaderType type)
{
    // TODO: Compile GLSL to SPIR-V, create shader module
    auto resource = std::make_shared<RenderResource>(RenderResource::Type::Pipeline);
    return resource;
}

std::shared_ptr<RenderResource> VulkanBackend::CreatePipeline(
    const void* config)
{
    // TODO: Implement
    auto resource = std::make_shared<RenderResource>(RenderResource::Type::Pipeline);
    return resource;
}

void VulkanBackend::UpdateBuffer(
    const std::shared_ptr<RenderResource>& buffer,
    size_t offset,
    size_t size,
    const void* data)
{
    // TODO: Implement
}

void VulkanBackend::UpdateTexture(
    const std::shared_ptr<RenderResource>& texture,
    uint32_t x,
    uint32_t y,
    uint32_t width,
    uint32_t height,
    const void* data)
{
    // TODO: Implement
}

void VulkanBackend::CopyBuffer(
    const std::shared_ptr<RenderResource>& src,
    const std::shared_ptr<RenderResource>& dst,
    size_t size,
    size_t srcOffset,
    size_t dstOffset)
{
    // TODO: Implement
}

void VulkanBackend::CopyBufferToTexture(
    const std::shared_ptr<RenderResource>& buffer,
    const std::shared_ptr<RenderResource>& texture,
    uint32_t width,
    uint32_t height)
{
    // TODO: Implement
}

void VulkanBackend::DestroyResource(
    const std::shared_ptr<RenderResource>& resource)
{
    // TODO: Implement
}

void VulkanBackend::WaitForGPU() {
    if (m_Devices.empty()) return;
    vkDeviceWaitIdle(GetVkDevice());
}

void VulkanBackend::BeginCommandBuffer() {
    // TODO: Allocate and begin command buffer
}

void VulkanBackend::EndCommandBuffer() {
    // TODO: End command buffer
}

void VulkanBackend::BeginRenderPass(
    const std::shared_ptr<RenderResource>& framebuffer,
    const glm::vec4& clearColor,
    float clearDepth)
{
    // TODO: Implement
}

void VulkanBackend::EndRenderPass() {
    // TODO: Implement
}

void VulkanBackend::SetViewport(
    uint32_t x,
    uint32_t y,
    uint32_t width,
    uint32_t height)
{
    // TODO: Record viewport command
}

void VulkanBackend::BindPipeline(
    const std::shared_ptr<RenderResource>& pipeline)
{
    // TODO: Implement
}

void VulkanBackend::BindVertexBuffer(
    const std::shared_ptr<RenderResource>& buffer,
    size_t offset)
{
    // TODO: Implement
}

void VulkanBackend::BindIndexBuffer(
    const std::shared_ptr<RenderResource>& buffer,
    size_t offset)
{
    // TODO: Implement
}

void VulkanBackend::BindTexture(
    uint32_t slot,
    const std::shared_ptr<RenderResource>& texture,
    const std::shared_ptr<RenderResource>& sampler)
{
    // TODO: Implement
}

void VulkanBackend::BindStorageBuffer(
    uint32_t slot,
    const std::shared_ptr<RenderResource>& buffer)
{
    // TODO: Implement
}

void VulkanBackend::SetPushConstants(
    const void* data,
    size_t size,
    size_t offset)
{
    // TODO: Implement
}

void VulkanBackend::Draw(
    uint32_t vertexCount,
    uint32_t instanceCount,
    uint32_t firstVertex,
    uint32_t firstInstance)
{
    // TODO: Record draw command
}

void VulkanBackend::DrawIndexed(
    uint32_t indexCount,
    uint32_t instanceCount,
    uint32_t firstIndex,
    int32_t vertexOffset,
    uint32_t firstInstance)
{
    // TODO: Record draw command
}

void VulkanBackend::DrawIndirect(
    const std::shared_ptr<RenderResource>& indirectBuffer,
    size_t offset,
    uint32_t drawCount)
{
    // TODO: Implement
}

void VulkanBackend::Dispatch(
    uint32_t groupCountX,
    uint32_t groupCountY,
    uint32_t groupCountZ)
{
    // TODO: Implement
}

void VulkanBackend::DispatchIndirect(
    const std::shared_ptr<RenderResource>& indirectBuffer,
    size_t offset)
{
    // TODO: Implement
}

void VulkanBackend::MemoryBarrier(uint32_t barrierType) {
    // TODO: Implement
}

void VulkanBackend::FramebufferBarrier() {
    // TODO: Implement
}

void VulkanBackend::SyncGPUs() {
    if (m_Devices.size() > 1) {
        WaitForGPU();
        // TODO: Implement linked GPU synchronization
    }
}

uint64_t VulkanBackend::GetDeviceHandle(uint32_t deviceIndex) const {
    if (deviceIndex >= m_PhysicalDevices.size()) {
        return 0;
    }
    return reinterpret_cast<uint64_t>(m_PhysicalDevices[deviceIndex]);
}

void VulkanBackend::BeginGPUQuery(const std::string& label) {
    // TODO: Implement timestamp query
}

double VulkanBackend::EndGPUQuery(const std::string& label) {
    // TODO: Implement
    return 0.0;
}

float VulkanBackend::GetGPUUtilization(uint32_t deviceIndex) const {
    // TODO: Implement via VkQueryPool or vendor extensions
    return 0.0f;
}

uint64_t VulkanBackend::GetGPUMemoryUsage(uint32_t deviceIndex) const {
    // TODO: Query from VMA
    return 0;
}

uint64_t VulkanBackend::GetGPUMemoryTotal(uint32_t deviceIndex) const {
    if (deviceIndex < m_PhysicalDevices.size()) {
        auto info = GetDeviceInfo(deviceIndex);
        return info.dedicatedMemory;
    }
    return 0;
}

VkPhysicalDevice VulkanBackend::GetVkPhysicalDevice(uint32_t deviceIndex) const {
    if (deviceIndex < m_PhysicalDevices.size()) {
        return m_PhysicalDevices[deviceIndex];
    }
    return VK_NULL_HANDLE;
}

VkDevice VulkanBackend::GetVkDevice() const {
    if (m_ActiveDevice < m_Devices.size()) {
        return m_Devices[m_ActiveDevice]->GetLogicalDevice();
    }
    return VK_NULL_HANDLE;
}

VkQueue VulkanBackend::GetVkQueue(uint32_t deviceIndex) const {
    if (deviceIndex < m_Devices.size()) {
        return m_Devices[deviceIndex]->GetGraphicsQueue();
    }
    return VK_NULL_HANDLE;
}

std::vector<uint32_t> VulkanBackend::CompileGLSLToSPIRV(
    const std::string& glslSource,
    ShaderType shaderType)
{
    // TODO: Implement using glslang
    return std::vector<uint32_t>();
}

void VulkanBackend::CreateInstance() {
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "GameEngine";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "GameEngine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_3;

    std::vector<const char*> extensions = {
        VK_KHR_SURFACE_EXTENSION_NAME,
        #ifdef _WIN32
        VK_KHR_WIN32_SURFACE_EXTENSION_NAME,
        #elif defined(__linux__)
        VK_KHR_XCLIB_SURFACE_EXTENSION_NAME,
        #endif
    };

    // Add debug extensions if validation enabled
    std::vector<const char*> layers;
    if (true) { // TODO: Check EngineConfig
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        layers.push_back("VK_LAYER_KHRONOS_validation");
    }

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;
    createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();
    createInfo.enabledLayerCount = static_cast<uint32_t>(layers.size());
    createInfo.ppEnabledLayerNames = layers.data();

    if (vkCreateInstance(&createInfo, nullptr, &m_Instance) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create Vulkan instance");
    }

    SPDLOG_INFO("Vulkan instance created");
}

void VulkanBackend::SelectPhysicalDevices() {
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(m_Instance, &deviceCount, nullptr);

    if (deviceCount == 0) {
        throw std::runtime_error("No Vulkan-capable devices found");
    }

    m_PhysicalDevices.resize(deviceCount);
    vkEnumeratePhysicalDevices(m_Instance, &deviceCount, m_PhysicalDevices.data());

    SPDLOG_INFO("Found {} Vulkan device(s)", deviceCount);

    for (uint32_t i = 0; i < deviceCount; i++) {
        auto info = GetDeviceInfo(i);
        SPDLOG_INFO("  GPU {}: {}", i, info.name);
    }
}

void VulkanBackend::CreateLogicalDevices() {
    for (auto physicalDevice : m_PhysicalDevices) {
        // TODO: Create logical device wrapper
        // auto device = std::make_unique<VulkanDevice>(this, physicalDevice);
        // device->Create();
        // m_Devices.push_back(std::move(device));
    }

    if (m_Devices.empty()) {
        throw std::runtime_error("Failed to create logical devices");
    }
}

void VulkanBackend::CreateSurface(void* windowHandle) {
    // TODO: Platform-specific surface creation
    #ifdef _WIN32
    // HWND hwnd = static_cast<HWND>(windowHandle);
    // VkWin32SurfaceCreateInfoKHR createInfo{};
    // ...
    #endif

    SPDLOG_INFO("Vulkan surface created");
}

void VulkanBackend::CreateSwapchain(uint32_t width, uint32_t height) {
    // TODO: Implement swapchain creation
    SPDLOG_INFO("Vulkan swapchain created: {}x{}", width, height);
}

void VulkanBackend::SetupDebugMessenger() {
    // TODO: Setup VK_EXT_debug_utils callback
}

void VulkanBackend::TransitionImageLayout(
    VkImage image,
    VkFormat format,
    VkImageLayout oldLayout,
    VkImageLayout newLayout)
{
    // TODO: Implement
}

void VulkanBackend::CopyBufferToBuffer(
    VkBuffer src,
    VkBuffer dst,
    VkDeviceSize size,
    VkDeviceSize srcOffset,
    VkDeviceSize dstOffset)
{
    // TODO: Implement
}

void VulkanBackend::CopyBufferToImage(
    VkBuffer buffer,
    VkImage image,
    uint32_t width,
    uint32_t height)
{
    // TODO: Implement
}

VkFormat VulkanBackend::GetVkFormat(const std::string& formatStr) const {
    if (formatStr == "RGBA8") return VK_FORMAT_R8G8B8A8_UNORM;
    if (formatStr == "RGB8") return VK_FORMAT_R8G8B8_UNORM;
    if (formatStr == "RGBA32F") return VK_FORMAT_R32G32B32A32_SFLOAT;
    if (formatStr == "RGB32F") return VK_FORMAT_R32G32B32_SFLOAT;
    if (formatStr == "D32F") return VK_FORMAT_D32_SFLOAT;
    if (formatStr == "D24S8") return VK_FORMAT_D24_UNORM_S8_UINT;
    return VK_FORMAT_R8G8B8A8_UNORM;
}

VkBufferUsageFlags VulkanBackend::GetVkBufferUsageFlags(uint32_t usageFlags) const {
    VkBufferUsageFlags vkFlags = 0;
    if (usageFlags & 0x1) vkFlags |= VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    if (usageFlags & 0x2) vkFlags |= VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
    if (usageFlags & 0x4) vkFlags |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    if (usageFlags & 0x8) vkFlags |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    if (usageFlags & 0x10) vkFlags |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    if (usageFlags & 0x20) vkFlags |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    return vkFlags;
}

