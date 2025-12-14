#include "RenderBackend.h"
#include "OpenGLBackend.h"

// Forward declaration - will be implemented when Vulkan support is enabled
class VulkanBackend;

/**
 * @brief Factory function to create render backend
 */
std::unique_ptr<RenderBackend> CreateRenderBackend(
    RenderBackend::API preferApi)
{
    // Check if Vulkan is available and preferred
#ifdef ENABLE_VULKAN
    if (preferApi == RenderBackend::API::Vulkan) {
        // Attempt to create Vulkan backend
        // This will be fully implemented when VulkanBackend is complete
        // For now, fall back to OpenGL
        SPDLOG_WARN("Vulkan backend requested but not yet fully implemented, using OpenGL");
    }
#endif

    // Default to OpenGL
    return std::make_unique<OpenGLBackend>();
}

/**
 * @brief Check if Vulkan is available on this system
 */
bool IsVulkanAvailable()
{
#ifdef ENABLE_VULKAN
    // This will check if Vulkan SDK is installed and usable
    // Placeholder implementation
    return false; // TODO: Implement Vulkan availability check
#else
    return false;
#endif
}

