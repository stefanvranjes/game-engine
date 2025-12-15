#pragma once

#include <vulkan/vulkan.h>
#include <string>
#include <vector>
#include <memory>

/**
 * @brief Vulkan debugging and profiling utilities
 * 
 * Provides:
 * - Debug messenger for validation errors
 * - GPU timestamp profiling
 * - Named objects for RenderDoc debugging
 * - Memory allocator statistics
 */
class VulkanDebugUtils {
public:
    /**
     * @brief Initialize debug utilities for an instance
     * @param instance Vulkan instance
     * @param enableValidationLayers Enable Khronos validation layer
     * @param enableGPUAssisted Enable GPU-assisted validation
     */
    static VkDebugUtilsMessengerEXT SetupDebugMessenger(
        VkInstance instance,
        bool enableValidationLayers = true,
        bool enableGPUAssisted = false);

    /**
     * @brief Destroy debug messenger
     */
    static void DestroyDebugMessenger(
        VkInstance instance,
        VkDebugUtilsMessengerEXT messenger);

    /**
     * @brief Set human-readable name for Vulkan object (visible in RenderDoc)
     * @param device Logical device
     * @param objectHandle Handle to object (e.g., VkBuffer, VkImage)
     * @param objectType Type of object
     * @param name Name to display in debugger
     */
    static void SetObjectName(
        VkDevice device,
        uint64_t objectHandle,
        VkObjectType objectType,
        const std::string& name);

    /**
     * @brief Begin named GPU command region (for RenderDoc)
     * @param commandBuffer Command buffer to record into
     * @param name Region name
     * @param color RGBA color for RenderDoc visualization
     */
    static void BeginCommandRegion(
        VkCommandBuffer commandBuffer,
        const std::string& name,
        const float* color = nullptr);

    /**
     * @brief End named GPU command region
     */
    static void EndCommandRegion(VkCommandBuffer commandBuffer);

    /**
     * @brief Insert debug marker into command stream
     */
    static void InsertDebugMarker(
        VkCommandBuffer commandBuffer,
        const std::string& message,
        const float* color = nullptr);

    /**
     * @brief GPU timestamp query for profiling
     */
    struct TimestampQuery {
        std::string name;
        VkQueryPool queryPool = VK_NULL_HANDLE;
        uint32_t startIndex = 0;
        uint32_t endIndex = 1;
        double elapsedMs = 0.0;
    };

    /**
     * @brief Create timestamp query pool
     * @param device Logical device
     * @param queryCount Number of queries to allocate
     * @return Query pool handle
     */
    static VkQueryPool CreateQueryPool(
        VkDevice device,
        uint32_t queryCount,
        VkQueryType queryType = VK_QUERY_TYPE_TIMESTAMP);

    /**
     * @brief Begin timestamp query
     * @param commandBuffer Command buffer
     * @param queryPool Query pool
     * @param queryIndex Index into pool
     */
    static void BeginTimestampQuery(
        VkCommandBuffer commandBuffer,
        VkQueryPool queryPool,
        uint32_t queryIndex);

    /**
     * @brief End timestamp query
     */
    static void EndTimestampQuery(
        VkCommandBuffer commandBuffer,
        VkQueryPool queryPool,
        uint32_t queryIndex);

    /**
     * @brief Get timestamp query result in milliseconds
     * @param device Logical device
     * @param queryPool Query pool
     * @param firstQuery First query index
     * @param queryCount Number of consecutive queries
     * @param timestampPeriod GPU timestamp period from device properties
     * @return Elapsed time in milliseconds
     */
    static double GetTimestampResult(
        VkDevice device,
        VkQueryPool queryPool,
        uint32_t firstQuery,
        uint32_t queryCount,
        float timestampPeriod);

    /**
     * @brief Enable/disable debug output
     */
    static void SetDebugOutputEnabled(bool enabled) { s_DebugOutputEnabled = enabled; }

    /**
     * @brief Get human-readable Vulkan error message
     */
    static std::string GetVkResultString(VkResult result);

    /**
     * @brief Print GPU memory statistics
     * @param device Logical device
     * @param physicalDevice Physical device
     */
    static void PrintMemoryStatistics(
        VkDevice device,
        VkPhysicalDevice physicalDevice);

private:
    static bool s_DebugOutputEnabled;

    // Debug callback function
    static VKAPI_ATTR VkBool32 VKAPI_CALL DebugCallback(
        VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
        VkDebugUtilsMessageTypeFlagsEXT messageType,
        const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
        void* pUserData);
};

/**
 * @brief RAII wrapper for command region naming
 */
class VulkanCommandRegion {
public:
    VulkanCommandRegion(
        VkCommandBuffer commandBuffer,
        const std::string& name,
        const float* color = nullptr)
        : m_CommandBuffer(commandBuffer)
    {
        VulkanDebugUtils::BeginCommandRegion(commandBuffer, name, color);
    }

    ~VulkanCommandRegion() {
        VulkanDebugUtils::EndCommandRegion(m_CommandBuffer);
    }

private:
    VkCommandBuffer m_CommandBuffer;
};

/**
 * @brief RAII wrapper for GPU profiling
 */
class VulkanGPUProfile {
public:
    VulkanGPUProfile(
        VkDevice device,
        VkCommandBuffer commandBuffer,
        VkQueryPool queryPool,
        uint32_t queryIndex,
        const std::string& label,
        float timestampPeriod)
        : m_Device(device),
          m_CommandBuffer(commandBuffer),
          m_QueryPool(queryPool),
          m_QueryIndex(queryIndex),
          m_Label(label),
          m_TimestampPeriod(timestampPeriod)
    {
        VulkanDebugUtils::BeginTimestampQuery(commandBuffer, queryPool, queryIndex * 2);
    }

    ~VulkanGPUProfile() {
        VulkanDebugUtils::EndTimestampQuery(m_CommandBuffer, m_QueryPool, m_QueryIndex * 2 + 1);
    }

private:
    VkDevice m_Device;
    VkCommandBuffer m_CommandBuffer;
    VkQueryPool m_QueryPool;
    uint32_t m_QueryIndex;
    std::string m_Label;
    float m_TimestampPeriod;
};

