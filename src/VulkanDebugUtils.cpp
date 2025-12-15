#include "VulkanDebugUtils.h"
#include <spdlog/spdlog.h>

bool VulkanDebugUtils::s_DebugOutputEnabled = true;

VkDebugUtilsMessengerEXT VulkanDebugUtils::SetupDebugMessenger(
    VkInstance instance,
    bool enableValidationLayers,
    bool enableGPUAssisted)
{
    if (!enableValidationLayers) {
        return VK_NULL_HANDLE;
    }

    VkDebugUtilsMessengerCreateInfoEXT createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    createInfo.messageSeverity =
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    createInfo.messageType =
        VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    createInfo.pfnUserCallback = DebugCallback;

    PFN_vkCreateDebugUtilsMessengerEXT func =
        (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
            instance, "vkCreateDebugUtilsMessengerEXT");

    if (func == nullptr) {
        SPDLOG_WARN("vkCreateDebugUtilsMessengerEXT not available");
        return VK_NULL_HANDLE;
    }

    VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;
    if (func(instance, &createInfo, nullptr, &messenger) != VK_SUCCESS) {
        SPDLOG_ERROR("Failed to create debug messenger");
        return VK_NULL_HANDLE;
    }

    SPDLOG_INFO("Vulkan debug messenger initialized");
    return messenger;
}

void VulkanDebugUtils::DestroyDebugMessenger(
    VkInstance instance,
    VkDebugUtilsMessengerEXT messenger)
{
    if (messenger == VK_NULL_HANDLE) return;

    PFN_vkDestroyDebugUtilsMessengerEXT func =
        (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
            instance, "vkDestroyDebugUtilsMessengerEXT");

    if (func != nullptr) {
        func(instance, messenger, nullptr);
    }
}

void VulkanDebugUtils::SetObjectName(
    VkDevice device,
    uint64_t objectHandle,
    VkObjectType objectType,
    const std::string& name)
{
    if (!s_DebugOutputEnabled) return;

    PFN_vkSetDebugUtilsObjectNameEXT func =
        (PFN_vkSetDebugUtilsObjectNameEXT)vkGetDeviceProcAddr(
            device, "vkSetDebugUtilsObjectNameEXT");

    if (func == nullptr) {
        return;
    }

    VkDebugUtilsObjectNameInfoEXT nameInfo{};
    nameInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
    nameInfo.objectType = objectType;
    nameInfo.objectHandle = objectHandle;
    nameInfo.pObjectName = name.c_str();

    func(device, &nameInfo);
}

void VulkanDebugUtils::BeginCommandRegion(
    VkCommandBuffer commandBuffer,
    const std::string& name,
    const float* color)
{
    if (!s_DebugOutputEnabled) return;

    VkDevice device = VK_NULL_HANDLE; // TODO: Get device from command buffer

    PFN_vkCmdBeginDebugUtilsLabelEXT func =
        (PFN_vkCmdBeginDebugUtilsLabelEXT)vkGetInstanceProcAddr(
            vkGetInstanceProcAddr(nullptr, "vkGetInstanceProcAddr"),
            "vkCmdBeginDebugUtilsLabelEXT");

    if (func == nullptr) {
        return;
    }

    VkDebugUtilsLabelEXT label{};
    label.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT;
    label.pLabelName = name.c_str();
    if (color) {
        for (int i = 0; i < 4; i++) {
            label.color[i] = color[i];
        }
    } else {
        label.color[0] = label.color[1] = label.color[2] = label.color[3] = 1.0f;
    }

    func(commandBuffer, &label);
}

void VulkanDebugUtils::EndCommandRegion(VkCommandBuffer commandBuffer) {
    if (!s_DebugOutputEnabled) return;

    PFN_vkCmdEndDebugUtilsLabelEXT func =
        (PFN_vkCmdEndDebugUtilsLabelEXT)vkGetInstanceProcAddr(
            vkGetInstanceProcAddr(nullptr, "vkGetInstanceProcAddr"),
            "vkCmdEndDebugUtilsLabelEXT");

    if (func == nullptr) {
        return;
    }

    func(commandBuffer);
}

void VulkanDebugUtils::InsertDebugMarker(
    VkCommandBuffer commandBuffer,
    const std::string& message,
    const float* color)
{
    if (!s_DebugOutputEnabled) return;

    PFN_vkCmdInsertDebugUtilsLabelEXT func =
        (PFN_vkCmdInsertDebugUtilsLabelEXT)vkGetInstanceProcAddr(
            vkGetInstanceProcAddr(nullptr, "vkGetInstanceProcAddr"),
            "vkCmdInsertDebugUtilsLabelEXT");

    if (func == nullptr) {
        return;
    }

    VkDebugUtilsLabelEXT label{};
    label.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT;
    label.pLabelName = message.c_str();
    if (color) {
        for (int i = 0; i < 4; i++) {
            label.color[i] = color[i];
        }
    } else {
        label.color[0] = label.color[1] = label.color[2] = label.color[3] = 1.0f;
    }

    func(commandBuffer, &label);
}

VkQueryPool VulkanDebugUtils::CreateQueryPool(
    VkDevice device,
    uint32_t queryCount,
    VkQueryType queryType)
{
    VkQueryPoolCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    createInfo.queryType = queryType;
    createInfo.queryCount = queryCount;

    VkQueryPool queryPool = VK_NULL_HANDLE;
    if (vkCreateQueryPool(device, &createInfo, nullptr, &queryPool) != VK_SUCCESS) {
        SPDLOG_ERROR("Failed to create query pool");
        return VK_NULL_HANDLE;
    }

    return queryPool;
}

void VulkanDebugUtils::BeginTimestampQuery(
    VkCommandBuffer commandBuffer,
    VkQueryPool queryPool,
    uint32_t queryIndex)
{
    vkCmdWriteTimestamp(
        commandBuffer,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        queryPool,
        queryIndex);
}

void VulkanDebugUtils::EndTimestampQuery(
    VkCommandBuffer commandBuffer,
    VkQueryPool queryPool,
    uint32_t queryIndex)
{
    vkCmdWriteTimestamp(
        commandBuffer,
        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        queryPool,
        queryIndex);
}

double VulkanDebugUtils::GetTimestampResult(
    VkDevice device,
    VkQueryPool queryPool,
    uint32_t firstQuery,
    uint32_t queryCount,
    float timestampPeriod)
{
    std::vector<uint64_t> results(queryCount);
    vkGetQueryPoolResults(
        device,
        queryPool,
        firstQuery,
        queryCount,
        results.size() * sizeof(uint64_t),
        results.data(),
        sizeof(uint64_t),
        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);

    if (results.size() < 2) {
        return 0.0;
    }

    uint64_t elapsedTicks = results[1] - results[0];
    return (elapsedTicks * timestampPeriod) / 1e6; // Convert to milliseconds
}

std::string VulkanDebugUtils::GetVkResultString(VkResult result) {
    switch (result) {
        case VK_SUCCESS: return "Success";
        case VK_NOT_READY: return "Not Ready";
        case VK_TIMEOUT: return "Timeout";
        case VK_EVENT_SET: return "Event Set";
        case VK_EVENT_RESET: return "Event Reset";
        case VK_INCOMPLETE: return "Incomplete";
        case VK_ERROR_OUT_OF_HOST_MEMORY: return "Out of Host Memory";
        case VK_ERROR_OUT_OF_DEVICE_MEMORY: return "Out of Device Memory";
        case VK_ERROR_INITIALIZATION_FAILED: return "Initialization Failed";
        case VK_ERROR_DEVICE_LOST: return "Device Lost";
        case VK_ERROR_MEMORY_MAP_FAILED: return "Memory Map Failed";
        case VK_ERROR_LAYER_NOT_PRESENT: return "Layer Not Present";
        case VK_ERROR_EXTENSION_NOT_PRESENT: return "Extension Not Present";
        case VK_ERROR_FEATURE_NOT_PRESENT: return "Feature Not Present";
        case VK_ERROR_INCOMPATIBLE_DRIVER: return "Incompatible Driver";
        case VK_ERROR_TOO_MANY_OBJECTS: return "Too Many Objects";
        case VK_ERROR_FORMAT_NOT_SUPPORTED: return "Format Not Supported";
        case VK_ERROR_FRAGMENTED_POOL: return "Fragmented Pool";
        default: return "Unknown Error";
    }
}

void VulkanDebugUtils::PrintMemoryStatistics(
    VkDevice device,
    VkPhysicalDevice physicalDevice)
{
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProps);

    SPDLOG_INFO("=== GPU Memory Statistics ===");
    SPDLOG_INFO("Memory Heaps: {}", memProps.memoryHeapCount);

    for (uint32_t i = 0; i < memProps.memoryHeapCount; i++) {
        const auto& heap = memProps.memoryHeaps[i];
        SPDLOG_INFO("  Heap {}: {:.1f} MB ({})",
                    i,
                    heap.size / (1024.0f * 1024.0f),
                    (heap.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) ? "device local" : "host");
    }

    SPDLOG_INFO("Memory Types: {}", memProps.memoryTypeCount);
    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
        const auto& type = memProps.memoryTypes[i];
        SPDLOG_INFO("  Type {}: heap {}, flags 0x{:x}",
                    i, type.heapIndex, type.propertyFlags);
    }
}

VKAPI_ATTR VkBool32 VKAPI_CALL VulkanDebugUtils::DebugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* pUserData)
{
    if (!s_DebugOutputEnabled) {
        return VK_FALSE;
    }

    std::string severity;
    switch (messageSeverity) {
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT:
            severity = "VERBOSE";
            break;
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT:
            severity = "INFO";
            SPDLOG_INFO("[Vulkan {}] {}", severity, pCallbackData->pMessage);
            break;
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT:
            severity = "WARNING";
            SPDLOG_WARN("[Vulkan {}] {}", severity, pCallbackData->pMessage);
            break;
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT:
            severity = "ERROR";
            SPDLOG_ERROR("[Vulkan {}] {}", severity, pCallbackData->pMessage);
            break;
        default:
            severity = "UNKNOWN";
            break;
    }

    return VK_FALSE;
}

