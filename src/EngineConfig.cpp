#include "EngineConfig.h"
#include <spdlog/spdlog.h>
#include <cstdlib>

EngineConfig g_EngineConfig;

EngineConfig EngineConfig::LoadFromEnvironment() {
    EngineConfig config;

    // Graphics API
    const char* apiEnv = std::getenv("GE_GRAPHICS_API");
    if (apiEnv) {
        std::string api(apiEnv);
        if (api == "vulkan") {
            config.preferredGraphicsAPI = GraphicsAPI::Vulkan;
            SPDLOG_INFO("Graphics API set to Vulkan from environment");
        } else if (api == "opengl") {
            config.preferredGraphicsAPI = GraphicsAPI::OpenGL;
            SPDLOG_INFO("Graphics API set to OpenGL from environment");
        }
    }

    // Multi-GPU
    const char* multiGpuEnv = std::getenv("GE_MULTI_GPU");
    if (multiGpuEnv) {
        config.enableMultiGPU = (std::string(multiGpuEnv) == "true" || std::string(multiGpuEnv) == "1");
        SPDLOG_INFO("Multi-GPU: {}", config.enableMultiGPU ? "enabled" : "disabled");
    }

    // GPU Count
    const char* gpuCountEnv = std::getenv("GE_GPU_COUNT");
    if (gpuCountEnv) {
        config.forcedGPUCount = std::stoi(gpuCountEnv);
        SPDLOG_INFO("Forcing GPU count: {}", config.forcedGPUCount);
    }

    // Vulkan Validation
    const char* vulkanValEnv = std::getenv("GE_VULKAN_VALIDATION");
    if (vulkanValEnv) {
        config.enableVulkanValidation = (std::string(vulkanValEnv) == "true" || std::string(vulkanValEnv) == "1");
        SPDLOG_INFO("Vulkan validation: {}", config.enableVulkanValidation ? "enabled" : "disabled");
    }

    // PhysX Visual Debugger
    const char* pvdEnv = std::getenv("GE_PHYSX_PVD");
    if (pvdEnv) {
        config.enablePhysXVisualDebugger = (std::string(pvdEnv) == "true" || std::string(pvdEnv) == "1");
        SPDLOG_INFO("PhysX Visual Debugger: {}", config.enablePhysXVisualDebugger ? "enabled" : "disabled");
    }

    // PhysX Determinism
    const char* detEnv = std::getenv("GE_PHYSX_DETERMINISM");
    if (detEnv) {
        config.enableDeterminism = (std::string(detEnv) == "true" || std::string(detEnv) == "1");
        SPDLOG_INFO("PhysX Determinism: {}", config.enableDeterminism ? "enabled" : "disabled");
    }

    return config;
}

EngineConfig EngineConfig::LoadFromFile(const std::string& filepath) {
    EngineConfig config;
    // TODO: Implement JSON file loading
    SPDLOG_WARN("Config file loading not yet implemented: {}", filepath);
    return config;
}

void EngineConfig::SaveToFile(const std::string& filepath) const {
    // TODO: Implement JSON file saving
    SPDLOG_WARN("Config file saving not yet implemented: {}", filepath);
}

