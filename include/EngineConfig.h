#pragma once

#include <string>
#include <cstdint>

/**
 * @brief Graphics API selection
 */
enum class GraphicsAPI {
    OpenGL,  ///< OpenGL 3.3+ (default, always available)
    Vulkan   ///< Vulkan 1.3+ (optional, if ENABLE_VULKAN is defined)
};

/**
 * @brief Multi-GPU rendering strategy
 */
enum class MultiGPUStrategy {
    Single,         ///< Use single GPU only
    SplitFrame,     ///< Each GPU renders screen region
    AlternateFrame  ///< GPUs render alternate frames
};

/**
 * @brief Engine-wide configuration
 * 
 * Load from environment variables or config file:
 * - GE_GRAPHICS_API: "opengl" or "vulkan"
 * - GE_GPU_COUNT: Force GPU count (1, 2, 4, etc.)
 * - GE_MULTI_GPU: "true" or "false"
 * - GE_VULKAN_VALIDATION: "true" or "false"
 */
struct EngineConfig {
    // Graphics API selection
    GraphicsAPI preferredGraphicsAPI = GraphicsAPI::OpenGL;
    
    // Multi-GPU configuration
    bool enableMultiGPU = true;
    MultiGPUStrategy multiGPUStrategy = MultiGPUStrategy::SplitFrame;
    int32_t forcedGPUCount = -1; ///< -1 = auto-detect all available
    
    // Vulkan-specific options
    bool enableVulkanValidation = true;
    bool enableVulkanDebugUtils = true;
    
    // Performance monitoring
    bool enableGPUProfiling = true;
    bool enableFrameTimeRecording = false;

    // PhysX Debugging
    bool enablePhysXVisualDebugger = false;
    
    // PhysX Determinism
    bool enableDeterminism = false;
    
    // Window/display settings
    uint32_t windowWidth = 1920;
    uint32_t windowHeight = 1080;
    bool vsyncEnabled = true;
    
    // Rendering features
    bool enableDeferred = true;
    bool enablePostProcessing = true;
    bool enableOcclusionCulling = true;
    
    /**
     * @brief Load configuration from environment variables
     */
    static EngineConfig LoadFromEnvironment();
    
    /**
     * @brief Load configuration from JSON file
     */
    static EngineConfig LoadFromFile(const std::string& filepath);
    
    /**
     * @brief Save configuration to JSON file
     */
    void SaveToFile(const std::string& filepath) const;
};

// Global configuration instance
extern EngineConfig g_EngineConfig;

