#pragma once

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

/**
 * @brief GLSL to SPIR-V shader compilation utilities
 * 
 * Compiles GLSL shaders to SPIR-V bytecode for Vulkan.
 * Includes caching and error reporting.
 */
class VulkanShaderCompiler {
public:
    enum class ShaderStage {
        Vertex,
        Fragment,
        Geometry,
        TessControl,
        TessEval,
        Compute
    };

    struct CompileResult {
        bool success = false;
        std::vector<uint32_t> spirvBytecode;
        std::string errorMessage;
        std::string warningMessage;
        size_t compilationTimeMs = 0;
    };

    /**
     * @brief Compile GLSL source to SPIR-V
     * @param glslSource GLSL shader source code
     * @param stage Shader stage (vertex, fragment, etc.)
     * @param entryPoint Function entry point (default: "main")
     * @param optimization Optimization level (0-2, default: 2)
     * @return Compilation result with SPIR-V bytecode or error info
     */
    static CompileResult CompileGLSL(
        const std::string& glslSource,
        ShaderStage stage,
        const std::string& entryPoint = "main",
        int optimizationLevel = 2);

    /**
     * @brief Compile and cache shader
     * @param shaderName Unique identifier for caching
     * @param glslSource Source code
     * @param stage Shader stage
     * @return SPIR-V bytecode from cache or compilation
     */
    static std::vector<uint32_t> CompileAndCache(
        const std::string& shaderName,
        const std::string& glslSource,
        ShaderStage stage);

    /**
     * @brief Clear shader cache
     */
    static void ClearCache();

    /**
     * @brief Validate SPIR-V bytecode
     * @return true if valid SPIR-V
     */
    static bool ValidateSPIRV(const std::vector<uint32_t>& spirv);

    /**
     * @brief Disassemble SPIR-V to human-readable format (for debugging)
     */
    static std::string DisassembleSPIRV(const std::vector<uint32_t>& spirv);

    /**
     * @brief Extract shader metadata (push constants, descriptors, etc.)
     */
    struct ShaderMetadata {
        std::vector<std::string> uniformBuffers;
        std::vector<std::string> storageBuffers;
        std::vector<std::string> samplers;
        std::vector<std::string> images;
        size_t pushConstantSize = 0;
        uint32_t workgroupSizeX = 1;
        uint32_t workgroupSizeY = 1;
        uint32_t workgroupSizeZ = 1;
    };

    static ShaderMetadata ExtractMetadata(const std::vector<uint32_t>& spirv);

private:
    static std::unordered_map<std::string, std::vector<uint32_t>> s_ShaderCache;

    static const char* GetShaderStageName(ShaderStage stage);
    static int GetGlslangShaderStage(ShaderStage stage);
};

