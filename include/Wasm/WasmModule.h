#pragma once

#include <string>
#include <memory>
#include <unordered_map>
#include <functional>
#include <vector>
#include <any>
#include <glm/glm.hpp>

// Forward declarations
class WasmInstance;

/**
 * @enum WasmValueType
 * Supported WASM value types for function calls
 */
enum class WasmValueType {
    I32,        // 32-bit integer
    I64,        // 64-bit integer
    F32,        // 32-bit float
    F64,        // 64-bit double
    V128,       // 128-bit vector (SIMD)
};

/**
 * @struct WasmValue
 * Type-safe wrapper for WASM values
 */
struct WasmValue {
    WasmValueType type;
    std::any value;

    // Factory methods
    static WasmValue I32(int32_t v);
    static WasmValue I64(int64_t v);
    static WasmValue F32(float v);
    static WasmValue F64(double v);
    static WasmValue Ptr(uintptr_t ptr);
};

/**
 * @class WasmModule
 * @brief Represents a loaded WASM module
 * 
 * Provides access to module exports and allows instantiation
 * and execution of WASM functions.
 */
class WasmModule {
public:
    /**
     * Get module name
     */
    const std::string& GetName() const { return m_Name; }

    /**
     * Get module file path
     */
    const std::string& GetFilePath() const { return m_FilePath; }

    /**
     * List all exported functions
     */
    std::vector<std::string> GetExportedFunctions() const;

    /**
     * List all exported memories
     */
    std::vector<std::string> GetExportedMemories() const;

    /**
     * List all exported globals
     */
    std::vector<std::string> GetExportedGlobals() const;

    /**
     * List all exported tables
     */
    std::vector<std::string> GetExportedTables() const;

    /**
     * Check if function is exported
     */
    bool HasExportedFunction(const std::string& name) const;

    /**
     * Get function signature (parameter and return types)
     */
    struct FunctionSignature {
        std::vector<WasmValueType> paramTypes;
        std::vector<WasmValueType> returnTypes;
    };

    bool GetFunctionSignature(const std::string& name, FunctionSignature& outSig) const;

    /**
     * Create an instance of this module
     * Multiple instances can be created from one module
     */
    std::shared_ptr<WasmInstance> CreateInstance();

    /**
     * Get module size in bytes
     */
    size_t GetModuleSize() const { return m_ModuleData.size(); }

    /**
     * Get module version (from custom section if available)
     */
    std::string GetVersion() const { return m_Version; }

    /**
     * Get memory requirements
     */
    struct MemoryRequirements {
        uint32_t minPages;  // Minimum memory pages (64KB each)
        uint32_t maxPages;  // Maximum memory pages (0 = unbounded)
    };

    MemoryRequirements GetMemoryRequirements() const { return m_MemoryRequirements; }

    /**
     * Validate module structure without executing
     */
    bool Validate() const;

    /**
     * Get detailed module information
     */
    struct ModuleInfo {
        std::string name;
        std::string filepath;
        std::string version;
        size_t sizeBytes;
        uint32_t numFunctions;
        uint32_t numExports;
        bool validated;
        std::vector<std::string> exportedFunctions;
    };

    ModuleInfo GetInfo() const;

private:
    friend class WasmRuntime;

    WasmModule(const std::string& name, const std::string& filepath);

    std::string m_Name;
    std::string m_FilePath;
    std::string m_Version;
    std::vector<uint8_t> m_ModuleData;
    MemoryRequirements m_MemoryRequirements = {1, 256};  // Default: 64KB to 16MB

    // Module exports cache
    mutable std::unordered_map<std::string, FunctionSignature> m_FunctionSignatures;
    mutable std::vector<std::string> m_ExportedFunctions;
    mutable bool m_ExportsCached = false;

    void* m_Module = nullptr;  // Opaque pointer to M3Module

    void CacheExports() const;
};

