#pragma once

#include <string>
#include <memory>
#include <unordered_map>
#include <functional>
#include <vector>
#include <cstdint>

// Forward declarations
class WasmModule;
class WasmInstance;

/**
 * @class WasmRuntime
 * @brief Core WebAssembly runtime environment
 * 
 * Manages WASM module loading, instantiation, and execution.
 * Uses wasm3 as the underlying WASM interpreter.
 */
class WasmRuntime {
public:
    using ExportFunction = std::function<void(void*)>;

    static WasmRuntime& GetInstance() {
        static WasmRuntime instance;
        return instance;
    }

    /**
     * Initialize the WASM runtime
     * @return true if initialization successful
     */
    bool Initialize();

    /**
     * Shutdown the WASM runtime and cleanup resources
     */
    void Shutdown();

    /**
     * Load a WASM module from file
     * @param filepath Path to .wasm file
     * @return Shared pointer to loaded module, nullptr if failed
     */
    std::shared_ptr<WasmModule> LoadModule(const std::string& filepath);

    /**
     * Load a WASM module from memory buffer
     * @param data Raw WASM binary data
     * @param size Size of data in bytes
     * @return Shared pointer to loaded module, nullptr if failed
     */
    std::shared_ptr<WasmModule> LoadModuleFromMemory(const uint8_t* data, size_t size);

    /**
     * Unload a WASM module
     * @param moduleName Name of module to unload
     */
    void UnloadModule(const std::string& moduleName);

    /**
     * Get loaded module by name
     * @param moduleName Name of module
     * @return Shared pointer to module, nullptr if not found
     */
    std::shared_ptr<WasmModule> GetModule(const std::string& moduleName);

    /**
     * List all loaded modules
     * @return Vector of module names
     */
    std::vector<std::string> GetLoadedModules() const;

    /**
     * Enable/disable WASM memory protection
     * Adds bounds checking to WASM memory access
     */
    void SetMemoryProtection(bool enabled) { m_MemoryProtectionEnabled = enabled; }

    /**
     * Set maximum allowed WASM memory size (in MB)
     * Default: 256 MB
     */
    void SetMaxMemorySize(uint32_t sizeInMB) { m_MaxMemorySizeMB = sizeInMB; }

    /**
     * Get memory statistics
     */
    struct MemoryStats {
        uint32_t totalAllocated;    // Total bytes allocated
        uint32_t currentUsage;      // Currently used bytes
        uint32_t peakUsage;         // Peak usage since reset
        uint32_t numAllocations;    // Number of active allocations
    };

    MemoryStats GetMemoryStats() const;

    /**
     * Enable/disable WASM execution timeout
     * @param enabled Enable timeout enforcement
     * @param timeoutMs Timeout in milliseconds (default 5000)
     */
    void SetExecutionTimeout(bool enabled, uint32_t timeoutMs = 5000);

    /**
     * Check if runtime is initialized
     */
    bool IsInitialized() const { return m_Initialized; }

    /**
     * Get error string from last operation
     */
    std::string GetLastError() const { return m_LastError; }

    /**
     * Clear error state
     */
    void ClearLastError() { m_LastError.clear(); }

private:
    WasmRuntime() = default;
    ~WasmRuntime();

    // Prevent copying
    WasmRuntime(const WasmRuntime&) = delete;
    WasmRuntime& operator=(const WasmRuntime&) = delete;

    bool m_Initialized = false;
    bool m_MemoryProtectionEnabled = true;
    uint32_t m_MaxMemorySizeMB = 256;
    uint32_t m_ExecutionTimeoutMs = 5000;
    bool m_ExecutionTimeoutEnabled = true;

    std::unordered_map<std::string, std::shared_ptr<WasmModule>> m_Modules;
    std::string m_LastError;

    void* m_Runtime = nullptr;  // Opaque pointer to wasm3::WasmMachine
    void* m_Environment = nullptr;  // Opaque pointer to M3Environment
};

