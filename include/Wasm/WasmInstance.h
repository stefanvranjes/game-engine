#pragma once

#include <string>
#include <memory>
#include <vector>
#include <functional>
#include <any>
#include "WasmModule.h"

// Forward declarations
class WasmModule;

/**
 * @class WasmInstance
 * @brief Represents an instantiated WASM module
 * 
 * Executes WASM functions and manages their state including
 * memory, globals, and call stacks.
 */
class WasmInstance {
public:
    /**
     * Get parent module
     */
    std::shared_ptr<WasmModule> GetModule() const { return m_Module; }

    /**
     * Call exported WASM function with no arguments
     * @param functionName Name of exported function
     * @return Return value as WasmValue
     */
    WasmValue Call(const std::string& functionName);

    /**
     * Call exported WASM function with arguments
     * @param functionName Name of exported function
     * @param args Arguments to pass
     * @return Return value as WasmValue
     */
    WasmValue Call(const std::string& functionName, const std::vector<WasmValue>& args);

    /**
     * Call with type-safe parameter forwarding
     * Variadic template version for compile-time type safety
     */
    template <typename... Args>
    WasmValue CallTyped(const std::string& functionName, Args... args) {
        std::vector<WasmValue> wasmArgs;
        PackArguments(wasmArgs, args...);
        return Call(functionName, wasmArgs);
    }

    /**
     * Direct memory access
     * Get pointer to WASM linear memory (use with care!)
     */
    uint8_t* GetMemory() const { return m_Memory; }

    /**
     * Get WASM memory size in bytes
     */
    size_t GetMemorySize() const { return m_MemorySize; }

    /**
     * Safe memory read (with bounds checking)
     * @param offset Offset in WASM memory
     * @param size Number of bytes to read
     * @return Vector of bytes
     */
    std::vector<uint8_t> ReadMemory(uint32_t offset, uint32_t size) const;

    /**
     * Safe memory write (with bounds checking)
     * @param offset Offset in WASM memory
     * @param data Data to write
     * @return true if successful
     */
    bool WriteMemory(uint32_t offset, const std::vector<uint8_t>& data);

    /**
     * Read string from WASM memory (null-terminated)
     * @param offset Offset in WASM memory
     * @return String value
     */
    std::string ReadString(uint32_t offset) const;

    /**
     * Write string to WASM memory
     * @param offset Offset in WASM memory
     * @param str String to write
     * @return true if successful
     */
    bool WriteString(uint32_t offset, const std::string& str);

    /**
     * Allocate memory in WASM heap
     * Requires exported malloc function or equivalent
     * @param size Number of bytes to allocate
     * @return Offset in WASM memory, 0 if failed
     */
    uint32_t Malloc(uint32_t size);

    /**
     * Free allocated memory
     * Requires exported free function or equivalent
     * @param offset Offset returned from Malloc
     */
    void Free(uint32_t offset);

    /**
     * Get/Set global variable
     * @param name Name of global export
     */
    WasmValue GetGlobal(const std::string& name) const;
    bool SetGlobal(const std::string& name, const WasmValue& value);

    /**
     * Get instance memory statistics
     */
    struct InstanceStats {
        size_t totalMemory;
        size_t usedMemory;
        uint32_t callDepth;
        uint64_t totalCalls;
        uint64_t lastCallTime;  // Microseconds
    };

    InstanceStats GetStats() const;

    /**
     * Reset instance state (clear globals, reset memory)
     */
    void Reset();

    /**
     * Get last error message
     */
    std::string GetLastError() const { return m_LastError; }

    /**
     * Profiling: Get execution time of last call
     */
    double GetLastCallDuration() const { return m_LastCallDuration; }

    /**
     * Register a host callback to be called from WASM
     * Allows WASM to call back into the engine
     */
    using HostCallback = std::function<WasmValue(const std::vector<WasmValue>&)>;
    bool RegisterHostCallback(const std::string& name, HostCallback callback);

    /**
     * Enable/disable execution profiling
     */
    void SetProfilingEnabled(bool enabled) { m_ProfilingEnabled = enabled; }

    /**
     * Get profiling data for all function calls
     */
    struct ProfilingData {
        std::string functionName;
        uint32_t callCount;
        double totalTime;  // Milliseconds
        double averageTime;
        double minTime;
        double maxTime;
    };

    std::vector<ProfilingData> GetProfilingData() const;

    /**
     * Bind engine object for access from WASM
     * Stores opaque pointer accessible via host callbacks
     */
    void SetEngineObject(void* engineObj) { m_EngineObject = engineObj; }
    void* GetEngineObject() const { return m_EngineObject; }

    WasmInstance(std::shared_ptr<WasmModule> module);
    ~WasmInstance();

private:
    friend class WasmModule;
    friend class WasmRuntime;

    std::shared_ptr<WasmModule> m_Module;
    
    uint8_t* m_Memory = nullptr;
    size_t m_MemorySize = 0;
    
    void* m_Execution = nullptr;  // Opaque pointer to M3Execution
    
    // Error tracking
    mutable std::string m_LastError;
    double m_LastCallDuration = 0.0;
    
    // Profiling
    bool m_ProfilingEnabled = false;
    std::unordered_map<std::string, ProfilingData> m_ProfilingMap;
    
    // Host callbacks and engine binding
    std::unordered_map<std::string, HostCallback> m_HostCallbacks;
    void* m_EngineObject = nullptr;

    // Argument packing for variadic templates
    template <typename T>
    void PackArguments(std::vector<WasmValue>& args, T arg) {
        PackSingleArgument(args, arg);
    }

    template <typename T, typename... Rest>
    void PackArguments(std::vector<WasmValue>& args, T arg, Rest... rest) {
        PackSingleArgument(args, arg);
        PackArguments(args, rest...);
    }

    void PackSingleArgument(std::vector<WasmValue>& args, int32_t val);
    void PackSingleArgument(std::vector<WasmValue>& args, int64_t val);
    void PackSingleArgument(std::vector<WasmValue>& args, float val);
    void PackSingleArgument(std::vector<WasmValue>& args, double val);
};

