#pragma once

#include "IScriptSystem.h"
#include <string>
#include <memory>
#include <unordered_map>
#include <functional>
#include <cstdint>

extern "C" {
#include "lua.h"
#include "lualib.h"
#include "lauxlib.h"
}

// Forward declarations
class GameObject;

/**
 * @class LuaJitScriptSystem
 * @brief LuaJIT script system with JIT compilation support
 * 
 * Provides Lua scripting via LuaJIT, which offers 10x+ performance improvement
 * over standard Lua through Just-In-Time compilation and aggressive optimization.
 * 
 * Features:
 * - Full Lua 5.1 compatibility with 5.2 mode support
 * - JIT compilation for hot loops
 * - FFI (Foreign Function Interface) for C/C++ interop
 * - Higher performance memory management
 * - Profiling hooks and optimization metrics
 * 
 * Performance Characteristics:
 * - Startup time: ~1ms (slightly higher than standard Lua due to JIT)
 * - Warm execution: 10-20x faster than standard Lua for CPU-bound code
 * - Memory per VM: ~300KB (more efficient than standard Lua)
 * - Best for: Game logic, AI, physics callbacks, event handlers
 * 
 * Usage:
 * ```cpp
 * auto& jit = LuaJitScriptSystem::GetInstance();
 * jit.Init();
 * jit.RunScript("scripts/game_logic.lua");
 * 
 * // Call Lua functions from C++
 * std::vector<std::any> args = {someObject, 0.016f};
 * jit.CallFunction("update_game", args);
 * 
 * // Enable profiling to measure performance
 * jit.SetProfilingEnabled(true);
 * auto stats = jit.GetProfilingStats();
 * ```
 */
class LuaJitScriptSystem : public IScriptSystem {
public:
    static LuaJitScriptSystem& GetInstance() {
        static LuaJitScriptSystem instance;
        return instance;
    }

    // IScriptSystem Implementation
    void Init() override;
    void Shutdown() override;
    void Update(float deltaTime) override;
    bool RunScript(const std::string& filepath) override;
    bool ExecuteString(const std::string& source) override;

    // Language metadata
    ScriptLanguage GetLanguage() const override { return ScriptLanguage::Lua; }
    ScriptExecutionMode GetExecutionMode() const override { return ScriptExecutionMode::Interpreted; }
    std::string GetLanguageName() const override { return "LuaJIT"; }
    std::string GetFileExtension() const override { return ".lua"; }

    // LuaJIT-specific API
    /**
     * Get the underlying Lua state for advanced operations
     */
    lua_State* GetLuaState() const { return m_LuaState; }

    /**
     * Enable/disable JIT compilation (default: enabled)
     * Disabling reduces memory usage but loses performance gains
     */
    void SetJitEnabled(bool enabled);

    /**
     * Get whether JIT is currently enabled
     */
    bool IsJitEnabled() const { return m_JitEnabled; }

    /**
     * Enable/disable profiling and performance metrics
     * Slight overhead when enabled (~2-5%)
     */
    void SetProfilingEnabled(bool enabled) { m_ProfilingEnabled = enabled; }

    /**
     * Profiling statistics for performance analysis
     */
    struct ProfilingStats {
        uint64_t totalExecutionTime;    // Total time spent in Lua
        uint64_t callCount;              // Number of calls made
        uint32_t activeTraces;           // Number of compiled traces
        uint32_t jitCompiledFunctions;   // Count of JIT-compiled functions
        double avgExecutionTime;         // Average execution time per call
        double jitCoveragePercent;       // % of code running through JIT
    };

    /**
     * Get current profiling statistics
     * Requires SetProfilingEnabled(true) to work
     */
    ProfilingStats GetProfilingStats() const { return m_ProfilingStats; }

    /**
     * Reset profiling statistics
     */
    void ResetProfilingStats();

    /**
     * Register a global function callable from Lua
     * @param functionName Name of function in Lua
     * @param function C++ function to call
     */
    using NativeFunction = std::function<void(void*)>;
    void RegisterNativeFunction(const std::string& functionName, NativeFunction function);

    /**
     * Register a type binding for Lua (Vec3, Transform, etc.)
     * Must be called during Init() before scripts run
     */
    void RegisterTypes() override;

    /**
     * Call a Lua function with arbitrary arguments
     * @param functionName Name of Lua function to call
     * @param args Arguments to pass
     * @return Return value from function
     */
    std::any CallFunction(const std::string& functionName,
                         const std::vector<std::any>& args) override;

    /**
     * Check if a function exists in the Lua state
     */
    bool HasFunction(const std::string& functionName) const override;

    /**
     * Set a global variable in Lua
     */
    void SetGlobalVariable(const std::string& name, const std::string& value);

    /**
     * Get a global variable from Lua as string
     */
    std::string GetGlobalVariable(const std::string& name) const;

    /**
     * Clear all Lua state and memory
     * Useful for resetting between levels
     */
    void ClearState();

    /**
     * Get memory usage statistics
     */
    struct MemoryStats {
        uint32_t currentUsage;      // Current memory used (bytes)
        uint32_t peakUsage;         // Peak memory usage (bytes)
        uint32_t numAllocations;    // Number of allocations
    };

    MemoryStats GetMemoryStats() const;

    /**
     * Force garbage collection
     * Blocks until collection completes
     */
    void ForceGarbageCollection();

    /**
     * Set memory limit (in MB)
     * Default: 256 MB
     */
    void SetMemoryLimit(uint32_t limitMB);

    /**
     * Enable hot-reload support for development
     * Allows scripts to be reloaded without full shutdown
     */
    void SetHotReloadEnabled(bool enabled) { m_HotReloadEnabled = enabled; }

    /**
     * Check if hot-reload is enabled
     */
    bool IsHotReloadEnabled() const { return m_HotReloadEnabled; }

    /**
     * Reload a script at runtime (requires hot-reload enabled)
     * @param filepath Path to script file
     * @return true if successful
     */
    bool HotReloadScript(const std::string& filepath);

    LuaJitScriptSystem();
    ~LuaJitScriptSystem();

private:

    lua_State* m_LuaState = nullptr;
    bool m_JitEnabled = true;
    bool m_ProfilingEnabled = false;
    bool m_HotReloadEnabled = false;
    
    ProfilingStats m_ProfilingStats = {};
    MemoryStats m_MemoryStats = {};
    
    std::unordered_map<std::string, NativeFunction> m_NativeFunctions;
    std::unordered_map<std::string, uint32_t> m_LoadedScripts; // For hot-reload tracking

    // Helper methods
    static void ReportError(lua_State* L, int status);
    void SetupMemoryPool();
    void SetupJitConfiguration();
    void EnableProfilingHooks();
};
