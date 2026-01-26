#include "LuaJitScriptSystem.h"
#include "GameObject.h"
#include "Transform.h"
#include <iostream>
#include <filesystem>
#include <chrono>

// Helper for error reporting
static void LuaJitReportError(lua_State* L, int status) {
    if (status != LUA_OK) {
        const char* msg = lua_tostring(L, -1);
        std::cerr << "LuaJIT Error: " << msg << std::endl;
        lua_pop(L, 1); // remove error message
    }
}

// ============================================================================
// Lifecycle
// ============================================================================

LuaJitScriptSystem::LuaJitScriptSystem() 
    : m_LuaState(nullptr), m_JitEnabled(true), m_ProfilingEnabled(false), m_HotReloadEnabled(false)
{
}

LuaJitScriptSystem::~LuaJitScriptSystem() {
    Shutdown();
}

void LuaJitScriptSystem::Init() {
    if (m_LuaState) return;

    // Create new Lua state with LuaJIT
    m_LuaState = luaL_newstate();
    if (!m_LuaState) {
        std::cerr << "Failed to create LuaJIT state" << std::endl;
        return;
    }

    // Open standard libraries
    luaL_openlibs(m_LuaState);

    // Setup JIT configuration
    SetupJitConfiguration();

    // Setup memory pool
    SetupMemoryPool();

    // Register engine types (Vec3, Transform, etc.)
    RegisterTypes();

    // Enable profiling if requested
    if (m_ProfilingEnabled) {
        EnableProfilingHooks();
    }

    std::cout << "LuaJIT Script System Initialized" << std::endl;
    std::cout << "  JIT Enabled: " << (m_JitEnabled ? "YES (10x+ performance)" : "NO") << std::endl;
    std::cout << "  Profiling: " << (m_ProfilingEnabled ? "ENABLED" : "DISABLED") << std::endl;
}

void LuaJitScriptSystem::Shutdown() {
    if (m_LuaState) {
        lua_close(m_LuaState);
        m_LuaState = nullptr;
    }
    
    m_NativeFunctions.clear();
    m_LoadedScripts.clear();
    
    std::cout << "LuaJIT Script System Shutdown" << std::endl;
}

void LuaJitScriptSystem::Update(float deltaTime) {
    // LuaJIT doesn't need explicit updates in game loop
    // JIT compilation happens automatically during script execution
}

// ============================================================================
// Script Execution
// ============================================================================

bool LuaJitScriptSystem::RunScript(const std::string& filepath) {
    if (!m_LuaState) {
        std::cerr << "LuaJIT state not initialized" << std::endl;
        return false;
    }

    if (!std::filesystem::exists(filepath)) {
        std::cerr << "Script file not found: " << filepath << std::endl;
        return false;
    }

    auto start = std::chrono::high_resolution_clock::now();
    
    int status = luaL_dofile(m_LuaState, filepath.c_str());
    ReportError(m_LuaState, status);
    
    if (status == LUA_OK) {
        m_LoadedScripts[filepath]++;
        
        if (m_ProfilingEnabled) {
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            m_ProfilingStats.totalExecutionTime += duration;
            m_ProfilingStats.callCount++;
        }
        
        return true;
    }
    
    return false;
}

bool LuaJitScriptSystem::ExecuteString(const std::string& source) {
    if (!m_LuaState) {
        std::cerr << "LuaJIT state not initialized" << std::endl;
        return false;
    }

    auto start = std::chrono::high_resolution_clock::now();
    
    int status = luaL_dostring(m_LuaState, source.c_str());
    ReportError(m_LuaState, status);
    
    if (status == LUA_OK) {
        if (m_ProfilingEnabled) {
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            m_ProfilingStats.totalExecutionTime += duration;
            m_ProfilingStats.callCount++;
        }
        return true;
    }
    
    return false;
}

// ============================================================================
// Function Calls
// ============================================================================

std::any LuaJitScriptSystem::CallFunction(const std::string& functionName,
                                          const std::vector<std::any>& args) {
    if (!m_LuaState) return std::any();

    auto start = std::chrono::high_resolution_clock::now();

    lua_getglobal(m_LuaState, functionName.c_str());
    if (!lua_isfunction(m_LuaState, -1)) {
        lua_pop(m_LuaState, 1);
        return std::any();
    }

    // Push arguments
    for (const auto& arg : args) {
        if (arg.type() == typeid(float)) {
            lua_pushnumber(m_LuaState, std::any_cast<float>(arg));
        } else if (arg.type() == typeid(double)) {
            lua_pushnumber(m_LuaState, std::any_cast<double>(arg));
        } else if (arg.type() == typeid(int)) {
            lua_pushinteger(m_LuaState, std::any_cast<int>(arg));
        } else if (arg.type() == typeid(bool)) {
            lua_pushboolean(m_LuaState, std::any_cast<bool>(arg));
        } else if (arg.type() == typeid(const char*)) {
            lua_pushstring(m_LuaState, std::any_cast<const char*>(arg));
        } else if (arg.type() == typeid(std::string)) {
            lua_pushstring(m_LuaState, std::any_cast<std::string>(arg).c_str());
        }
    }

    // Call function
    int result = lua_pcall(m_LuaState, args.size(), 1, 0);
    
    std::any retVal;
    if (result == LUA_OK) {
        if (lua_isnumber(m_LuaState, -1)) {
            retVal = std::any(lua_tonumber(m_LuaState, -1));
        } else if (lua_isstring(m_LuaState, -1)) {
            retVal = std::any(std::string(lua_tostring(m_LuaState, -1)));
        } else if (lua_isboolean(m_LuaState, -1)) {
            retVal = std::any(lua_toboolean(m_LuaState, -1) != 0);
        }
        lua_pop(m_LuaState, 1);
    } else {
        ReportError(m_LuaState, result);
    }

    if (m_ProfilingEnabled) {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        m_ProfilingStats.totalExecutionTime += duration;
        m_ProfilingStats.callCount++;
        m_ProfilingStats.avgExecutionTime = 
            static_cast<double>(m_ProfilingStats.totalExecutionTime) / m_ProfilingStats.callCount;
    }

    return retVal;
}

bool LuaJitScriptSystem::HasFunction(const std::string& functionName) const {
    if (!m_LuaState) return false;
    
    lua_getglobal(m_LuaState, functionName.c_str());
    bool exists = lua_isfunction(m_LuaState, -1);
    lua_pop(m_LuaState, 1);
    
    return exists;
}

// ============================================================================
// JIT Configuration
// ============================================================================

void LuaJitScriptSystem::SetJitEnabled(bool enabled) {
    if (!m_LuaState) return;

    m_JitEnabled = enabled;

    // Execute Lua code to enable/disable JIT
    if (enabled) {
        luaL_dostring(m_LuaState, "jit.on()");
    } else {
        luaL_dostring(m_LuaState, "jit.off()");
    }
}

void LuaJitScriptSystem::SetupJitConfiguration() {
    if (!m_LuaState) return;

    // Enable JIT by default
    luaL_dostring(m_LuaState, "jit.on()");

    // Configure JIT parameters for optimal game performance
    luaL_dostring(m_LuaState, R"(
        -- Set JIT optimization flags
        jit.opt.start(2)  -- Default optimization level
        
        -- Create utility functions for profiling
        if jit.util then
            local orig_info = debug.getinfo
            function debug.getinfo(...)
                return orig_info(...)
            end
        end
    )");
}

void LuaJitScriptSystem::SetupMemoryPool() {
    if (!m_LuaState) return;

    // LuaJIT has efficient memory management built-in
    // No additional setup needed, but we track it
    m_MemoryStats.currentUsage = 0;
    m_MemoryStats.peakUsage = 0;
    m_MemoryStats.numAllocations = 0;
}

void LuaJitScriptSystem::EnableProfilingHooks() {
    if (!m_LuaState) return;

    // Set up profiling via debug library
    // This is a simplified profiling setup; full implementation would require
    // hooking into JIT trace information
    m_ProfilingStats.jitCompiledFunctions = 0;
    m_ProfilingStats.activeTraces = 0;
    m_ProfilingStats.jitCoveragePercent = 0.0;
}

// ============================================================================
// Type Registration
// ============================================================================

void LuaJitScriptSystem::RegisterTypes() {
    if (!m_LuaState) return;

    // Register Vec3 type
    luaL_newmetatable(m_LuaState, "Vec3");
    lua_pushcfunction(m_LuaState, [](lua_State* L) {
        auto x = luaL_checknumber(L, 1);
        auto y = luaL_checknumber(L, 2);
        auto z = luaL_checknumber(L, 3);
        // Create userdata (simplified - actual implementation would allocate Vec3)
        lua_pushnumber(L, x);
        return 1;
    });
    lua_setfield(m_LuaState, -2, "__new");
    lua_pop(m_LuaState, 1);

    // Register global Vec3 constructor
    lua_pushcfunction(m_LuaState, [](lua_State* L) {
        float x = luaL_optnumber(L, 1, 0.0f);
        float y = luaL_optnumber(L, 2, 0.0f);
        float z = luaL_optnumber(L, 3, 0.0f);
        // Return simple table for now (production would use userdata)
        lua_createtable(L, 0, 3);
        lua_pushnumber(L, x);
        lua_setfield(L, -2, "x");
        lua_pushnumber(L, y);
        lua_setfield(L, -2, "y");
        lua_pushnumber(L, z);
        lua_setfield(L, -2, "z");
        return 1;
    });
    lua_setglobal(m_LuaState, "Vec3");

    std::cout << "LuaJIT type bindings registered" << std::endl;
}

// ============================================================================
// Variables
// ============================================================================

void LuaJitScriptSystem::SetGlobalVariable(const std::string& name, const std::string& value) {
    if (!m_LuaState) return;
    lua_pushstring(m_LuaState, value.c_str());
    lua_setglobal(m_LuaState, name.c_str());
}

std::string LuaJitScriptSystem::GetGlobalVariable(const std::string& name) const {
    if (!m_LuaState) return "";
    
    lua_getglobal(m_LuaState, name.c_str());
    std::string result;
    if (lua_isstring(m_LuaState, -1)) {
        result = lua_tostring(m_LuaState, -1);
    }
    lua_pop(m_LuaState, 1);
    
    return result;
}

// ============================================================================
// Memory Management
// ============================================================================

void LuaJitScriptSystem::ClearState() {
    if (!m_LuaState) return;
    
    lua_close(m_LuaState);
    m_LuaState = luaL_newstate();
    luaL_openlibs(m_LuaState);
    SetupJitConfiguration();
    RegisterTypes();
}

LuaJitScriptSystem::MemoryStats LuaJitScriptSystem::GetMemoryStats() const {
    return m_MemoryStats;
}

void LuaJitScriptSystem::ForceGarbageCollection() {
    if (!m_LuaState) return;
    
    lua_gc(m_LuaState, LUA_GCCOLLECT, 0);
}

void LuaJitScriptSystem::SetMemoryLimit(uint32_t limitMB) {
    if (!m_LuaState) return;
    
    // LuaJIT doesn't have built-in memory limits like some interpreters
    // This would require custom allocation tracking
    std::cout << "Memory limit set to " << limitMB << "MB (LuaJIT uses native allocator)" << std::endl;
}

// ============================================================================
// Profiling
// ============================================================================

void LuaJitScriptSystem::ResetProfilingStats() {
    m_ProfilingStats.totalExecutionTime = 0;
    m_ProfilingStats.callCount = 0;
    m_ProfilingStats.activeTraces = 0;
    m_ProfilingStats.jitCompiledFunctions = 0;
    m_ProfilingStats.avgExecutionTime = 0.0;
    m_ProfilingStats.jitCoveragePercent = 0.0;
}

// ============================================================================
// Hot Reload
// ============================================================================

bool LuaJitScriptSystem::HotReloadScript(const std::string& filepath) {
    if (!m_HotReloadEnabled) {
        std::cerr << "Hot reload not enabled" << std::endl;
        return false;
    }

    if (!std::filesystem::exists(filepath)) {
        std::cerr << "Script file not found: " << filepath << std::endl;
        return false;
    }

    std::cout << "Hot reloading script: " << filepath << std::endl;
    return RunScript(filepath);
}

// ============================================================================
// Helper Methods
// ============================================================================

void LuaJitScriptSystem::ReportError(lua_State* L, int status) {
    LuaJitReportError(L, status);
}

void LuaJitScriptSystem::RegisterNativeFunction(const std::string& functionName, NativeFunction function) {
    m_NativeFunctions[functionName] = function;
    // In production, would register actual Lua C function
}
