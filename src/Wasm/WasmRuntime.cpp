#include "Wasm/WasmRuntime.h"
#include "Wasm/WasmModule.h"
#include "Wasm/WasmInstance.h"
#include <fstream>
#include <algorithm>
#include <chrono>

// wasm3 includes - will be added via CMakeLists.txt FetchContent
#include <wasm3.h>
#include <m3_env.h>
#include <m3_api_wasi.h>

bool WasmRuntime::Initialize() {
    if (m_Initialized) {
        return true;
    }

    try {
        // Create M3 environment
        m_Environment = (void*)m3_NewEnvironment();
        if (!m_Environment) {
            m_LastError = "Failed to create M3 environment";
            return false;
        }

        // Initialize runtime
        m_Runtime = (void*)m3_NewRuntime((IM3Environment)m_Environment, m_MaxMemorySizeMB * 1024, nullptr);
        if (!m_Runtime) {
            m_LastError = "Failed to create M3 runtime";
            m3_FreeEnvironment((IM3Environment)m_Environment);
            m_Environment = nullptr;
            return false;
        }

        m_Initialized = true;
        m_LastError.clear();
        return true;
    }
    catch (const std::exception& e) {
        m_LastError = std::string("Exception during WASM runtime initialization: ") + e.what();
        return false;
    }
}

void WasmRuntime::Shutdown() {
    if (!m_Initialized) {
        return;
    }

    // Unload all modules
    auto moduleNames = GetLoadedModules();
    for (const auto& name : moduleNames) {
        UnloadModule(name);
    }
    m_Modules.clear();

    // Cleanup M3 environment
    if (m_Environment) {
        m3_FreeEnvironment((M3Environment*)m_Environment);
        m_Environment = nullptr;
    }

    m_Runtime = nullptr;
    m_Initialized = false;
}

std::shared_ptr<WasmModule> WasmRuntime::LoadModule(const std::string& filepath) {
    if (!m_Initialized) {
        m_LastError = "WASM runtime not initialized";
        return nullptr;
    }

    // Read file
    std::ifstream file(filepath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        m_LastError = std::string("Failed to open WASM file: ") + filepath;
        return nullptr;
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> data(size);
    if (!file.read(reinterpret_cast<char*>(data.data()), size)) {
        m_LastError = std::string("Failed to read WASM file: ") + filepath;
        return nullptr;
    }

    return LoadModuleFromMemory(data.data(), data.size());
}

std::shared_ptr<WasmModule> WasmRuntime::LoadModuleFromMemory(const uint8_t* data, size_t size) {
    if (!m_Initialized) {
        m_LastError = "WASM runtime not initialized";
        return nullptr;
    }

    if (!data || size == 0) {
        m_LastError = "Invalid WASM data";
        return nullptr;
    }

    try {
        // Extract module name from binary or use hash
        std::string moduleName = "wasm_module_" + std::to_string(m_Modules.size());

        // Create module wrapper
        auto module = std::shared_ptr<WasmModule>(new WasmModule(moduleName, ""));
        if (!module) {
            m_LastError = "Failed to create WasmModule";
            return nullptr;
        }

        // Verify module magic number
        if (size < 4 || data[0] != 0x00 || data[1] != 0x61 || 
            data[2] != 0x73 || data[3] != 0x6d) {
            m_LastError = "Invalid WASM magic number";
            return nullptr;
        }

        // Store module data
        module->m_ModuleData.assign(data, data + size);

        // Parse module with wasm3
        M3Result result = m3_ParseModule((M3Environment*)m_Environment, 
                                         (M3Module**)&module->m_Module, 
                                         module->m_ModuleData.data(), 
                                         module->m_ModuleData.size());
        
        if (result != m3Err_none) {
            m_LastError = std::string("Failed to parse WASM module: ") + result;
            return nullptr;
        }

        // Register module with runtime
        if (m_Runtime) {
            // Set up WASI compatibility if needed
            m3_LinkWASI((IM3Module)module->m_Module);
            
            result = m3_LoadModule((IM3Runtime)m_Runtime, (IM3Module)module->m_Module);
            if (result != m3Err_none) {
                m_LastError = std::string("Failed to load module into runtime: ") + result;
                return nullptr;
            }
        }

        // Cache exports
        module->CacheExports();

        m_Modules[moduleName] = module;
        m_LastError.clear();
        return module;
    }
    catch (const std::exception& e) {
        m_LastError = std::string("Exception loading WASM module: ") + e.what();
        return nullptr;
    }
}

void WasmRuntime::UnloadModule(const std::string& moduleName) {
    auto it = m_Modules.find(moduleName);
    if (it != m_Modules.end()) {
        // Module will be cleaned up when shared_ptr reference count reaches 0
        m_Modules.erase(it);
    }
}

std::shared_ptr<WasmModule> WasmRuntime::GetModule(const std::string& moduleName) {
    auto it = m_Modules.find(moduleName);
    if (it != m_Modules.end()) {
        return it->second;
    }
    return nullptr;
}

std::vector<std::string> WasmRuntime::GetLoadedModules() const {
    std::vector<std::string> names;
    for (const auto& pair : m_Modules) {
        names.push_back(pair.first);
    }
    return names;
}

WasmRuntime::MemoryStats WasmRuntime::GetMemoryStats() const {
    MemoryStats stats = {0, 0, 0, 0};
    
    // Track memory usage across all modules
    for (const auto& pair : m_Modules) {
        // This would require wasm3 API to track per-module memory
        // For now, return estimated values
    }

    return stats;
}

void WasmRuntime::SetExecutionTimeout(bool enabled, uint32_t timeoutMs) {
    m_ExecutionTimeoutEnabled = enabled;
    m_ExecutionTimeoutMs = timeoutMs;
}

WasmRuntime::~WasmRuntime() {
    Shutdown();
}

