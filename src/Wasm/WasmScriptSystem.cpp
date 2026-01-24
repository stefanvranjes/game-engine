#include "Wasm/WasmScriptSystem.h"
#include "Wasm/WasmRuntime.h"
#include "Wasm/WasmModule.h"
#include "Wasm/WasmInstance.h"
#include "GameObject.h"
#include <algorithm>
#include <filesystem>

namespace fs = std::filesystem;

void WasmScriptSystem::Init() {
    auto& runtime = WasmRuntime::GetInstance();
    if (!runtime.IsInitialized()) {
        if (!runtime.Initialize()) {
            // Log error but continue - WASM might not be available
        }
    }
}

void WasmScriptSystem::Shutdown() {
    // Unload all modules
    auto moduleNames = GetLoadedModules();
    for (const auto& name : moduleNames) {
        CallWasmShutdown(name);
        UnloadModule(name);
    }
    m_ModuleInstances.clear();
    m_BoundGameObjects.clear();
    m_EngineExports.clear();
    m_PerformanceMetrics.clear();
}

void WasmScriptSystem::Update(float deltaTime) {
    m_AccumulatedTime += deltaTime;

    auto& runtime = WasmRuntime::GetInstance();
    
    // Update all loaded modules
    auto moduleNames = GetLoadedModules();
    for (const auto& name : moduleNames) {
        // Call update function if it exists
        CallWasmUpdate(name, deltaTime);

        // Check for hot reload if enabled
        if (m_HotReloadEnabled) {
            auto module = runtime.GetModule(name);
            if (module) {
                UpdateModuleIfChanged(name, module->GetFilePath());
            }
        }
    }
}

bool WasmScriptSystem::RunScript(const std::string& filepath) {
    if (!fs::exists(filepath)) {
        return false;
    }

    std::string moduleName = ExtractModuleName(filepath);
    return LoadWasmModule(filepath, moduleName);
}

bool WasmScriptSystem::ExecuteString(const std::string& source) {
    // WASM is compiled bytecode, cannot execute from string
    // This would require compiling WebAssembly from text format (WAT)
    // which is beyond the scope of basic WASM support
    return false;
}

bool WasmScriptSystem::LoadWasmModule(const std::string& filepath, const std::string& moduleName) {
    auto& runtime = WasmRuntime::GetInstance();
    
    if (!runtime.IsInitialized()) {
        if (!runtime.Initialize()) {
            return false;
        }
    }

    // Check if module already loaded
    std::string actualName = moduleName.empty() ? ExtractModuleName(filepath) : moduleName;
    
    if (IsModuleLoaded(actualName)) {
        return true;  // Already loaded
    }

    // Load module from file
    auto module = runtime.LoadModule(filepath);
    if (!module) {
        return false;
    }

    // Create instance
    auto instance = module->CreateInstance();
    if (!instance) {
        runtime.UnloadModule(actualName);
        return false;
    }

    // Store instance
    m_ModuleInstances[actualName] = instance;

    // Setup engine bindings
    SetupEngineBindings(actualName);

    // Call init function if it exists
    CallWasmInit(actualName);

    // Initialize performance metrics
    m_PerformanceMetrics[actualName] = {
        actualName,
        0,
        0.0,
        0.0,
        0.0
    };

    return true;
}

bool WasmScriptSystem::CallWasmFunction(const std::string& moduleName, const std::string& functionName) {
    auto it = m_ModuleInstances.find(moduleName);
    if (it == m_ModuleInstances.end()) {
        return false;
    }

    auto instance = it->second;
    instance->Call(functionName);

    // Update performance metrics
    auto& metrics = m_PerformanceMetrics[moduleName];
    metrics.functionCallCount++;
    metrics.totalExecutionTime += instance->GetLastCallDuration();
    metrics.averageCallTime = metrics.totalExecutionTime / metrics.functionCallCount;

    return true;
}

void WasmScriptSystem::BindGameObject(const std::string& moduleName, std::shared_ptr<GameObject> gameObject) {
    auto it = m_ModuleInstances.find(moduleName);
    if (it != m_ModuleInstances.end()) {
        m_BoundGameObjects[moduleName] = gameObject;
        it->second->SetEngineObject(gameObject.get());
    }
}

std::shared_ptr<WasmInstance> WasmScriptSystem::GetModuleInstance(const std::string& moduleName) {
    auto it = m_ModuleInstances.find(moduleName);
    if (it != m_ModuleInstances.end()) {
        return it->second;
    }
    return nullptr;
}

bool WasmScriptSystem::IsModuleLoaded(const std::string& moduleName) const {
    return m_ModuleInstances.find(moduleName) != m_ModuleInstances.end();
}

std::vector<std::string> WasmScriptSystem::GetLoadedModules() const {
    std::vector<std::string> names;
    for (const auto& pair : m_ModuleInstances) {
        names.push_back(pair.first);
    }
    return names;
}

void WasmScriptSystem::UnloadModule(const std::string& moduleName) {
    auto it = m_ModuleInstances.find(moduleName);
    if (it != m_ModuleInstances.end()) {
        m_ModuleInstances.erase(it);
    }

    auto boundIt = m_BoundGameObjects.find(moduleName);
    if (boundIt != m_BoundGameObjects.end()) {
        m_BoundGameObjects.erase(boundIt);
    }

    auto metricsIt = m_PerformanceMetrics.find(moduleName);
    if (metricsIt != m_PerformanceMetrics.end()) {
        m_PerformanceMetrics.erase(metricsIt);
    }

    WasmRuntime::GetInstance().UnloadModule(moduleName);
}

void WasmScriptSystem::CallWasmInit(const std::string& moduleName) {
    auto it = m_ModuleInstances.find(moduleName);
    if (it != m_ModuleInstances.end()) {
        auto instance = it->second;
        auto module = instance->GetModule();
        
        if (module && module->HasExportedFunction("init")) {
            instance->Call("init");
        }
    }
}

void WasmScriptSystem::CallWasmUpdate(const std::string& moduleName, float deltaTime) {
    auto it = m_ModuleInstances.find(moduleName);
    if (it != m_ModuleInstances.end()) {
        auto instance = it->second;
        auto module = instance->GetModule();
        
        if (module && module->HasExportedFunction("update")) {
            instance->Call("update", {WasmValue::F32(deltaTime)});
        }
    }
}

void WasmScriptSystem::CallWasmShutdown(const std::string& moduleName) {
    auto it = m_ModuleInstances.find(moduleName);
    if (it != m_ModuleInstances.end()) {
        auto instance = it->second;
        auto module = instance->GetModule();
        
        if (module && module->HasExportedFunction("shutdown")) {
            instance->Call("shutdown");
        }
    }
}

std::vector<WasmScriptSystem::PerformanceMetrics> WasmScriptSystem::GetPerformanceMetrics() const {
    std::vector<PerformanceMetrics> metrics;
    for (const auto& pair : m_PerformanceMetrics) {
        metrics.push_back(pair.second);
    }
    return metrics;
}

void WasmScriptSystem::RegisterEngineExport(const std::string& name, EngineExportFunction func) {
    m_EngineExports[name] = func;
}

std::string WasmScriptSystem::ExtractModuleName(const std::string& filepath) const {
    size_t lastSlash = filepath.find_last_of("/\\");
    size_t lastDot = filepath.find_last_of('.');
    
    if (lastSlash == std::string::npos) {
        lastSlash = 0;
    } else {
        lastSlash++;
    }

    if (lastDot == std::string::npos || lastDot < lastSlash) {
        return filepath.substr(lastSlash);
    }

    return filepath.substr(lastSlash, lastDot - lastSlash);
}

void WasmScriptSystem::SetupEngineBindings(const std::string& moduleName) {
    auto instance = GetModuleInstance(moduleName);
    if (!instance) {
        return;
    }

    // Register all engine exports as host callbacks
    for (const auto& pair : m_EngineExports) {
        instance->RegisterHostCallback(pair.first, 
            [this, moduleName, callback = pair.second](const std::vector<WasmValue>& args) {
                auto inst = GetModuleInstance(moduleName);
                if (inst) {
                    callback(inst);
                }
                return WasmValue::I32(0);
            }
        );
    }
}

void WasmScriptSystem::UpdateModuleIfChanged(const std::string& moduleName, const std::string& filepath) {
    if (filepath.empty()) {
        return;
    }

    // Check file modification time
    auto lastWriteTime = fs::last_write_time(filepath);
    // Compare with cached time and reload if changed
    // This would require tracking last load times
}

WasmScriptSystem::~WasmScriptSystem() {
    Shutdown();
}

