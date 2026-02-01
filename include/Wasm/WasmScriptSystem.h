#pragma once

#include "IScriptSystem.h"
#include <string>
#include <memory>
#include <unordered_map>
#include <functional>

// Forward declarations
class WasmRuntime;
class WasmInstance;
class GameObject;

/**
 * @class WasmScriptSystem
 * @brief WebAssembly script system integration
 * 
 * Implements IScriptSystem for WASM modules, allowing them to be used
 * as game scripts alongside Lua, Python, and other language runtimes.
 */
class WasmScriptSystem : public IScriptSystem {
public:
    WasmScriptSystem() = default;
    ~WasmScriptSystem();

    static WasmScriptSystem& GetInstance() {
        static WasmScriptSystem instance;
        return instance;
    }

    // IScriptSystem Implementation
    void Init() override;
    void Shutdown() override;
    void Update(float deltaTime) override;
    bool RunScript(const std::string& filepath) override;
    bool ExecuteString(const std::string& source) override;

    // Language metadata
    ScriptLanguage GetLanguage() const override { return ScriptLanguage::WebAssembly; }
    ScriptExecutionMode GetExecutionMode() const override { return ScriptExecutionMode::Compiled; }
    std::string GetLanguageName() const override { return "WebAssembly"; }
    std::string GetFileExtension() const override { return ".wasm"; }

    /**
     * Load and instantiate WASM module as a component script
     * @param filepath Path to .wasm file
     * @param moduleName Unique module identifier (defaults to filename)
     * @return true if successful
     */
    bool LoadWasmModule(const std::string& filepath, const std::string& moduleName = "");

    /**
     * Call WASM function directly
     * @param moduleName Name of loaded WASM module
     * @param functionName Name of exported function
     * @param args Optional arguments (must match function signature)
     * @return true if successful
     */
    bool CallWasmFunction(const std::string& moduleName, const std::string& functionName);

    /**
     * Bind a GameObject to a WASM module instance
     * Makes engine functionality accessible to the WASM script
     * @param moduleName Name of WASM module
     * @param gameObject GameObject to bind
     */
    void BindGameObject(const std::string& moduleName, std::shared_ptr<GameObject> gameObject);

    /**
     * Get WASM instance for a module
     * Allows direct access for advanced use cases
     */
    std::shared_ptr<WasmInstance> GetModuleInstance(const std::string& moduleName);

    /**
     * Check if WASM module is loaded
     */
    bool IsModuleLoaded(const std::string& moduleName) const;

    /**
     * List all loaded WASM modules
     */
    std::vector<std::string> GetLoadedModules() const;

    /**
     * Unload WASM module and cleanup
     * @param moduleName Name of module to unload
     */
    void UnloadModule(const std::string& moduleName);

    /**
     * Call WASM lifecycle hooks if they exist
     * Modules can export these optional functions
     */
    void CallWasmInit(const std::string& moduleName);
    void CallWasmUpdate(const std::string& moduleName, float deltaTime);
    void CallWasmShutdown(const std::string& moduleName);

    /**
     * Enable hot-reload for WASM modules
     * Watches for file changes and reloads automatically
     */
    void EnableHotReload(bool enabled) { m_HotReloadEnabled = enabled; }

    /**
     * Get runtime instance for advanced operations
     */
    WasmRuntime& GetRuntime() const;

    /**
     * Performance monitoring
     */
    struct PerformanceMetrics {
        std::string moduleName;
        uint32_t functionCallCount;
        double totalExecutionTime;  // Milliseconds
        double averageCallTime;
        double peakMemoryUsage;
    };

    std::vector<PerformanceMetrics> GetPerformanceMetrics() const;

    /**
     * Export engine function to all WASM modules
     * Allows modules to call back into the engine
     */
    using EngineExportFunction = std::function<void(std::shared_ptr<WasmInstance>)>;
    void RegisterEngineExport(const std::string& name, EngineExportFunction func);

    /**
     * Set default memory allocation limit for new modules
     */
    void SetDefaultMemoryLimit(uint32_t sizeInMB) { m_DefaultMemoryLimitMB = sizeInMB; }

private:
    // Prevent copying
    WasmScriptSystem(const WasmScriptSystem&) = delete;
    WasmScriptSystem& operator=(const WasmScriptSystem&) = delete;

    std::unordered_map<std::string, std::shared_ptr<WasmInstance>> m_ModuleInstances;
    std::unordered_map<std::string, std::weak_ptr<GameObject>> m_BoundGameObjects;
    std::unordered_map<std::string, EngineExportFunction> m_EngineExports;
    std::unordered_map<std::string, PerformanceMetrics> m_PerformanceMetrics;

    bool m_HotReloadEnabled = false;
    uint32_t m_DefaultMemoryLimitMB = 256;
    double m_AccumulatedTime = 0.0;

    std::string ExtractModuleName(const std::string& filepath) const;
    void SetupEngineBindings(const std::string& moduleName);
    void UpdateModuleIfChanged(const std::string& moduleName, const std::string& filepath);
};

