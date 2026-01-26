#pragma once

#include "IScriptSystem.h"
#include <string>
#include <memory>
#include <map>
#include <functional>
#include <vector>
#include <unordered_map>

// Forward declarations of AngelScript engine
class asIScriptEngine;
class asIScriptContext;
class asIScriptObject;
class asIScriptModule;

// Forward declarations of game engine types
class GameObject;
class Transform;

/**
 * @class AngelScriptSystem
 * @brief Lightweight AngelScript integration for gameplay scripting
 * 
 * AngelScript is a flexible, lightweight scripting language designed specifically
 * for use in games and other performance-critical applications.
 * 
 * Key Features:
 * - C++-like syntax familiar to game developers
 * - Static typing with type inference
 * - Object-oriented and procedural programming
 * - Low memory footprint (~2-3MB for full engine)
 * - Excellent performance for game logic and AI
 * - Simple, intuitive C++ binding system
 * - Hot-reload support for rapid development
 * 
 * Performance Characteristics:
 * - Startup time: ~5-10ms per script
 * - Execution: 2-5x slower than native C++, comparable to Lua
 * - Memory: ~2-3MB engine + ~100KB per module
 * - Best use: Game logic, AI behavior, event handlers, game mechanics
 * 
 * Usage Example:
 * ```cpp
 * auto& angel = AngelScriptSystem::GetInstance();
 * angel.Init();
 * angel.RunScript("scripts/player.as");
 * angel.CallFunction("UpdatePlayer", {player_object, delta_time});
 * ```
 * 
 * AngelScript Code Example:
 * ```angelscript
 * class Player {
 *     float speed;
 *     Vec3 position;
 *     
 *     Player(float spd) {
 *         speed = spd;
 *     }
 *     
 *     void Update(float dt) {
 *         position.x += speed * dt;
 *     }
 * }
 * 
 * void OnPlayerSpawn(Player@ player) {
 *     player.speed = 10.0f;
 * }
 * ```
 */
class AngelScriptSystem : public IScriptSystem {
public:
    using AngelScriptFunction = std::function<void(asIScriptContext*)>;
    using MessageHandler = std::function<void(const std::string&)>;

    static AngelScriptSystem& GetInstance() {
        static AngelScriptSystem instance;
        return instance;
    }

    // IScriptSystem Implementation
    void Init() override;
    void Shutdown() override;
    void Update(float deltaTime) override;
    bool RunScript(const std::string& filepath) override;
    bool ExecuteString(const std::string& source) override;

    // Language metadata
    ScriptLanguage GetLanguage() const override { return ScriptLanguage::AngelScript; }
    ScriptExecutionMode GetExecutionMode() const override { return ScriptExecutionMode::Bytecode; }
    std::string GetLanguageName() const override { return "AngelScript"; }
    std::string GetFileExtension() const override { return ".as"; }

    // Type registration
    void RegisterTypes() override;
    bool HasType(const std::string& typeName) const override;

    /**
     * Call a global AngelScript function with arguments
     * @param functionName Name of the function to call
     * @param args Vector of std::any arguments to pass
     * @return Result as std::any, empty if no return value
     */
    std::any CallFunction(const std::string& functionName, 
                         const std::vector<std::any>& args) override;

    /**
     * Call a method on a script object
     * @param objectName Global variable name of the object
     * @param methodName Method to call
     * @param args Arguments to pass
     * @return Result as std::any
     */
    std::any CallMethod(const std::string& objectName,
                       const std::string& methodName,
                       const std::vector<std::any>& args);

    /**
     * Set a global variable
     * @param varName Variable name
     * @param value Value (for simple types: int, float, bool, string)
     */
    void SetGlobalVariable(const std::string& varName, const std::any& value);

    /**
     * Get a global variable as a string representation
     * @param varName Variable name
     * @return String representation of the value
     */
    std::string GetGlobalVariable(const std::string& varName) const;

    /**
     * Get the AngelScript engine pointer for advanced operations
     * @return Raw pointer to asIScriptEngine
     */
    asIScriptEngine* GetEngine() const { return m_Engine; }

    /**
     * Get the current execution context
     * @return Raw pointer to asIScriptContext
     */
    asIScriptContext* GetContext() const { return m_Context; }

    /**
     * Create a new module (namespace) for organizing scripts
     * @param moduleName Name of the module
     * @return true if successful
     */
    bool CreateModule(const std::string& moduleName);

    /**
     * Build (compile) a module
     * @param moduleName Module to build
     * @return true if successful
     */
    bool BuildModule(const std::string& moduleName);

    /**
     * Discard a module to free memory
     * @param moduleName Module to discard
     */
    void DiscardModule(const std::string& moduleName);

    /**
     * Get active module
     * @return Pointer to current module
     */
    asIScriptModule* GetModule(const std::string& moduleName) const;

    /**
     * Set the active module for function calls
     * @param moduleName Module to activate
     */
    void SetActiveModule(const std::string& moduleName);

    /**
     * Check if a function exists
     * @param functionName Function to check
     * @return true if function exists in active module
     */
    bool HasFunction(const std::string& functionName) const;

    /**
     * Hot-reload support
     */
    bool SupportsHotReload() const override { return true; }
    void ReloadScript(const std::string& filepath) override;

    /**
     * Performance profiling
     */
    uint64_t GetMemoryUsage() const override;
    double GetLastExecutionTime() const override { return m_LastExecutionTime; }

    /**
     * Error handling
     */
    bool HasErrors() const override { return !m_LastError.empty(); }
    std::string GetLastError() const override { return m_LastError; }

    /**
     * Subscribe to print output from scripts
     * @param handler Callback for print messages
     */
    void SetPrintHandler(MessageHandler handler) { m_PrintHandler = handler; }

    /**
     * Subscribe to error messages
     * @param handler Callback for error messages
     */
    void SetErrorHandler(MessageHandler handler) { m_ErrorHandler = handler; }

    /**
     * Enable/disable optimization during compilation
     * @param enabled true to enable optimizations
     */
    void SetOptimizationEnabled(bool enabled) { m_OptimizeScripts = enabled; }

    /**
     * Enable/disable debugging
     * @param enabled true to enable debug mode
     */
    void SetDebugEnabled(bool enabled) { m_DebugMode = enabled; }

    /**
     * Get script compilation statistics
     */
    struct CompileStats {
        uint32_t totalFunctions = 0;
        uint32_t totalClasses = 0;
        uint32_t totalModules = 0;
        uint64_t memoryUsed = 0;
    };
    CompileStats GetCompileStats() const;

    /**
     * Force garbage collection
     */
    void ForceGarbageCollection();

    /**
     * Clear all loaded scripts and reset engine state
     */
    void ClearState();

private:
    AngelScriptSystem();
    ~AngelScriptSystem();

    // Prevent copy/move
    AngelScriptSystem(const AngelScriptSystem&) = delete;
    AngelScriptSystem& operator=(const AngelScriptSystem&) = delete;

    // Helper functions
    void SetupCallbacks();
    void RegisterEngineTypes();
    void RegisterGameObjectTypes();
    void RegisterMathTypes();
    void RegisterPhysicsTypes();
    void LogMessage(const std::string& message, bool isError = false);
    int ExecuteFunction(asIScriptContext* ctx, const std::string& funcName);

    // Members
    asIScriptEngine* m_Engine = nullptr;
    asIScriptContext* m_Context = nullptr;
    asIScriptModule* m_ActiveModule = nullptr;

    std::unordered_map<std::string, asIScriptModule*> m_Modules;
    std::unordered_map<std::string, std::string> m_ScriptPaths;

    std::string m_LastError;
    double m_LastExecutionTime = 0.0;

    MessageHandler m_PrintHandler;
    MessageHandler m_ErrorHandler;

    bool m_OptimizeScripts = true;
    bool m_DebugMode = false;
    bool m_Initialized = false;

    // Statistics
    uint32_t m_TotalFunctionCalls = 0;
    uint64_t m_TotalExecutionTime = 0;
};
