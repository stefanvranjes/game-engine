#pragma once

#include "IScriptSystem.h"
#include <string>
#include <memory>
#include <map>
#include <functional>
#include <vector>

/**
 * @class ScriptLanguageRegistry
 * @brief Central registry for managing multiple scripting language systems
 * 
 * Provides a unified interface to work with multiple scripting languages,
 * handles language detection, system instantiation, and inter-language communication.
 * 
 * Features:
 * - Language auto-detection by file extension
 * - Runtime language switching
 * - Multi-language script execution from single interface
 * - Language system lifecycle management
 * - Performance profiling across all languages
 * - Error aggregation and reporting
 * 
 * Usage:
 * ```cpp
 * auto& registry = ScriptLanguageRegistry::GetInstance();
 * registry.Init();
 * 
 * // Auto-detect language from file
 * registry.ExecuteScript("gameplay/player.lua");      // Lua
 * registry.ExecuteScript("gameplay/npc.wren");        // Wren
 * registry.ExecuteScript("gameplay/logic.js");        // TypeScript/JavaScript
 * registry.ExecuteScript("gameplay/physics.dll");     // Rust
 * 
 * // Call function across any language
 * auto result = registry.CallFunction("update_game", args);
 * ```
 */
class ScriptLanguageRegistry {
public:
    static ScriptLanguageRegistry& GetInstance() {
        static ScriptLanguageRegistry instance;
        return instance;
    }

    // Lifecycle
    /**
     * Initialize all registered script systems
     */
    void Init();

    /**
     * Shutdown all registered script systems
     */
    void Shutdown();

    /**
     * Update all script systems (called once per frame)
     * @param deltaTime Frame delta time
     */
    void Update(float deltaTime);

    // Script execution
    /**
     * Execute a script, auto-detecting language by file extension
     * @param filepath Path to script file
     * @return true if successful, false on error
     */
    bool ExecuteScript(const std::string& filepath);

    /**
     * Execute script with explicit language
     * @param filepath Path to script file
     * @param language Script language to use
     * @return true if successful, false on error
     */
    bool ExecuteScript(const std::string& filepath, ScriptLanguage language);

    /**
     * Execute source code with explicit language
     * @param source Source code string
     * @param language Script language to use
     * @return true if successful, false on error
     */
    bool ExecuteString(const std::string& source, ScriptLanguage language);

    // Language system access
    /**
     * Get a specific script system by language
     * @param language The script language
     * @return Pointer to script system, nullptr if not initialized
     */
    IScriptSystem* GetScriptSystem(ScriptLanguage language) const;

    /**
     * Register a custom script system
     * @param language Language identifier
     * @param system Pointer to script system implementation
     */
    void RegisterScriptSystem(ScriptLanguage language, std::shared_ptr<IScriptSystem> system);

    // Scripting operations
    /**
     * Call a function by name across all registered systems
     * Searches in order: Lua, Wren, Python, C#, TypeScript, Rust, Squirrel, Custom
     * @param functionName Function to call
     * @param args Arguments to pass
     * @return Return value from function
     */
    std::any CallFunction(const std::string& functionName, const std::vector<std::any>& args);

    /**
     * Call a function in a specific language
     * @param language The script language
     * @param functionName Function to call
     * @param args Arguments to pass
     * @return Return value from function
     */
    std::any CallFunction(ScriptLanguage language, const std::string& functionName,
                          const std::vector<std::any>& args);

    // Language detection
    /**
     * Detect script language by file extension
     * @param filepath Path to file
     * @return Detected language, or Custom if unknown
     */
    ScriptLanguage DetectLanguage(const std::string& filepath) const;

    /**
     * Get language from file extension
     * @param extension File extension (e.g., ".lua", ".js")
     * @return Detected language, or Custom if unknown
     */
    ScriptLanguage GetLanguageFromExtension(const std::string& extension) const;

    // Language information
    /**
     * Get all supported languages
     * @return Vector of supported ScriptLanguage values
     */
    std::vector<ScriptLanguage> GetSupportedLanguages() const;

    /**
     * Get language name
     * @param language The script language
     * @return Human-readable language name
     */
    std::string GetLanguageName(ScriptLanguage language) const;

    /**
     * Get file extension for language
     * @param language The script language
     * @return File extension (e.g., ".lua")
     */
    std::string GetFileExtension(ScriptLanguage language) const;

    /**
     * Check if language is initialized and ready
     * @param language The script language
     * @return true if language system is ready, false otherwise
     */
    bool IsLanguageReady(ScriptLanguage language) const;

    // Performance and diagnostics
    /**
     * Get memory usage across all script systems
     * @return Total memory in bytes
     */
    uint64_t GetTotalMemoryUsage() const;

    /**
     * Get execution time of last operation
     * @param language The script language
     * @return Time in seconds
     */
    double GetLastExecutionTime(ScriptLanguage language) const;

    /**
     * Check if any language system has errors
     * @return true if any system has errors, false otherwise
     */
    bool HasErrors() const;

    /**
     * Get all error messages from all systems
     * @return Error message map (language -> error)
     */
    std::map<std::string, std::string> GetAllErrors() const;

    /**
     * Clear all error states
     */
    void ClearErrors();

    // Hot-reload
    /**
     * Reload a script from disk
     * @param filepath Path to script file
     * @return true if successful, false on error
     */
    bool ReloadScript(const std::string& filepath);

    /**
     * Check if language supports hot-reload
     * @param language The script language
     * @return true if hot-reload is supported, false otherwise
     */
    bool SupportsHotReload(ScriptLanguage language) const;

    // Callbacks
    /**
     * Set error callback to be invoked when a script error occurs
     * @param callback Function to call with error information
     */
    using ErrorCallback = std::function<void(ScriptLanguage, const std::string&)>;
    void SetErrorCallback(ErrorCallback callback);

    /**
     * Set success callback to be invoked when a script executes successfully
     * @param callback Function to call with language information
     */
    using SuccessCallback = std::function<void(ScriptLanguage, const std::string&)>;
    void SetSuccessCallback(SuccessCallback callback);

private:
    ScriptLanguageRegistry();
    ~ScriptLanguageRegistry();

    std::map<ScriptLanguage, std::shared_ptr<IScriptSystem>> m_Systems;
    std::map<std::string, ScriptLanguage> m_ExtensionMap;

    ErrorCallback m_ErrorCallback;
    SuccessCallback m_SuccessCallback;

    void InitializeExtensionMap();
    void RegisterDefaultSystems();
};
