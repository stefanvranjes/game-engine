#pragma once

#include "IScriptSystem.h"
#include <string>
#include <memory>
#include <map>
#include <vector>
#include <functional>
#include <any>
#include <chrono>
#include <filesystem>

/**
 * @class MunScriptSystem
 * @brief Mun language scripting system with compiled hot-reload support
 * 
 * Mun is a compiled scripting language specifically designed for hot-reload gaming scenarios.
 * It provides the performance of compiled languages with the iteration speed of interpreted ones.
 * 
 * Key Features:
 * - **Compiled Hot-Reload**: Scripts compiled to native code, reloadable without shutdown
 * - **Static Typing**: Strong type system prevents runtime errors
 * - **Performance**: Near C++ performance with no garbage collection overhead
 * - **Safety**: Ownership system similar to Rust for memory safety
 * - **First-Class Struct Support**: Perfect for game data structures
 * - **Module System**: Organize code into reusable modules
 * 
 * Mun Strengths:
 * - Game behavior scripts (AI, movement, combat)
 * - Component systems with hot-reload
 * - Server-side game logic
 * - Physics interactions
 * - Gameplay balancing (real-time parameter tweaking)
 * 
 * File Extension: .mun (Mun source) → compiled to .dll/.so/.dylib
 * 
 * Usage Example:
 * ```cpp
 * auto munSystem = std::make_unique<MunScriptSystem>();
 * munSystem->Init();
 * 
 * // Load Mun script
 * munSystem->LoadScript("scripts/gameplay.mun");
 * 
 * // Call functions
 * std::vector<std::any> args = {100, 50.5f};
 * auto result = munSystem->CallFunction("update_player", args);
 * 
 * // Hot-reload is automatic via file watching
 * munSystem->Update(16.67f);
 * munSystem->Shutdown();
 * ```
 * 
 * Example Mun Script (scripts/gameplay.mun):
 * ```mun
 * pub fn update_player(health: f32, damage: f32) -> f32 {
 *     let new_health = health - damage;
 *     if new_health < 0.0 {
 *         0.0
 *     } else {
 *         new_health
 *     }
 * }
 * 
 * pub struct Player {
 *     name: String,
 *     health: f32,
 *     speed: f32,
 * }
 * 
 * impl Player {
 *     pub fn take_damage(self: &mut Self, damage: f32) {
 *         self.health -= damage;
 *         if self.health < 0.0 {
 *             self.health = 0.0;
 *         }
 *     }
 * }
 * ```
 */
class MunScriptSystem : public IScriptSystem {
public:
    /**
     * Mun compilation options
     */
    struct CompilationOptions {
        bool optimize = true;           // Release mode optimization
        std::string targetDir = "mun-target";  // Mun compilation target directory
        bool verbose = false;           // Verbose compiler output
        bool emitMetadata = true;       // Emit type metadata for runtime reflection
    };

    static MunScriptSystem& GetInstance() {
        static MunScriptSystem instance;
        return instance;
    }

    // ============ IScriptSystem Implementation ============

    void Init() override;
    void Shutdown() override;
    void Update(float deltaTime) override;

    bool RunScript(const std::string& filepath) override;
    bool ExecuteString(const std::string& source) override { 
        // Mun doesn't support dynamic string execution; source must be in files
        SetError("Mun requires script files, ExecuteString not supported");
        return false; 
    }

    // Language metadata
    ScriptLanguage GetLanguage() const override { return ScriptLanguage::Mun; }
    ScriptExecutionMode GetExecutionMode() const override { return ScriptExecutionMode::NativeCompiled; }
    std::string GetLanguageName() const override { return "Mun (Compiled Hot-Reload)"; }
    std::string GetFileExtension() const override { return ".mun"; }

    // Type registration and interop
    void RegisterTypes() override;
    bool HasType(const std::string& typeName) const override;
    std::any CallFunction(const std::string& functionName,
                          const std::vector<std::any>& args) override;

    // ============ Mun-Specific Functionality ============

    /**
     * Load and compile a Mun script file
     * @param filepath Path to .mun source file
     * @param options Compilation options
     * @return true if successful, false on compilation error
     */
    bool LoadScript(const std::string& filepath, const CompilationOptions& options = CompilationOptions());

    /**
     * Compile Mun script without loading
     * Used for validation or pre-compilation
     * @param filepath Path to .mun source file
     * @param outputDir Directory for compiled output
     * @return true if compilation successful
     */
    bool CompileScript(const std::string& filepath, const std::string& outputDir = "");

    /**
     * Get the compiled library path for a script
     */
    std::string GetCompiledLibraryPath(const std::string& scriptName) const;

    /**
     * Manually trigger recompilation and hot-reload of a script
     * Called automatically during Update() if file change detected
     */
    bool RecompileAndReload(const std::string& scriptName);

    /**
     * Set compilation options for all future compilations
     */
    void SetCompilationOptions(const CompilationOptions& options) {
        m_compilationOptions = options;
    }

    /**
     * Get current compilation options
     */
    const CompilationOptions& GetCompilationOptions() const {
        return m_compilationOptions;
    }

    /**
     * Enable/disable automatic hot-reload on file changes
     */
    void SetAutoHotReload(bool enabled) { m_autoHotReload = enabled; }
    bool IsAutoHotReloadEnabled() const { return m_autoHotReload; }

    /**
     * Set callback for when a script is reloaded
     */
    using ReloadCallback = std::function<void(const std::string& scriptName)>;
    void SetOnScriptReloaded(const ReloadCallback& callback) {
        m_onScriptReloaded = callback;
    }

    // ============ Hot-Reload Support ============

    bool SupportsHotReload() const override { return true; }
    void ReloadScript(const std::string& filepath) override {
        RecompileAndReload(filepath);
    }

    /**
     * Watch a Mun script file for changes
     * Automatically recompiles and reloads when changed
     */
    void WatchScriptFile(const std::string& filepath);

    /**
     * Stop watching a script file
     */
    void UnwatchScriptFile(const std::string& filepath);

    /**
     * Watch entire directory for .mun files
     */
    void WatchScriptDirectory(const std::string& dirpath);

    /**
     * Get list of currently loaded scripts
     */
    std::vector<std::string> GetLoadedScripts() const;

    /**
     * Get list of watched files
     */
    std::vector<std::string> GetWatchedFiles() const;

    // ============ Performance & Debugging ============

    uint64_t GetMemoryUsage() const override;
    double GetLastExecutionTime() const override;

    // Error handling
    bool HasErrors() const override { return !m_lastError.empty(); }
    std::string GetLastError() const override { return m_lastError; }

    /**
     * Get compilation statistics
     */
    struct CompilationStats {
        uint32_t totalCompiles = 0;
        uint32_t successfulCompiles = 0;
        uint32_t failedCompiles = 0;
        uint32_t totalReloads = 0;
        double totalCompileTime = 0.0;  // seconds
        double lastCompileTime = 0.0;   // seconds
    };

    const CompilationStats& GetCompilationStats() const { return m_stats; }
    void ResetStats() { m_stats = CompilationStats(); }

    /**
     * Check if Mun compiler is available in system PATH
     */
    static bool IsMunCompilerAvailable();

    /**
     * Get Mun compiler version
     */
    static std::string GetMunCompilerVersion();

private:
    /**
     * Internal structure to track loaded libraries and metadata
     */
    struct LoadedScript {
        std::string sourceFile;
        std::string compiledLib;
        void* libHandle = nullptr;
        std::chrono::file_clock::time_point lastModified;
        bool needsReload = false;
    };

    /**
     * Compile .mun source to native library
     * Returns path to compiled library on success, empty string on failure
     */
    std::string CompileMunSource(const std::string& sourceFile);

    /**
     * Load compiled library and cache function pointers
     */
    bool LoadCompiledLibrary(const std::string& scriptName, const std::string& libPath);

    /**
     * Unload previously loaded library
     */
    void UnloadLibrary(const std::string& scriptName);

    /**
     * Check if any watched files have been modified
     */
    void CheckForChanges();

    /**
     * Set error message
     */
    void SetError(const std::string& error) {
        m_lastError = error;
    }

    // Member variables
    std::map<std::string, LoadedScript> m_loadedScripts;
    std::map<std::string, std::function<void*()>> m_functionCache;  // Function pointers
    std::vector<std::string> m_watchedFiles;
    CompilationOptions m_compilationOptions;
    ReloadCallback m_onScriptReloaded;
    std::string m_lastError;
    bool m_autoHotReload = true;
    bool m_initialized = false;
    CompilationStats m_stats;
    std::chrono::high_resolution_clock::time_point m_lastUpdateTime;
};

// Register Mun language in the ScriptLanguage enum update
// (Update IScriptSystem.h ScriptLanguage enum to include: Mun)
