#pragma once

#include "IScriptSystem.h"
#include "KotlinRuntime.h"
#include <memory>
#include <map>
#include <string>
#include <vector>
#include <any>
#include <chrono>

/**
 * @class KotlinScriptSystem
 * @brief Kotlin language scripting system for the game engine
 * 
 * Provides full Kotlin support with:
 * - JVM bytecode execution
 * - Coroutine-based async operations
 * - Complete OOP with null safety
 * - Standard library integration
 * - Hot-reload for development
 * - Type-safe interop with C++
 * 
 * Kotlin excels at:
 * - Game state management (sealed classes, data classes)
 * - Event systems (higher-order functions)
 * - Behavior trees with coroutines
 * - Scripted logic with strong typing
 * - Rapid prototyping with null safety
 * 
 * File Extension: .kt (Kotlin source) or .class/.jar (compiled)
 * 
 * Usage:
 * ```cpp
 * auto kotlinSystem = std::make_unique<KotlinScriptSystem>();
 * kotlinSystem->Init();
 * 
 * // Load compiled Kotlin class
 * kotlinSystem->RunScript("scripts/game_logic.class");
 * 
 * // Call Kotlin function
 * std::vector<std::any> args = {100, 50.5f};
 * auto result = kotlinSystem->CallFunction("GameLogic.initialize", args);
 * 
 * kotlinSystem->Update(16.67f);  // Update each frame
 * kotlinSystem->Shutdown();
 * ```
 */
class KotlinScriptSystem : public IScriptSystem {
public:
    static KotlinScriptSystem& GetInstance() {
        static KotlinScriptSystem instance;
        return instance;
    }

    // Lifecycle management
    void Init() override;
    void Shutdown() override;
    void Update(float deltaTime) override;

    // Script execution
    bool RunScript(const std::string& filepath) override;
    bool ExecuteString(const std::string& source) override;

    // Language metadata
    ScriptLanguage GetLanguage() const override { return ScriptLanguage::Kotlin; }
    ScriptExecutionMode GetExecutionMode() const override { return ScriptExecutionMode::JustInTime; }
    std::string GetLanguageName() const override { return "Kotlin"; }
    std::string GetFileExtension() const override { return ".kt"; }

    // Type and function management
    void RegisterTypes() override;
    bool HasType(const std::string& typeName) const override;
    std::any CallFunction(const std::string& functionName, 
                         const std::vector<std::any>& args) override;

    // Hot-reload support
    bool SupportsHotReload() const override { return true; }
    void ReloadScript(const std::string& filepath) override;

    // Performance profiling
    uint64_t GetMemoryUsage() const override;
    double GetLastExecutionTime() const override { return lastExecutionTime; }

    // Error handling
    bool HasErrors() const override;
    std::string GetLastError() const override;

    // Kotlin-specific features
    /**
     * Add a classpath for loading Kotlin classes
     * @param path Directory or JAR file path
     */
    void AddClassPath(const std::string& path);

    /**
     * Load a compiled Kotlin class
     * @param className Fully qualified class name (e.g., "gameplay.PlayerController")
     * @return true if successful
     */
    bool LoadClass(const std::string& className);

    /**
     * Unload a Kotlin class
     * @param className Fully qualified class name
     */
    void UnloadClass(const std::string& className);

    /**
     * Get loaded Kotlin runtime
     * @return Pointer to KotlinRuntime instance
     */
    KotlinRuntime* GetRuntime() { return kotlinRuntime.get(); }

    /**
     * Call a static Kotlin function
     * @param className Fully qualified class name
     * @param methodName Method name
     * @param args Method arguments
     * @return Return value
     */
    std::any CallStaticFunction(const std::string& className,
                                const std::string& methodName,
                                const std::vector<std::any>& args = {});

    /**
     * Create a new Kotlin class instance
     * @param className Fully qualified class name
     * @param constructorArgs Constructor arguments
     * @return KotlinRuntime::KotlinObject instance
     */
    KotlinRuntime::KotlinObject CreateInstance(const std::string& className,
                                               const std::vector<std::any>& constructorArgs = {});

    /**
     * Call a method on a Kotlin object
     * @param obj The Kotlin object
     * @param methodName Method name
     * @param args Method arguments
     * @return Return value
     */
    std::any CallMethod(const KotlinRuntime::KotlinObject& obj,
                       const std::string& methodName,
                       const std::vector<std::any>& args = {});

    /**
     * Call a suspend function (coroutine) with callback
     * @param className Fully qualified class name
     * @param methodName Suspend method name
     * @param args Method arguments
     * @param callback Called when coroutine completes
     */
    void CallSuspendFunction(const std::string& className,
                            const std::string& methodName,
                            const std::vector<std::any>& args,
                            std::function<void(std::any)> callback);

    /**
     * Compile Kotlin source file to bytecode
     * Requires kotlinc in system PATH
     * @param sourceFile Path to .kt file
     * @param outputDir Output directory for compiled classes
     * @return true if compilation successful
     */
    bool CompileKotlinFile(const std::string& sourceFile, const std::string& outputDir);

    /**
     * Compile entire directory of Kotlin sources
     * @param sourceDir Directory containing .kt files
     * @param outputDir Output directory for compiled classes
     * @return Number of successfully compiled files
     */
    int CompileKotlinDirectory(const std::string& sourceDir, const std::string& outputDir);

    /**
     * Get list of loaded classes
     * @return Vector of fully qualified class names
     */
    std::vector<std::string> GetLoadedClasses() const;

    /**
     * Check if a class is loaded
     * @param className Fully qualified class name
     * @return true if loaded
     */
    bool IsClassLoaded(const std::string& className) const;

    /**
     * Force garbage collection in JVM
     */
    void RequestGarbageCollection();

private:
    KotlinScriptSystem() : initialized(false), lastExecutionTime(0.0) {}
    ~KotlinScriptSystem() { if (initialized) Shutdown(); }

    std::unique_ptr<KotlinRuntime> kotlinRuntime;
    bool initialized;
    double lastExecutionTime;
    std::chrono::high_resolution_clock::time_point lastUpdateTime;

    std::map<std::string, KotlinRuntime::KotlinObject> activeObjects;
    std::vector<std::string> loadedScripts;
    std::string lastError;

    // Helper methods
    bool CompileAndLoad(const std::string& sourceFile);
    std::string GetCompiledOutputPath(const std::string& sourceFile);
};
