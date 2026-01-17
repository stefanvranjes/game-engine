#pragma once

#include "IScriptSystem.h"
#include <string>
#include <memory>
#include <map>
#include <functional>
#include <vector>

// Forward declare QuickJS types
typedef struct JSRuntime JSRuntime;
typedef struct JSContext JSContext;
typedef struct JSValue JSValue;

// Forward declarations
class GameObject;

/**
 * @class TypeScriptScriptSystem
 * @brief Integrates TypeScript/JavaScript scripting via QuickJS engine
 * 
 * QuickJS provides a lightweight, fast JavaScript engine ideal for gameplay scripting
 * with minimal memory overhead. Supports ES2020 syntax with async/await capabilities.
 * 
 * Features:
 * - Full ES2020 JavaScript/TypeScript support (transpiled to JS)
 * - Async/await for coroutine-like gameplay sequences
 * - Module system with import/export
 * - Fast startup and low memory footprint
 * - Native bindings for game engine types
 * 
 * Usage:
 * ```cpp
 * TypeScriptScriptSystem::GetInstance().Init();
 * TypeScriptScriptSystem::GetInstance().RunScript("gameplay_logic.js");
 * ```
 * 
 * Example TypeScript Script:
 * ```typescript
 * export class Player {
 *     private gameObject: GameObject;
 *     private speed: number = 5.0;
 * 
 *     constructor(gameObject: GameObject) {
 *         this.gameObject = gameObject;
 *     }
 * 
 *     update(dt: number): void {
 *         const pos = this.gameObject.transform.position;
 *         pos.x += this.speed * dt;
 *         this.gameObject.transform.position = pos;
 *     }
 * 
 *     async onCollisionAsync(other: GameObject): Promise<void> {
 *         console.log(`Collided with: ${other.name}`);
 *         await this.playSoundAsync();
 *     }
 * }
 * ```
 */
class TypeScriptScriptSystem : public IScriptSystem {
public:
    using MessageHandler = std::function<void(const std::string&)>;

    static TypeScriptScriptSystem& GetInstance() {
        static TypeScriptScriptSystem instance;
        return instance;
    }

    // IScriptSystem Implementation
    void Init() override;
    void Shutdown() override;
    void Update(float deltaTime) override;
    bool RunScript(const std::string& filepath) override;

    /**
     * Execute JavaScript from source code string
     * @param source The JavaScript source code to execute
     * @return true if successful, false on error
     */
    bool ExecuteString(const std::string& source) override;

    // Language metadata
    ScriptLanguage GetLanguage() const override { return ScriptLanguage::TypeScript; }
    ScriptExecutionMode GetExecutionMode() const override { return ScriptExecutionMode::JustInTime; }
    std::string GetLanguageName() const override { return "TypeScript/JavaScript (QuickJS)"; }
    std::string GetFileExtension() const override { return ".js"; }

    // Type registration
    void RegisterTypes() override;
    bool HasType(const std::string& typeName) const override;
    std::any CallFunction(const std::string& functionName,
                          const std::vector<std::any>& args) override;

    // Hot-reload support
    bool SupportsHotReload() const override { return true; }
    void ReloadScript(const std::string& filepath) override;

    // Performance metrics
    uint64_t GetMemoryUsage() const override;
    double GetLastExecutionTime() const override;

    // Error handling
    bool HasErrors() const override;
    std::string GetLastError() const override;

    // Advanced API
    /**
     * Register a native C++ function callable from JavaScript
     * @param name Function name in JavaScript scope
     * @param func C++ lambda/function to call
     */
    void RegisterFunction(const std::string& name,
                          std::function<JSValue(JSContext*, const std::vector<JSValue>&)> func);

    /**
     * Register a native C++ class for use in JavaScript
     * @param className Name of the class in JavaScript
     * @param constructor Function to construct instances
     */
    void RegisterClass(const std::string& className,
                       std::function<void*(JSContext*)> constructor);

    /**
     * Get the underlying QuickJS context
     */
    JSContext* GetContext() const { return m_Context; }

    /**
     * Get the underlying QuickJS runtime
     */
    JSRuntime* GetRuntime() const { return m_Runtime; }

private:
    TypeScriptScriptSystem();
    ~TypeScriptScriptSystem();

    JSRuntime* m_Runtime = nullptr;
    JSContext* m_Context = nullptr;

    std::string m_LastError;
    double m_LastExecutionTime = 0.0;
    uint64_t m_MemoryUsage = 0;

    std::map<std::string, std::function<JSValue(JSContext*, const std::vector<JSValue>&)>> m_Functions;

    void RegisterGameObjectTypes();
    void RegisterMathTypes();
    void RegisterPhysicsTypes();
    void RegisterAudioTypes();
    void RegisterAssetTypes();

    void SetError(const std::string& error);
};
