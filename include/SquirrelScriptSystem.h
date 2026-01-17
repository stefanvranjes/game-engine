#pragma once

#include "IScriptSystem.h"
#include <string>
#include <memory>
#include <map>
#include <functional>
#include <vector>

// Forward declare Squirrel types
typedef struct SQVM* HSQUIRRELVM;
typedef struct SQObject HSQOBJECT;

/**
 * @class SquirrelScriptSystem
 * @brief Integrates Squirrel scripting language for C-like gameplay logic
 * 
 * Squirrel is a lightweight, dynamically-typed language with C-like syntax,
 * designed for embedded scripting. It's smaller than Lua and easier to learn for C++ developers.
 * 
 * Features:
 * - Simple C-like syntax familiar to C++ developers
 * - Object-oriented with classes and inheritance
 * - Exception handling and error recovery
 * - Class-based (not prototype-based like JavaScript)
 * - Delegates for function pointers and closures
 * - Small memory footprint (~100KB)
 * - Hash table and array support
 * 
 * Usage:
 * ```cpp
 * SquirrelScriptSystem::GetInstance().Init();
 * SquirrelScriptSystem::GetInstance().RunScript("gameplay_logic.nut");
 * ```
 * 
 * Example Squirrel Script:
 * ```squirrel
 * class Player {
 *     gameObject = null;
 *     speed = 5.0;
 *     health = 100;
 * 
 *     constructor(gameObj) {
 *         gameObject = gameObj;
 *     }
 * 
 *     function update(dt) {
 *         local pos = gameObject.transform.position;
 *         pos.x += speed * dt;
 *         gameObject.transform.position = pos;
 *     }
 * 
 *     function takeDamage(damage) {
 *         health -= damage;
 *         if (health <= 0) {
 *             print("Player died!");
 *         }
 *     }
 * }
 * ```
 */
class SquirrelScriptSystem : public IScriptSystem {
public:
    static SquirrelScriptSystem& GetInstance() {
        static SquirrelScriptSystem instance;
        return instance;
    }

    // IScriptSystem Implementation
    void Init() override;
    void Shutdown() override;
    void Update(float deltaTime) override;
    bool RunScript(const std::string& filepath) override;

    /**
     * Execute Squirrel code from source string
     * @param source The Squirrel source code
     * @return true if successful, false on error
     */
    bool ExecuteString(const std::string& source) override;

    // Language metadata
    ScriptLanguage GetLanguage() const override { return ScriptLanguage::Squirrel; }
    ScriptExecutionMode GetExecutionMode() const override { return ScriptExecutionMode::Interpreted; }
    std::string GetLanguageName() const override { return "Squirrel"; }
    std::string GetFileExtension() const override { return ".nut"; }

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

    /**
     * Register a native C++ function callable from Squirrel
     * @param name Function name in Squirrel scope
     * @param func C++ lambda/function to call
     */
    void RegisterFunction(const std::string& name,
                          std::function<void(HSQUIRRELVM)> func);

    /**
     * Register a native C++ class for use in Squirrel
     * @param className Name of the class in Squirrel
     * @param constructor Function to construct instances
     */
    void RegisterClass(const std::string& className,
                       std::function<void(HSQUIRRELVM)> constructor);

    /**
     * Get the underlying Squirrel VM
     */
    HSQUIRRELVM GetVM() const { return m_VM; }

private:
    SquirrelScriptSystem();
    ~SquirrelScriptSystem();

    HSQUIRRELVM m_VM = nullptr;
    std::string m_LastError;
    double m_LastExecutionTime = 0.0;
    uint64_t m_MemoryUsage = 0;

    std::map<std::string, std::function<void(HSQUIRRELVM)>> m_Functions;

    void RegisterGameObjectTypes();
    void RegisterMathTypes();
    void RegisterPhysicsTypes();
    void RegisterAudioTypes();
    void RegisterAssetTypes();

    void SetError(const std::string& error);
};
