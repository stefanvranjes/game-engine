#pragma once

#include "IScriptSystem.h"
#include <string>
#include <memory>
#include <map>
#include <functional>
#include <vector>

extern "C" {
#include <wren.h>
}

// Forward declarations of game engine types
class GameObject;
class Transform;

/**
 * @class WrenScriptSystem
 * @brief Integrates Wren scripting language for gameplay logic
 * 
 * Wren is a lightweight, dynamically-typed scripting language ideal for
 * gameplay mechanics, AI behavior, event handling, and rapid prototyping.
 * 
 * Features:
 * - Hot-reloading of scripts
 * - Native bindings for game engine types (Vec3, Transform, GameObject, Physics)
 * - Fiber support for coroutines and async gameplay sequences
 * - Object-oriented design via classes and inheritance
 * 
 * Usage:
 * ```cpp
 * WrenScriptSystem::GetInstance().Init();
 * WrenScriptSystem::GetInstance().RunScript("gameplay_logic.wren");
 * ```
 * 
 * Example Wren Script:
 * ```wren
 * class Player {
 *     construct new(gameObject) {
 *         _gameObject = gameObject
 *         _speed = 5.0
 *     }
 *     
 *     update(dt) {
 *         var pos = _gameObject.transform.position
 *         pos.x = pos.x + _speed * dt
 *         _gameObject.transform.position = pos
 *     }
 *     
 *     onCollision(other) {
 *         System.print("Collided with: %(other.name)")
 *     }
 * }
 * ```
 */
class WrenScriptSystem : public IScriptSystem {
public:
    using WrenForeignFunction = std::function<void(WrenVM*)>;
    using MessageHandler = std::function<void(const std::string&)>;

    static WrenScriptSystem& GetInstance() {
        static WrenScriptSystem instance;
        return instance;
    }

    // IScriptSystem Implementation
    void Init() override;
    void Shutdown() override;
    void Update(float deltaTime) override;
    bool RunScript(const std::string& filepath) override;

    /**
     * Execute a Wren script from source code string
     * @param source The Wren source code to execute
     * @return true if successful, false on error
     */
    bool ExecuteString(const std::string& source) override;

    // Language metadata
    ScriptLanguage GetLanguage() const override { return ScriptLanguage::Wren; }
    ScriptExecutionMode GetExecutionMode() const override { return ScriptExecutionMode::Interpreted; }
    std::string GetLanguageName() const override { return "Wren"; }
    std::string GetFileExtension() const override { return ".wren"; }

    /**
     * Call a global Wren function with arguments
     * @param functionName Name of the global function to call
     * @param args Wren values to pass as arguments (space-separated string representation)
     * @return true if successful, false on error
     */
    bool CallFunction(const std::string& functionName, const std::string& args = "");

    /**
     * Call a method on a Wren class instance
     * @param instanceName Global variable name of the instance
     * @param methodName Method name to call
     * @param args Method arguments as string representation
     * @return true if successful, false on error
     */
    bool CallMethod(const std::string& instanceName, const std::string& methodName, 
                   const std::string& args = "");

    /**
     * Set a global variable in Wren
     * @param varName Variable name
     * @param value Value as string (will be evaluated as Wren code)
     */
    void SetGlobalVariable(const std::string& varName, const std::string& value);

    /**
     * Get a global variable as a string
     * @param varName Variable name
     * @return String representation of the variable value
     */
    std::string GetGlobalVariable(const std::string& varName);

    /**
     * Register a native C++ function callable from Wren
     * @param className Class name in Wren
     * @param methodName Method name
     * @param numParams Number of parameters (including implicit 'this')
     * @param function C++ function implementation
     */
    void RegisterNativeMethod(const std::string& className, 
                            const std::string& methodName,
                            int numParams,
                            WrenForeignFunction function);

    /**
     * Register a static native function callable from Wren
     * @param moduleName Module name
     * @param className Class name
     * @param methodName Static method name
     * @param numParams Number of parameters
     * @param function C++ function implementation
     */
    void RegisterStaticMethod(const std::string& moduleName,
                            const std::string& className,
                            const std::string& methodName,
                            int numParams,
                            WrenForeignFunction function);

    /**
     * Subscribe to Wren print output
     * @param handler Function called with print messages
     */
    void SetPrintHandler(MessageHandler handler) { m_PrintHandler = handler; }

    /**
     * Subscribe to Wren error output
     * @param handler Function called with error messages
     */
    void SetErrorHandler(MessageHandler handler) { m_ErrorHandler = handler; }

    /**
     * Get the raw Wren VM pointer (for advanced usage)
     * @return WrenVM pointer
     */
    WrenVM* GetVM() const { return m_VM; }

    /**
     * Reload all loaded scripts (for hot-reload support)
     */
    void ReloadAll();

    /**
     * Check if a global function exists in Wren
     * @param functionName Function name
     * @return true if function exists, false otherwise
     */
    bool HasFunction(const std::string& functionName) const override;

    /**
     * Check if a global variable exists in Wren
     * @param variableName Variable name
     * @return true if variable exists, false otherwise
     */
    bool HasVariable(const std::string& variableName);
    
    // Internal error handler (Static for use in C callbacks)
    static void HandleError(const std::string& error);

    WrenScriptSystem();
    ~WrenScriptSystem();

private:

    WrenVM* m_VM = nullptr;
    std::vector<std::string> m_LoadedScripts;
    MessageHandler m_PrintHandler;
    MessageHandler m_ErrorHandler;

    // Register all built-in bindings
    void RegisterGameObjectBindings();
    void RegisterTransformBindings();
    void RegisterVec3Bindings();
    void RegisterPhysicsBindings();
    void RegisterAudioBindings();
    void RegisterParticleBindings();
    void RegisterTimeBindings();
    void RegisterInputBindings();
    void RegisterUtilityBindings();

    // Helper methods for binding
    static void BindClass(const std::string& className);
    static void BindForeignMethod(const std::string& className,
                                 const std::string& methodName,
                                 bool isStatic,
                                 WrenForeignFunction function);

    
    void CallMessageHandler(const std::string& msg, bool isError) {
        if (isError && m_ErrorHandler) m_ErrorHandler(msg);
        else if (!isError && m_PrintHandler) m_PrintHandler(msg);
    }
};

/**
 * Helper class for passing game engine objects to Wren
 */
class WrenBinding {
public:
    /**
     * Store a shared pointer in Wren foreign data
     */
    template<typename T>
    static void SetForeignData(WrenVM* vm, std::shared_ptr<T> ptr) {
        auto heap_ptr = new std::shared_ptr<T>(ptr);
        wrenSetSlotNewForeign(vm, 0, 0, sizeof(std::shared_ptr<T>));
        std::shared_ptr<T>* foreign = 
            static_cast<std::shared_ptr<T>*>(wrenGetSlotForeign(vm, 0));
        *foreign = ptr;
    }

    /**
     * Retrieve a shared pointer from Wren foreign data
     */
    template<typename T>
    static std::shared_ptr<T> GetForeignData(WrenVM* vm, int slot) {
        std::shared_ptr<T>* foreign = 
            static_cast<std::shared_ptr<T>*>(wrenGetSlotForeign(vm, slot));
        return foreign ? *foreign : nullptr;
    }
};
