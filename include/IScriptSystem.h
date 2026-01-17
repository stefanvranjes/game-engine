#pragma once

#include <string>
#include <memory>
#include <vector>
#include <map>
#include <functional>
#include <any>

/**
 * @enum ScriptLanguage
 * @brief Supported scripting languages in the engine
 */
enum class ScriptLanguage {
    Lua,        // Lua 5.4 - General purpose
    Wren,       // Wren - Lightweight OOP
    Python,     // Python 3.x - Data science & AI
    CSharp,     // C# - .NET integration (requires Mono)
    Custom,     // Custom bytecode VM
    TypeScript, // TypeScript/JavaScript - V8/QuickJS
    Rust,       // Rust - Safe, compiled scripting
    Squirrel    // Squirrel - C-like syntax for games
};

/**
 * @enum ScriptExecutionMode
 * @brief How scripts are executed
 */
enum class ScriptExecutionMode {
    Interpreted,    // Script interpreted at runtime (Lua, Python, Wren)
    JustInTime,     // Script compiled on first run (TypeScript, Rust)
    Bytecode,       // Pre-compiled bytecode execution (Custom VM)
    NativeCompiled  // Native compiled library (Rust DLLs)
};

/**
 * @class IScriptSystem
 * @brief Abstract base class for all script language implementations
 * 
 * Each language implementation (Lua, Python, Wren, TypeScript, Rust, Squirrel, etc.)
 * derives from this interface to provide consistent scripting capabilities.
 */
class IScriptSystem {
public:
    virtual ~IScriptSystem() = default;

    // Core lifecycle
    virtual void Init() = 0;
    virtual void Shutdown() = 0;
    virtual void Update(float deltaTime) = 0;

    // Script execution
    virtual bool RunScript(const std::string& filepath) = 0;
    virtual bool ExecuteString(const std::string& source) = 0;

    // Language metadata
    virtual ScriptLanguage GetLanguage() const = 0;
    virtual ScriptExecutionMode GetExecutionMode() const = 0;
    virtual std::string GetLanguageName() const = 0;
    virtual std::string GetFileExtension() const = 0;

    // Type registration and interop
    virtual void RegisterTypes() {}
    virtual bool HasType(const std::string& typeName) const { return false; }
    virtual std::any CallFunction(const std::string& functionName, 
                                   const std::vector<std::any>& args) { return std::any(); }

    // Hot-reload support
    virtual bool SupportsHotReload() const { return false; }
    virtual void ReloadScript(const std::string& filepath) {}

    // Performance profiling
    virtual uint64_t GetMemoryUsage() const { return 0; }
    virtual double GetLastExecutionTime() const { return 0.0; }

    // Error handling
    virtual bool HasErrors() const { return false; }
    virtual std::string GetLastError() const { return ""; }
};
