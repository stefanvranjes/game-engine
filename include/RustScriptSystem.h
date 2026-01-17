#pragma once

#include "IScriptSystem.h"
#include <string>
#include <memory>
#include <map>
#include <dlfcn.h>
#include <functional>
#include <vector>

/**
 * @class RustScriptSystem
 * @brief Integrates Rust scripting for safe, high-performance gameplay logic
 * 
 * Rust scripts are compiled to native code or WASM, providing memory safety
 * without garbage collection overhead. Ideal for performance-critical gameplay systems.
 * 
 * Features:
 * - Type-safe systems programming language
 * - Memory safety without garbage collection
 * - Zero-cost abstractions
 * - Compiled to native libraries (.dll/.so) or WASM
 * - Easy C FFI for integration with game engine
 * - Excellent for physics integration, AI, networking
 * 
 * Usage:
 * ```cpp
 * RustScriptSystem::GetInstance().Init();
 * RustScriptSystem::GetInstance().LoadLibrary("gameplay.dll");
 * RustScriptSystem::GetInstance().CallFunction("update_player", args);
 * ```
 * 
 * Example Rust Script:
 * ```rust
 * #[no_mangle]
 * pub extern "C" fn update_player(game_obj: *mut GameObject, dt: f32) {
 *     unsafe {
 *         let obj = &mut *game_obj;
 *         obj.update(dt);
 *     }
 * }
 * 
 * pub struct Player {
 *     speed: f32,
 *     health: u32,
 * }
 * 
 * impl Player {
 *     pub fn new(speed: f32) -> Self {
 *         Player { speed, health: 100 }
 *     }
 * 
 *     pub fn take_damage(&mut self, damage: u32) {
 *         if damage < self.health {
 *             self.health -= damage;
 *         } else {
 *             self.health = 0;
 *         }
 *     }
 * }
 * ```
 */
class RustScriptSystem : public IScriptSystem {
public:
    // Function pointer types for Rust functions
    using RustInitFunc = void (*)(void);
    using RustShutdownFunc = void (*)(void);
    using RustUpdateFunc = void (*)(float);
    using RustGenericFunc = void* (*)(const void*, size_t);

    static RustScriptSystem& GetInstance() {
        static RustScriptSystem instance;
        return instance;
    }

    // IScriptSystem Implementation
    void Init() override;
    void Shutdown() override;
    void Update(float deltaTime) override;
    bool RunScript(const std::string& filepath) override;

    /**
     * Load a compiled Rust library
     * @param libPath Path to .dll/.so file
     * @return true if successful, false on error
     */
    bool LoadLibrary(const std::string& libPath);

    /**
     * Execute string (not applicable for Rust, returns false)
     */
    bool ExecuteString(const std::string& source) override;

    // Language metadata
    ScriptLanguage GetLanguage() const override { return ScriptLanguage::Rust; }
    ScriptExecutionMode GetExecutionMode() const override { return ScriptExecutionMode::NativeCompiled; }
    std::string GetLanguageName() const override { return "Rust (Native/WASM)"; }
    std::string GetFileExtension() const override { return ".dll"; }

    // Type registration
    void RegisterTypes() override {}
    bool HasType(const std::string& typeName) const override;

    /**
     * Call a Rust function from C++
     * @param functionName Name of exported Rust function
     * @param args Arguments to pass
     * @return Return value from Rust function
     */
    std::any CallFunction(const std::string& functionName,
                          const std::vector<std::any>& args) override;

    // Hot-reload support (requires careful unloading of previous library)
    bool SupportsHotReload() const override { return true; }
    void ReloadScript(const std::string& filepath) override;

    // Performance metrics
    uint64_t GetMemoryUsage() const override;
    double GetLastExecutionTime() const override;

    // Error handling
    bool HasErrors() const override;
    std::string GetLastError() const override;

    /**
     * Get loaded library handle (platform-specific)
     */
    void* GetLibraryHandle(const std::string& libPath) const;

    /**
     * Get function pointer from loaded library
     */
    template<typename FuncType>
    FuncType GetFunction(void* libHandle, const std::string& functionName) {
        if (!libHandle) {
            SetError("Invalid library handle");
            return nullptr;
        }

#ifdef _WIN32
        FuncType func = reinterpret_cast<FuncType>(GetProcAddress((HMODULE)libHandle, functionName.c_str()));
#else
        FuncType func = reinterpret_cast<FuncType>(dlsym(libHandle, functionName.c_str()));
#endif
        
        if (!func) {
            SetError("Function not found: " + functionName);
        }
        return func;
    }

private:
    RustScriptSystem();
    ~RustScriptSystem();

    std::map<std::string, void*> m_LoadedLibraries; // library name -> handle
    std::string m_LastError;
    double m_LastExecutionTime = 0.0;
    uint64_t m_MemoryUsage = 0;

    void SetError(const std::string& error);
    bool UnloadLibrary(const std::string& libPath);
};
