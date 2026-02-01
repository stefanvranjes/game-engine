#pragma once

#include "IScriptSystem.h"
#include <string>
#include <memory>
extern "C" {
#include "lua.h"
#include "lualib.h"
#include "lauxlib.h"
}

// Forward declarations
class GameObject;
class ScriptComponent;

// Renamed to LuaScriptSystem, but for now we keep the class name "ScriptSystem" 
// to avoid breaking all existing code immediately, or we rename it and update widely.
// Let's alias it or rename. Rename is cleaner.
class LuaScriptSystem : public IScriptSystem {
public:
    static LuaScriptSystem& GetInstance() {
        static LuaScriptSystem instance;
        return instance;
    }

    // IScriptSystem Implementation
    void Init() override;
    void Shutdown() override;
    void Update(float deltaTime) override;
    bool RunScript(const std::string& filepath) override;
    bool ExecuteString(const std::string& source) override;

    // Language metadata
    ScriptLanguage GetLanguage() const override { return ScriptLanguage::Lua; }
    ScriptExecutionMode GetExecutionMode() const override { return ScriptExecutionMode::Interpreted; }
    std::string GetLanguageName() const override { return "Lua"; }
    std::string GetFileExtension() const override { return ".lua"; }

    lua_State* GetLuaState() const { return L; }

    // Register engine types
    void RegisterTypes() override;

    bool HasFunction(const std::string& functionName) const override;

    LuaScriptSystem();
    ~LuaScriptSystem();

private:

    lua_State* L = nullptr;
};
