#pragma once

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

class ScriptSystem {
public:
    static ScriptSystem& GetInstance() {
        static ScriptSystem instance;
        return instance;
    }

    ScriptSystem(const ScriptSystem&) = delete;
    ScriptSystem& operator=(const ScriptSystem&) = delete;

    void Init();
    void Shutdown();
    void Update(float deltaTime);

    lua_State* GetLuaState() const { return L; }

    // Helper to run a script file
    bool RunScript(const std::string& filepath);

    // Register engine types
    void RegisterTypes();

private:
    ScriptSystem();
    ~ScriptSystem();

    lua_State* L = nullptr;
};
