#include "ScriptSystem.h"
#include <iostream>
#include <filesystem>
#include "GameObject.h"
#include "Transform.h"

// Helper for error reporting
static void ReportError(lua_State* L, int status) {
    if (status != LUA_OK) {
        const char* msg = lua_tostring(L, -1);
        std::cerr << "Lua Error: " << msg << std::endl;
        lua_pop(L, 1); // remove error message
    }
}

LuaScriptSystem::LuaScriptSystem() {}

LuaScriptSystem::~LuaScriptSystem() {
    Shutdown();
}

void LuaScriptSystem::Init() {
    if (L) return;

    L = luaL_newstate();
    luaL_openlibs(L);

    RegisterTypes();
    
    std::cout << "LuaScriptSystem Initialized (Lua " << LUA_VERSION_MAJOR << "." << LUA_VERSION_MINOR << ")" << std::endl;
}

void LuaScriptSystem::Shutdown() {
    if (L) {
        lua_close(L);
        L = nullptr;
    }
}

void LuaScriptSystem::Update(float deltaTime) {
    // Optional
}

bool LuaScriptSystem::RunScript(const std::string& filepath) {
    if (!L) return false;

    int status = luaL_dofile(L, filepath.c_str());
    ReportError(L, status);
    return status == LUA_OK;
}

// ... rest of bindings ...

void LuaScriptSystem::RegisterTypes() {
    // Register Vec3
    luaL_newmetatable(L, "Vec3");
    lua_pushcfunction(L, Vec3_ToString); lua_setfield(L, -2, "__tostring");
    lua_pushcfunction(L, Vec3_Index); lua_setfield(L, -2, "__index");
    lua_pushcfunction(L, Vec3_NewIndex); lua_setfield(L, -2, "__newindex");
    lua_pop(L, 1);

    // Global 'Vec3' constructor
    lua_register(L, "Vec3", Vec3_New);
}
