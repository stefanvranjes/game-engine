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

// Vec3 Bindings
static int Vec3_New(lua_State* L) {
    float x = (float)luaL_checknumber(L, 1);
    float y = (float)luaL_checknumber(L, 2);
    float z = (float)luaL_checknumber(L, 3);
    
    // Allocate userdata
    Vec3* v = (Vec3*)lua_newuserdata(L, sizeof(Vec3));
    v->x = x;
    v->y = y;
    v->z = z;
    
    luaL_getmetatable(L, "Vec3");
    lua_setmetatable(L, -2);
    return 1;
}

static int Vec3_ToString(lua_State* L) {
    Vec3* v = (Vec3*)luaL_checkudata(L, 1, "Vec3");
    lua_pushfstring(L, "Vec3(%.2f, %.2f, %.2f)", v->x, v->y, v->z);
    return 1;
}

static int Vec3_Index(lua_State* L) {
    Vec3* v = (Vec3*)luaL_checkudata(L, 1, "Vec3");
    std::string key = luaL_checkstring(L, 2);
    
    if (key == "x") lua_pushnumber(L, v->x);
    else if (key == "y") lua_pushnumber(L, v->y);
    else if (key == "z") lua_pushnumber(L, v->z);
    else lua_pushnil(L);
    
    return 1;
}

static int Vec3_NewIndex(lua_State* L) {
    Vec3* v = (Vec3*)luaL_checkudata(L, 1, "Vec3");
    std::string key = luaL_checkstring(L, 2);
    float val = (float)luaL_checknumber(L, 3);
    
    if (key == "x") v->x = val;
    else if (key == "y") v->y = val;
    else if (key == "z") v->z = val;
    
    return 0;
}

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
