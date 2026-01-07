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

ScriptSystem::ScriptSystem() {}

ScriptSystem::~ScriptSystem() {
    Shutdown();
}

void ScriptSystem::Init() {
    if (L) return;

    L = luaL_newstate();
    luaL_openlibs(L);

    RegisterTypes();
    
    std::cout << "ScriptSystem Initialized (Lua " << LUA_VERSION_MAJOR << "." << LUA_VERSION_MINOR << ")" << std::endl;
}

void ScriptSystem::Shutdown() {
    if (L) {
        lua_close(L);
        L = nullptr;
    }
}

void ScriptSystem::Update(float deltaTime) {
    // Optional: Global script update if needed
    // Typically, ScriptComponents update themselves
}

bool ScriptSystem::RunScript(const std::string& filepath) {
    if (!L) return false;

    int status = luaL_dofile(L, filepath.c_str());
    ReportError(L, status);
    return status == LUA_OK;
}

// --- Bindings ---

// Vec3 Userdata
struct LuaVec3 {
    float x, y, z;
};

static int Vec3_New(lua_State* L) {
    float x = (float)luaL_checknumber(L, 1);
    float y = (float)luaL_checknumber(L, 2);
    float z = (float)luaL_checknumber(L, 3);

    LuaVec3* v = (LuaVec3*)lua_newuserdata(L, sizeof(LuaVec3));
    v->x = x;
    v->y = y;
    v->z = z;

    luaL_getmetatable(L, "Vec3");
    lua_setmetatable(L, -2);
    return 1;
}

static int Vec3_ToString(lua_State* L) {
    LuaVec3* v = (LuaVec3*)luaL_checkudata(L, 1, "Vec3");
    lua_pushfstring(L, "Vec3(%f, %f, %f)", v->x, v->y, v->z);
    return 1;
}

static int Vec3_Index(lua_State* L) {
    LuaVec3* v = (LuaVec3*)luaL_checkudata(L, 1, "Vec3");
    const char* key = luaL_checkstring(L, 2);

    if (strcmp(key, "x") == 0) { lua_pushnumber(L, v->x); return 1; }
    if (strcmp(key, "y") == 0) { lua_pushnumber(L, v->y); return 1; }
    if (strcmp(key, "z") == 0) { lua_pushnumber(L, v->z); return 1; }

    return 0; 
}

static int Vec3_NewIndex(lua_State* L) {
    LuaVec3* v = (LuaVec3*)luaL_checkudata(L, 1, "Vec3");
    const char* key = luaL_checkstring(L, 2);
    float val = (float)luaL_checknumber(L, 3);

    if (strcmp(key, "x") == 0) v->x = val;
    else if (strcmp(key, "y") == 0) v->y = val;
    else if (strcmp(key, "z") == 0) v->z = val;

    return 0;
}

void ScriptSystem::RegisterTypes() {
    // Register Vec3
    luaL_newmetatable(L, "Vec3");
    lua_pushcfunction(L, Vec3_ToString); lua_setfield(L, -2, "__tostring");
    lua_pushcfunction(L, Vec3_Index); lua_setfield(L, -2, "__index");
    lua_pushcfunction(L, Vec3_NewIndex); lua_setfield(L, -2, "__newindex");
    lua_pop(L, 1);

    // Global 'Vec3' constructor
    lua_register(L, "Vec3", Vec3_New);
}
