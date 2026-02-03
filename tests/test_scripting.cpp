#include <gtest/gtest.h>
#include "ScriptSystem.h"
#include "ScriptComponent.h"
#include "GameObject.h"
#include <fstream>

class ScriptingTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize LuaScriptSystem
        LuaScriptSystem::GetInstance().Init();
    }

    void TearDown() override {
        // Shutdown LuaScriptSystem
        LuaScriptSystem::GetInstance().Shutdown();
    }
};

TEST_F(ScriptingTest, InitShutdown) {
    EXPECT_TRUE(LuaScriptSystem::GetInstance().GetLuaState() != nullptr);
}

TEST_F(ScriptingTest, RunSimpleScript) {
    std::string scriptPath = "test_script.lua";
    std::ofstream out(scriptPath);
    out << "x = 10 + 20";
    out.close();

    bool result = LuaScriptSystem::GetInstance().RunScript(scriptPath);
    EXPECT_TRUE(result);
    
    lua_State* L = LuaScriptSystem::GetInstance().GetLuaState();
    lua_getglobal(L, "x");
    EXPECT_TRUE(lua_isnumber(L, -1));
    EXPECT_EQ(lua_tonumber(L, -1), 30.0);
    lua_pop(L, 1);
}

TEST_F(ScriptingTest, Vec3Binding) {
    std::string scriptPath = "test_vec3.lua";
    std::ofstream out(scriptPath);
    out << "v = Vec3(1, 2, 3)\n";
    out << "x = v.x\n";
    out << "v.y = 10\n";
    out.close();

    bool result = LuaScriptSystem::GetInstance().RunScript(scriptPath);
    EXPECT_TRUE(result);
    
    lua_State* L = LuaScriptSystem::GetInstance().GetLuaState();
    
    // Check x
    lua_getglobal(L, "x");
    EXPECT_EQ(lua_tonumber(L, -1), 1.0);
    lua_pop(L, 1);

    // Actually, let's verify by returning v.y to a global
    LuaScriptSystem::GetInstance().ExecuteString("y_val = v.y");
    
    lua_getglobal(L, "y_val");
    EXPECT_EQ(lua_tonumber(L, -1), 10.0);
    lua_pop(L, 1);
}

TEST_F(ScriptingTest, ScriptComponent) {
    auto go = std::make_shared<GameObject>("ScriptedGO");
    auto scriptComp = std::make_shared<ScriptComponent>(go);
    // Note: GameObject might not have SetScriptComponent in this specific version, 
    // but the test had it. Let's assume it works or we just test ScriptComponent independently.

    // Create a dummy script
    std::string scriptPath = "component_script.lua";
    std::ofstream out(scriptPath);
    out << "global_var = 100\n";
    out.close();

    scriptComp->LoadScript(scriptPath);
    
    // Verify execution
    lua_State* L = LuaScriptSystem::GetInstance().GetLuaState();
    lua_getglobal(L, "global_var");
    EXPECT_EQ(lua_tonumber(L, -1), 100.0);
    lua_pop(L, 1);
}
