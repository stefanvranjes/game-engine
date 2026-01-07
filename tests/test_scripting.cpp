#include <gtest/gtest.h>
#include "ScriptSystem.h"
#include "ScriptComponent.h"
#include "GameObject.h"
#include <fstream>

class ScriptingTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize ScriptSystem
        ScriptSystem::GetInstance().Init();
    }

    void TearDown() override {
        // Shutdown ScriptSystem
        ScriptSystem::GetInstance().Shutdown();
    }
};

TEST_F(ScriptingTest, InitShutdown) {
    EXPECT_TRUE(ScriptSystem::GetInstance().GetLuaState() != nullptr);
}

TEST_F(ScriptingTest, RunSimpleScript) {
    std::string scriptPath = "test_script.lua";
    std::ofstream out(scriptPath);
    out << "x = 10 + 20";
    out.close();

    bool result = ScriptSystem::GetInstance().RunScript(scriptPath);
    EXPECT_TRUE(result);
    
    lua_State* L = ScriptSystem::GetInstance().GetLuaState();
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

    bool result = ScriptSystem::GetInstance().RunScript(scriptPath);
    EXPECT_TRUE(result);
    
    lua_State* L = ScriptSystem::GetInstance().GetLuaState();
    
    // Check x
    lua_getglobal(L, "x");
    EXPECT_EQ(lua_tonumber(L, -1), 1.0);
    lua_pop(L, 1);

    // Check v.y modification (need to inspect userdata)
    lua_getglobal(L, "v");
    // We can't easily inspect C++ userdata from C++ side via public API without casting, 
    // but the script execution success implies bindings work if we add assertions in script.
    // Let's rely on script side validation or simple checks.
    
    // Actually, let's verify by returning v.y to a global
    ScriptSystem::GetInstance().RunScript("y_val = v.y"); // This assumes runscript keeps state (it does)
    
    // Wait, RunScript uses luaL_dofile. If ScriptSystem::RunScript re-uses same L, state persists.
    // Let's check.
    // Yes, ScriptSystem::Init() creates L once.
    // RunScript helper uses luaL_dofile.
    
    // Use string execution for quick checks? ScriptSystem doesn't expose DoString.
    // Let's create another file or just append to previous
    std::ofstream out2("test_vec3_check.lua");
    out2 << "y_val = v.y";
    out2.close();
    
    ScriptSystem::GetInstance().RunScript("test_vec3_check.lua");
    
    lua_getglobal(L, "y_val");
    EXPECT_EQ(lua_tonumber(L, -1), 10.0);
    lua_pop(L, 1);
}

TEST_F(ScriptingTest, ScriptComponent) {
    auto go = std::make_shared<GameObject>("ScriptedGO");
    auto scriptComp = std::make_shared<ScriptComponent>(go);
    go->SetScriptComponent(scriptComp);

    // Create a dummy script
    std::string scriptPath = "component_script.lua";
    std::ofstream out(scriptPath);
    out << "global_update_count = 0\n";
    out << "function Update(dt)\n"; // Note: Our current simple implementation doesn't call Update function yet
                                     // It just runs code immediately.
                                     // Let's verify what we implemented in ScriptComponent.cpp
                                     // "For this iteration, let's execute the script immediately."
                                     // And "Update(dt)" was empty.
    out << "global_var = 100\n";
    out.close();

    scriptComp->LoadScript(scriptPath);
    
    // Verify execution
    lua_State* L = ScriptSystem::GetInstance().GetLuaState();
    lua_getglobal(L, "global_var");
    EXPECT_EQ(lua_tonumber(L, -1), 100.0);
    lua_pop(L, 1);
}
