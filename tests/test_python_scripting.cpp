#include <gtest/gtest.h>
#include "PythonScriptSystem.h"
#include "ScriptComponent.h"
#include "GameObject.h"
#include <fstream>

class PythonScriptingTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize PythonScriptSystem
        PythonScriptSystem::GetInstance().Init();
    }

    void TearDown() override {
        // Shutdown PythonScriptSystem
        PythonScriptSystem::GetInstance().Shutdown();
    }
};

TEST_F(PythonScriptingTest, RunSimpleScript) {
    std::string scriptPath = "test_script.py";
    std::ofstream out(scriptPath);
    out << "x = 10 + 20\n";
    out << "print('Computed x:', x)";
    out.close();

    bool result = PythonScriptSystem::GetInstance().RunScript(scriptPath);
    EXPECT_TRUE(result);
    // Note: We can't easily check 'x' in global scope without exposing more of pybind11 to public interface
    // but a return of true means no exception was thrown.
}

TEST_F(PythonScriptingTest, Vec3Binding) {
    std::string scriptPath = "test_vec3.py";
    std::ofstream out(scriptPath);
    out << "v = Vec3(1, 2, 3)\n";
    out << "assert v.x == 1.0\n";
    out << "v.y = 10\n";
    out << "assert v.y == 10.0\n";
    out.close();

    bool result = PythonScriptSystem::GetInstance().RunScript(scriptPath);
    EXPECT_TRUE(result);
}

TEST_F(PythonScriptingTest, GameObjectBinding) {
    auto go = std::make_shared<GameObject>("PythonGO");
    
    // We would need to pass this GO to python. 
    // Since our RunScript only runs a file, we can't easily pass arguments yet unless we expose a global or register function.
    // For now, let's verify we can create a GameObject in Python.
    
    std::string scriptPath = "test_go.py";
    std::ofstream out(scriptPath);
    out << "go = GameObject('PyCreatedGO')\n";
    out << "print(go.get_name())\n";
    out << "assert go.get_name() == 'PyCreatedGO'\n";
    out.close();

    bool result = PythonScriptSystem::GetInstance().RunScript(scriptPath);
    EXPECT_TRUE(result);
}
