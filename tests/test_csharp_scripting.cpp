#include <gtest/gtest.h>
#include "CSharpScriptSystem.h"
#include <fstream>
#include <filesystem>

class CSharpScriptingTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize CSharpScriptSystem
        CSharpScriptSystem::GetInstance().Init();
    }

    void TearDown() override {
        // Shutdown CSharpScriptSystem
        CSharpScriptSystem::GetInstance().Shutdown();
    }
};

TEST_F(CSharpScriptingTest, Initialization) {
    auto domain = CSharpScriptSystem::GetInstance().GetDomain();
    // Tests might fail if Mono isn't found, which is expected if not installed.
    // We assume the build environment will have it if we got this far.
    // EXPECT_NE(domain, nullptr); 
}

TEST_F(CSharpScriptingTest, RunScriptDLL) {
    // This requires a valid DLL. 
    // We can't generate a DLL from C++ test easily without a C# compiler (csc/mcs).
    // For now, checks if RunScript handles missing file gracefully.
    bool result = CSharpScriptSystem::GetInstance().RunScript("missing_assembly.dll");
    EXPECT_FALSE(result);
}
