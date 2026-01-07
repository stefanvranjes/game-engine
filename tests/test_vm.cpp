#include <gtest/gtest.h>
#include "CustomScriptSystem.h"
#include <vector>

class CustomVMTest : public ::testing::Test {
protected:
    CustomScriptSystem& sys = CustomScriptSystem::GetInstance();

    void SetUp() override {
        sys.Init();
    }

    void TearDown() override {
        sys.Shutdown();
    }
};

TEST_F(CustomVMTest, AssemblerBasic) {
    std::string source = "PUSH_FLOAT 10.0\nADD\nHALT";
    std::vector<Instruction> prog = sys.Assemble(source);
    
    EXPECT_EQ(prog.size(), 3);
    EXPECT_EQ(prog[0].op, OpCode::PUSH_FLOAT);
    EXPECT_FLOAT_EQ(prog[0].operand.as.f, 10.0f);
    EXPECT_EQ(prog[1].op, OpCode::ADD);
    EXPECT_EQ(prog[2].op, OpCode::HALT);
}

TEST_F(CustomVMTest, ExecutionArithmetic) {
    // 5 * 4 = 20
    std::string source = 
        "PUSH_FLOAT 5.0\n"
        "PUSH_FLOAT 4.0\n"
        "MUL\n"
        "HALT";
    
    // We need to inspect the stack after run, but CustomScriptSystem wraps it tight.
    // For testability, providing a way to access VM or return value would be better.
    // Assuming CustomScriptSystem::RunScript returns true on success.
    // Ideally we'd modify CustomScriptSystem to expose VM or add a Test function.
    // For now, we test if it runs without crashing.
    
    // Actually, let's test VirtualMachine directly for logic correctness
    VirtualMachine vm;
    std::vector<Instruction> prog;
    prog.push_back({OpCode::PUSH_FLOAT, Value(5.0f)});
    prog.push_back({OpCode::PUSH_FLOAT, Value(4.0f)});
    prog.push_back({OpCode::MUL}); // Result 20.0
    
    vm.LoadProgram(prog);
    vm.Run();
    
    EXPECT_FLOAT_EQ(vm.Pop().as.f, 20.0f);
}

TEST_F(CustomVMTest, Logic) {
    VirtualMachine vm;
    std::vector<Instruction> prog;
    // 5 < 10 -> True
    prog.push_back({OpCode::PUSH_FLOAT, Value(5.0f)});
    prog.push_back({OpCode::PUSH_FLOAT, Value(10.0f)});
    prog.push_back({OpCode::LT}); 
    
    vm.LoadProgram(prog);
    vm.Run();
    
    EXPECT_TRUE(vm.Pop().as.b);
}
