#pragma once

#include "Bytecode.h"
#include <vector>
#include <functional>
#include <iostream>

class VirtualMachine {
public:
    VirtualMachine();
    ~VirtualMachine();

    // Load a program into the VM
    void LoadProgram(const std::vector<Instruction>& program);

    // Reset VM state (IP, Stack)
    void Reset();

    // Execute the loaded program until HALT or end
    void Run();

    // Stack Operations
    void Push(const Value& v);
    Value Pop();
    Value Peek(int offset = 0);

    // Globals/Variables Access
    void SetGlobal(int index, const Value& v);
    Value GetGlobal(int index);

    // Callbacks
    using APICallback = std::function<void(int, VirtualMachine&)>;
    void SetAPICallback(APICallback callback) { m_APICallback = callback; }

    // Helpers
    int StackSize() const { return (int)m_Stack.size(); }
    void PrintStack();

private:
    std::vector<Value> m_Stack;
    std::vector<Instruction> m_Program;
    std::vector<Value> m_Globals; // Simple register file

    size_t m_IP; // Instruction Pointer
    bool m_Running;

    APICallback m_APICallback;

    // Helper for math ops to reduce duplication
    void ExecuteArithmetic(OpCode op);
    void ExecuteLogic(OpCode op);
};
