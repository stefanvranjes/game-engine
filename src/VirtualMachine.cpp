#include "VirtualMachine.h"
#include <iostream>
#include <cmath>

VirtualMachine::VirtualMachine() : m_IP(0), m_Running(false) {
    m_Stack.reserve(1024);
    m_Globals.resize(256); // 256 global registers for now
}

VirtualMachine::~VirtualMachine() {
}

void VirtualMachine::LoadProgram(const std::vector<Instruction>& program) {
    m_Program = program;
    Reset();
}

void VirtualMachine::Reset() {
    m_IP = 0;
    m_Stack.clear();
    m_Running = false;
}

void VirtualMachine::Push(const Value& v) {
    m_Stack.push_back(v);
}

Value VirtualMachine::Pop() {
    if (m_Stack.empty()) {
        std::cerr << "VM Error: Stack Underflow" << std::endl;
        return Value(0);
    }
    Value val = m_Stack.back();
    m_Stack.pop_back();
    return val;
}

Value VirtualMachine::Peek(int offset) {
    if (m_Stack.size() <= offset) return Value(0);
    return m_Stack[m_Stack.size() - 1 - offset];
}

void VirtualMachine::SetGlobal(int index, const Value& v) {
    if (index >= 0 && index < m_Globals.size()) {
        m_Globals[index] = v;
    }
}

Value VirtualMachine::GetGlobal(int index) {
    if (index >= 0 && index < m_Globals.size()) {
        return m_Globals[index];
    }
    return Value(0);
}

void VirtualMachine::Run() {
    m_Running = true;
    while (m_Running && m_IP < m_Program.size()) {
        const Instruction& instr = m_Program[m_IP];
        m_IP++; // Advance IP

        switch (instr.op) {
            case OpCode::HALT:
                m_Running = false;
                break;

            case OpCode::PUSH_FLOAT:
            case OpCode::PUSH_INT:
            case OpCode::PUSH_BOOL:
            case OpCode::PUSH_STR:
                Push(instr.operand);
                break;

            case OpCode::POP:
                Pop();
                break;
            
            case OpCode::DUP:
                Push(Peek());
                break;

            case OpCode::ADD:
            case OpCode::SUB:
            case OpCode::MUL:
            case OpCode::DIV:
                ExecuteArithmetic(instr.op);
                break;

            case OpCode::EQ:
            case OpCode::NEQ:
            case OpCode::GT:
            case OpCode::LT:
                ExecuteLogic(instr.op);
                break;

            case OpCode::PRINT: {
                Value val = Peek(); // Don't pop for print? Or pop? Usually Pop.
                // Let's Pop.
                val = Pop();
                if (val.type == ValueType::Float) std::cout << val.as.f << std::endl;
                else if (val.type == ValueType::Int) std::cout << val.as.i << std::endl;
                else if (val.type == ValueType::Bool) std::cout << (val.as.b ? "true" : "false") << std::endl;
                else if (val.type == ValueType::String) std::cout << val.str << std::endl;
                break;
            }

            case OpCode::JMP:
                // Relative jump using operand.as.i
                m_IP += instr.operand.as.i; 
                break;
            
            case OpCode::JIF: {
                Value val = Pop();
                bool condition = false;
                if (val.type == ValueType::Bool) condition = val.as.b;
                else if (val.type == ValueType::Int) condition = val.as.i != 0;
                
                if (!condition) {
                    m_IP += instr.operand.as.i;
                }
                break;
            }

            case OpCode::LOAD:
                Push(GetGlobal(instr.operand.as.i));
                break;
            
            case OpCode::STORE:
                SetGlobal(instr.operand.as.i, Pop());
                break;

            case OpCode::API_CALL:
                if (m_APICallback) {
                    m_APICallback(instr.operand.as.i, *this);
                }
                break;
        }
    }
}

void VirtualMachine::ExecuteArithmetic(OpCode op) {
    Value b = Pop();
    Value a = Pop();

    // Promote to float if either is float
    if (a.type == ValueType::Float || b.type == ValueType::Float) {
        float fA = a.AsFloat();
        float fB = b.AsFloat();
        float res = 0;
        
        switch(op) {
            case OpCode::ADD: res = fA + fB; break;
            case OpCode::SUB: res = fA - fB; break;
            case OpCode::MUL: res = fA * fB; break;
            case OpCode::DIV: res = fA / fB; break;
        }
        Push(Value(res));
    } else {
        int iA = a.as.i;
        int iB = b.as.i;
        int res = 0;
        
        switch(op) {
            case OpCode::ADD: res = iA + iB; break;
            case OpCode::SUB: res = iA - iB; break;
            case OpCode::MUL: res = iA * iB; break;
            case OpCode::DIV: res = (iB != 0) ? iA / iB : 0; break;
        }
        Push(Value(res));
    }
}

void VirtualMachine::ExecuteLogic(OpCode op) {
    Value b = Pop();
    Value a = Pop();
    bool res = false;
    
    // Simplification: compare as floats
    float fA = a.AsFloat();
    float fB = b.AsFloat();

    switch(op) {
        case OpCode::EQ:  res = std::abs(fA - fB) < 0.0001f; break;
        case OpCode::NEQ: res = std::abs(fA - fB) >= 0.0001f; break;
        case OpCode::GT:  res = fA > fB; break;
        case OpCode::LT:  res = fA < fB; break;
    }
    Push(Value(res));
}

void VirtualMachine::PrintStack() {
    std::cout << "Stack [" << m_Stack.size() << "]: ";
    for (const auto& v : m_Stack) {
        if (v.type == ValueType::Float) std::cout << v.as.f << " ";
        else if (v.type == ValueType::Int) std::cout << v.as.i << " ";
        else if (v.type == ValueType::String) std::cout << "\"" << v.str << "\" ";
        else if (v.type == ValueType::Bool) std::cout << (v.as.b ? "T" : "F") << " ";
    }
    std::cout << std::endl;
}
