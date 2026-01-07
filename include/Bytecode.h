#pragma once

#include <vector>
#include <string>
#include <variant>
#include <cstdint>

// Supported value types in the VM
enum class ValueType {
    Float,
    Int,
    Bool,
    String, // Index to string table usually, or raw string for simplicity
    Handle  // For Engine Objects (GameObject ID)
};

struct Value {
    ValueType type;
    union {
        float f;
        int i;
        bool b;
        uint32_t handle;
    } as;
    std::string str; // keeping string outside union for simplicity (C++17 variant would be better but keeping it simple)

    Value() : type(ValueType::Float) { as.f = 0.0f; }
    explicit Value(float v) : type(ValueType::Float) { as.f = v; }
    explicit Value(int v) : type(ValueType::Int) { as.i = v; }
    explicit Value(bool v) : type(ValueType::Bool) { as.b = v; }
    explicit Value(const std::string& v) : type(ValueType::String), str(v) {}
    explicit Value(uint32_t v, ValueType t) : type(t) { as.handle = v; } // For handles

    // Helpers
    bool IsNumber() const { return type == ValueType::Float || type == ValueType::Int; }
    float AsFloat() const { return type == ValueType::Float ? as.f : (float)as.i; }
};

enum class OpCode : uint8_t {
    HALT = 0,
    
    // Stack Ops
    PUSH_FLOAT, // Followed by float operand
    PUSH_INT,   // Followed by int operand
    PUSH_BOOL,  // Followed by bool operand
    PUSH_STR,   // Followed by string operand (simplified)
    POP,
    DUP,        // Duplicate top of stack

    // Arithmetic
    ADD,
    SUB,
    MUL,
    DIV,

    // Logic
    EQ,
    NEQ,
    GT,
    LT,
    
    // Control Flow
    JMP,        // Unconditional Jump (operand: relative offset or absolute index)
    JIF,        // Jump If False (operand: relative offset or absolute index)
    
    // Variables (Simple global register file or indexed memory)
    LOAD,       // Load from variable index (operand: int index)
    STORE,      // Store to variable index (operand: int index)

    // System
    PRINT,      // Print top of stack
    API_CALL    // Call engine function (operand: function ID)
};

struct Instruction {
    OpCode op;
    Value operand; // Optional operand
};
