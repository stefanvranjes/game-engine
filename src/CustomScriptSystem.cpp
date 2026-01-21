#include "CustomScriptSystem.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>

// API Function IDs
enum APIFunctions {
    API_LOG = 0,
    API_VEC3_CREATE = 1,
    API_VEC3_PRINT = 2,
    API_GAMEOBJECT_FIND = 3,
    API_GAMEOBJECT_MOVE = 4
};

CustomScriptSystem::CustomScriptSystem() {
}

CustomScriptSystem::~CustomScriptSystem() {
    Shutdown();
}

void CustomScriptSystem::Init() {
    std::cout << "CustomScriptSystem Initialized (Bytecode VM)" << std::endl;
    
    // Bind API Callback
    m_VM.SetAPICallback([this](int id, VirtualMachine& vm) {
        this->HandleAPICall(id, vm);
    });
}

void CustomScriptSystem::Shutdown() {
    m_VM.Reset();
}

void CustomScriptSystem::Update(float deltaTime) {
    // Could eventually run a coroutine scheduler here
}

bool CustomScriptSystem::RunScript(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Failed to open script: " << filepath << std::endl;
        return false;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string source = buffer.str();

    std::vector<Instruction> program = Assemble(source);
    if (program.empty()) {
        std::cerr << "Script assembled to empty program or error" << std::endl;
        return false;
    }

    std::cout << "Running script: " << filepath << " (" << program.size() << " instructions)" << std::endl;
    m_VM.LoadProgram(program);
    m_VM.Run();
    
    return true;
}

bool CustomScriptSystem::ExecuteString(const std::string& source) {
    std::vector<Instruction> program = Assemble(source);
    if (program.empty()) {
        std::cerr << "Script assembled to empty program or error" << std::endl;
        return false;
    }

    // std::cout << "Executing string script (" << program.size() << " instructions)" << std::endl;
    m_VM.LoadProgram(program);
    m_VM.Run();
    
    return true;
}

std::vector<Instruction> CustomScriptSystem::Assemble(const std::string& source) {
    std::vector<Instruction> program;
    std::stringstream ss(source);
    std::string line;
    
    while (std::getline(ss, line)) {
        // Simple tokenization
        std::stringstream lineStream(line);
        std::string token;
        lineStream >> token;

        if (token.empty() || token[0] == ';' || token.substr(0, 2) == "//") continue; // Comment/Empty

        Instruction instr;
        
        if (token == "PUSH_FLOAT") {
            instr.op = OpCode::PUSH_FLOAT;
            float val; lineStream >> val;
            instr.operand = Value(val);
        } else if (token == "PUSH_INT") {
            instr.op = OpCode::PUSH_INT;
            int val; lineStream >> val;
            instr.operand = Value(val);
        } else if (token == "PUSH_BOOL") {
            instr.op = OpCode::PUSH_BOOL;
            std::string bVal; lineStream >> bVal;
            instr.operand = Value(bVal == "true");
        } else if (token == "PUSH_STR") {
            instr.op = OpCode::PUSH_STR;
            std::string sVal; 
            // Read rest of line as string
            std::getline(lineStream, sVal); 
            // Trim leading space
            size_t first = sVal.find_first_not_of(' ');
            if (first != std::string::npos) sVal = sVal.substr(first);
            // Remove quotes if present
            if (sVal.size() >= 2 && sVal.front() == '"' && sVal.back() == '"') {
                sVal = sVal.substr(1, sVal.size() - 2);
            }
            instr.operand = Value(sVal);
        } else if (token == "ADD") instr.op = OpCode::ADD;
        else if (token == "SUB") instr.op = OpCode::SUB;
        else if (token == "MUL") instr.op = OpCode::MUL;
        else if (token == "DIV") instr.op = OpCode::DIV;
        else if (token == "PRINT") instr.op = OpCode::PRINT;
        else if (token == "HALT") instr.op = OpCode::HALT;
        else if (token == "POP") instr.op = OpCode::POP;
        else if (token == "DUP") instr.op = OpCode::DUP;
        else if (token == "EQ") instr.op = OpCode::EQ;
        else if (token == "GT") instr.op = OpCode::GT;
        else if (token == "LT") instr.op = OpCode::LT;
        else if (token == "JMP") {
            instr.op = OpCode::JMP;
            int val; lineStream >> val;
            instr.operand = Value(val);
        } else if (token == "JIF") {
            instr.op = OpCode::JIF;
            int val; lineStream >> val;
            instr.operand = Value(val);
        } else if (token == "STORE") {
            instr.op = OpCode::STORE;
            int val; lineStream >> val;
            instr.operand = Value(val);
        } else if (token == "LOAD") {
            instr.op = OpCode::LOAD;
            int val; lineStream >> val;
            instr.operand = Value(val);
        } else if (token == "API_CALL") {
            instr.op = OpCode::API_CALL;
            std::string func; lineStream >> func;
            int id = 0;
            if (func == "LOG") id = API_LOG;
            else if (func == "VEC3_CREATE") id = API_VEC3_CREATE;
            else if (func == "VEC3_PRINT") id = API_VEC3_PRINT;
            else if (func == "GO_FIND") id = API_GAMEOBJECT_FIND;
            else if (func == "GO_MOVE") id = API_GAMEOBJECT_MOVE;
            else {
                try { id = std::stoi(func); } catch(...) {}
            }
            instr.operand = Value(id);
        } else {
            std::cerr << "Unknown opcode: " << token << std::endl;
            continue;
        }

        program.push_back(instr);
    }
    
    return program;
}

void CustomScriptSystem::HandleAPICall(int funcID, VirtualMachine& vm) {
    switch(funcID) {
        case API_LOG: {
            Value msg = vm.Pop();
            std::cout << "[SCRIPT LOG] " << msg.str << std::endl;
            break;
        }
        case API_VEC3_CREATE: {
            // Expects x, y, z on stack
            // Returns handle (simplified, just prints for now as we don't have struct support in VM yet really)
            Value z = vm.Pop();
            Value y = vm.Pop();
            Value x = vm.Pop();
            std::cout << "[API] Created Vec3(" << x.AsFloat() << ", " << y.AsFloat() << ", " << z.AsFloat() << ")" << std::endl;
            vm.Push(Value(100)); // Dummy handle
            break;
        }
        default:
            std::cerr << "[API] Unknown Function ID: " << funcID << std::endl;
            break;
    }
}
