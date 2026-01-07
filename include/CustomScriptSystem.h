#pragma once

#include "IScriptSystem.h"
#include "VirtualMachine.h"
#include <string>
#include <vector>
#include <map>

class CustomScriptSystem : public IScriptSystem {
public:
    static CustomScriptSystem& GetInstance() {
        static CustomScriptSystem instance;
        return instance;
    }

    void Init() override;
    void Shutdown() override;
    void Update(float deltaTime) override;
    bool RunScript(const std::string& filepath) override;

    // Helper to parse assembly text into bytecode
    std::vector<Instruction> Assemble(const std::string& source);

private:
    CustomScriptSystem();
    ~CustomScriptSystem();

    VirtualMachine m_VM;

    // API Call Binding
    void HandleAPICall(int funcID, VirtualMachine& vm);
};
