#pragma once

#include <string>

class IScriptSystem {
public:
    virtual ~IScriptSystem() = default;

    virtual void Init() = 0;
    virtual void Shutdown() = 0;
    virtual void Update(float deltaTime) = 0;

    virtual bool RunScript(const std::string& filepath) = 0;
    
    // Generic way to register types? 
    // Implementing this generically is hard without templates/macros bridging the gap.
    // For now, each system registers types internally in Init().
    // virtual void RegisterTypes() = 0; 
};
