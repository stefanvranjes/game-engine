#include "WrenScriptComponent.h"
#include "WrenScriptSystem.h"
#include "GameObject.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>

WrenScriptComponent::WrenScriptComponent(std::weak_ptr<GameObject> owner)
    : m_Owner(owner), m_UpdateEnabled(true), m_Initialized(false) {
}

WrenScriptComponent::~WrenScriptComponent() {
    ReleaseFunctionHandles();
}

bool WrenScriptComponent::LoadScript(const std::string& filepath) {
    return LoadScripts({ filepath });
}

bool WrenScriptComponent::LoadScripts(const std::vector<std::string>& filepaths) {
    WrenScriptSystem& wrenSystem = WrenScriptSystem::GetInstance();
    
    // Set module name from first script
    if (!filepaths.empty()) {
        m_ModuleName = std::filesystem::path(filepaths[0]).stem().string();
    }
    
    // Load all scripts
    for (const auto& filepath : filepaths) {
        if (!wrenSystem.RunScript(filepath)) {
            std::cerr << "Failed to load Wren script: " << filepath << std::endl;
            return false;
        }
        m_LoadedScripts.push_back(filepath);
    }
    
    // Cache function handles for common lifecycle methods
    CacheFunctionHandles();
    
    return true;
}

bool WrenScriptComponent::LoadCode(const std::string& source, const std::string& scriptName) {
    WrenScriptSystem& wrenSystem = WrenScriptSystem::GetInstance();
    
    if (!wrenSystem.ExecuteString(source)) {
        std::cerr << "Failed to load Wren code: " << scriptName << std::endl;
        return false;
    }
    
    m_ModuleName = scriptName;
    m_LoadedScripts.push_back("[" + scriptName + "]");
    
    // Cache function handles
    CacheFunctionHandles();
    
    return true;
}

bool WrenScriptComponent::Init() {
    if (m_Initialized) return true;
    
    if (!ExecuteFunction("init")) {
        std::cerr << "Wren init() function not found or execution failed" << std::endl;
    }
    
    m_Initialized = true;
    return true;
}

bool WrenScriptComponent::Update(float deltaTime) {
    if (!m_UpdateEnabled || !m_Initialized) return true;
    
    return ExecuteFunctionWithFloat("update", deltaTime);
}

bool WrenScriptComponent::Destroy() {
    if (!ExecuteFunction("destroy")) {
        std::cerr << "Wren destroy() function not found or execution failed" << std::endl;
    }
    
    ReleaseFunctionHandles();
    m_Initialized = false;
    return true;
}

bool WrenScriptComponent::Reload() {
    ReleaseFunctionHandles();
    m_Initialized = false;
    
    auto scripts = m_LoadedScripts;
    m_LoadedScripts.clear();
    
    return LoadScripts(scripts);
}

std::shared_ptr<GameObject> WrenScriptComponent::GetOwner() const {
    return m_Owner.lock();
}

bool WrenScriptComponent::HasFunction(const std::string& functionName) const {
    WrenScriptSystem& wrenSystem = WrenScriptSystem::GetInstance();
    return wrenSystem.HasFunction(functionName);
}

bool WrenScriptComponent::InvokeEvent(const std::string& eventName, const std::string& args) {
    WrenScriptSystem& wrenSystem = WrenScriptSystem::GetInstance();
    
    if (!wrenSystem.HasFunction(eventName)) {
        return false;
    }
    
    // Call the event function with arguments
    if (!args.empty()) {
        return wrenSystem.CallFunction(eventName, args);
    }
    
    return wrenSystem.CallFunction(eventName);
}

void WrenScriptComponent::SetVariable(const std::string& varName, const std::string& value) {
    WrenScriptSystem& wrenSystem = WrenScriptSystem::GetInstance();
    wrenSystem.SetGlobalVariable(varName, value);
}

std::string WrenScriptComponent::GetVariable(const std::string& varName) const {
    WrenScriptSystem& wrenSystem = WrenScriptSystem::GetInstance();
    return wrenSystem.GetGlobalVariable(varName);
}

WrenVM* WrenScriptComponent::GetVM() const {
    return WrenScriptSystem::GetInstance().GetVM();
}

void WrenScriptComponent::OnEvent(const std::string& eventName, EventCallback callback) {
    m_EventCallbacks[eventName] = callback;
}

bool WrenScriptComponent::ExecuteFunction(const std::string& functionName) {
    WrenScriptSystem& wrenSystem = WrenScriptSystem::GetInstance();
    
    if (!wrenSystem.HasFunction(functionName)) {
        return false;
    }
    
    return wrenSystem.CallFunction(functionName);
}

bool WrenScriptComponent::ExecuteFunctionWithFloat(const std::string& functionName, 
                                                    float argument) {
    WrenScriptSystem& wrenSystem = WrenScriptSystem::GetInstance();
    
    if (!wrenSystem.HasFunction(functionName)) {
        return false;
    }
    
    // Build argument string for Wren
    std::string args = std::to_string(argument);
    return wrenSystem.CallFunction(functionName, args);
}

void WrenScriptComponent::CacheFunctionHandles() {
    // Pre-cache common function handles for performance
    // This would store WrenHandles for frequently called functions
}

void WrenScriptComponent::ReleaseFunctionHandles() {
    // Release cached function handles
    m_FunctionHandles.clear();
}

// ============================================================================
// WrenScriptFactory Implementation
// ============================================================================

std::shared_ptr<WrenScriptComponent> WrenScriptFactory::CreateComponent(
    std::shared_ptr<GameObject> gameObject,
    const std::string& scriptPath) {
    
    auto component = std::make_shared<WrenScriptComponent>(gameObject);
    
    if (!component->LoadScript(scriptPath)) {
        std::cerr << "Failed to create Wren script component for: " << scriptPath << std::endl;
        return nullptr;
    }
    
    return component;
}

std::shared_ptr<WrenScriptComponent> WrenScriptFactory::CreateComponent(
    std::shared_ptr<GameObject> gameObject,
    const std::vector<std::string>& scriptPaths) {
    
    auto component = std::make_shared<WrenScriptComponent>(gameObject);
    
    if (!component->LoadScripts(scriptPaths)) {
        std::cerr << "Failed to create Wren script component with multiple scripts" << std::endl;
        return nullptr;
    }
    
    return component;
}

bool WrenScriptFactory::AttachScript(
    std::shared_ptr<WrenScriptComponent> component,
    const std::string& scriptPath) {
    
    if (!component) return false;
    
    return component->LoadScript(scriptPath);
}
