#include "ScriptComponent.h"
#include "GameObject.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include "PythonScriptSystem.h"

ScriptComponent::ScriptComponent(std::weak_ptr<GameObject> owner) : m_Owner(owner) {
}

ScriptComponent::~ScriptComponent() {
    // Cleanup Lua references if necessary
}

void ScriptComponent::LoadScript(const std::string& filepath) {
    m_ScriptPath = filepath;
    // For now, we just run the file. In a real component system, we might look for specific functions
    // inside the script like "OnCreate" or "OnUpdate".
    // A simple approach: The script executes immediate global code on load.
    // Ideally, to support multiple objects, the script should return a table or class.
    
    // For this iteration, let's execute the script immediately.
    PythonScriptSystem::GetInstance().RunScript(filepath);
}

void ScriptComponent::Init() {
    // Call "Init" function in script if it exists
    // Implementation requires table management (omitted for Step 1 simplicity)
}

void ScriptComponent::Update(float deltaTime) {
    // Call "Update" function in script if it exists
    // Implementation requires table management (omitted for Step 1 simplicity)
}

void ScriptComponent::Reload() {
    LoadScript(m_ScriptPath);
}
