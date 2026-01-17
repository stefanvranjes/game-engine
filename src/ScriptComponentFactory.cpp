#include "ScriptComponentFactory.h"
#include "ScriptLanguageRegistry.h"
#include "GameObject.h"
#include <iostream>
#include <algorithm>
#include <cctype>

std::shared_ptr<ScriptComponent> ScriptComponentFactory::CreateScriptComponent(
    const std::string& scriptPath,
    std::shared_ptr<GameObject> gameObject)
{
    ScriptLanguage language = DetectLanguage(scriptPath);
    return CreateScriptComponent(scriptPath, gameObject, language);
}

std::shared_ptr<ScriptComponent> ScriptComponentFactory::CreateScriptComponent(
    const std::string& scriptPath,
    std::shared_ptr<GameObject> gameObject,
    ScriptLanguage language)
{
    if (!gameObject) {
        std::cerr << "Cannot create ScriptComponent: GameObject is null" << std::endl;
        return nullptr;
    }

    auto component = std::make_shared<ScriptComponent>(gameObject);
    
    IScriptSystem* system = ScriptLanguageRegistry::GetInstance().GetScriptSystem(language);
    if (!system) {
        std::cerr << "Script language not available: "
                  << ScriptLanguageRegistry::GetInstance().GetLanguageName(language) << std::endl;
        return nullptr;
    }

    if (!system->RunScript(scriptPath)) {
        std::cerr << "Failed to load script: " << scriptPath << std::endl;
        if (system->HasErrors()) {
            std::cerr << "Error: " << system->GetLastError() << std::endl;
        }
        return nullptr;
    }

    return component;
}

std::shared_ptr<ScriptComponent> ScriptComponentFactory::CreateMultiLanguageComponent(
    std::shared_ptr<GameObject> gameObject)
{
    if (!gameObject) {
        std::cerr << "Cannot create MultiLanguageScriptComponent: GameObject is null" << std::endl;
        return nullptr;
    }

    return std::make_shared<MultiLanguageScriptComponent>(gameObject);
}

std::shared_ptr<ScriptComponent> ScriptComponentFactory::CreateScriptComponentFromString(
    const std::string& source,
    ScriptLanguage language,
    std::shared_ptr<GameObject> gameObject)
{
    if (!gameObject) {
        std::cerr << "Cannot create ScriptComponent: GameObject is null" << std::endl;
        return nullptr;
    }

    auto component = std::make_shared<ScriptComponent>(gameObject);

    IScriptSystem* system = ScriptLanguageRegistry::GetInstance().GetScriptSystem(language);
    if (!system) {
        std::cerr << "Script language not available: "
                  << ScriptLanguageRegistry::GetInstance().GetLanguageName(language) << std::endl;
        return nullptr;
    }

    if (!system->ExecuteString(source)) {
        std::cerr << "Failed to execute script source" << std::endl;
        if (system->HasErrors()) {
            std::cerr << "Error: " << system->GetLastError() << std::endl;
        }
        return nullptr;
    }

    return component;
}

std::shared_ptr<ScriptComponent> ScriptComponentFactory::CloneScriptComponent(
    const std::shared_ptr<ScriptComponent>& source,
    std::shared_ptr<GameObject> newGameObject)
{
    if (!source || !newGameObject) {
        std::cerr << "Cannot clone ScriptComponent: invalid source or target" << std::endl;
        return nullptr;
    }

    // Create new component and copy properties
    auto cloned = std::make_shared<ScriptComponent>(newGameObject);
    
    // Copy enabled state and other properties
    // Note: Actual implementation would copy script paths and reload them
    
    return cloned;
}

ScriptLanguage ScriptComponentFactory::DetectLanguage(const std::string& scriptPath)
{
    return ScriptLanguageRegistry::GetInstance().DetectLanguage(scriptPath);
}

bool ScriptComponentFactory::IsLanguageSupported(ScriptLanguage language)
{
    return ScriptLanguageRegistry::GetInstance().IsLanguageReady(language);
}

std::map<std::string, ScriptLanguage> ScriptComponentFactory::GetSupportedExtensions()
{
    std::map<std::string, ScriptLanguage> extensions;
    
    const auto& supported = ScriptLanguageRegistry::GetInstance().GetSupportedLanguages();
    for (ScriptLanguage lang : supported) {
        std::string ext = ScriptLanguageRegistry::GetInstance().GetFileExtension(lang);
        if (!ext.empty()) {
            extensions[ext] = lang;
        }
    }
    
    return extensions;
}

// ============================================================================
// MultiLanguageScriptComponent Implementation
// ============================================================================

MultiLanguageScriptComponent::MultiLanguageScriptComponent(std::shared_ptr<GameObject> gameObject)
    : ScriptComponent(gameObject)
{
}

bool MultiLanguageScriptComponent::AddScript(const std::string& scriptPath)
{
    ScriptLanguage language = ScriptComponentFactory::DetectLanguage(scriptPath);
    return AddScript(scriptPath, language);
}

bool MultiLanguageScriptComponent::AddScript(const std::string& source, ScriptLanguage language)
{
    auto& registry = ScriptLanguageRegistry::GetInstance();
    IScriptSystem* system = registry.GetScriptSystem(language);
    
    if (!system) {
        std::cerr << "Script language not available: " << registry.GetLanguageName(language) << std::endl;
        return false;
    }

    bool success = system->ExecuteString(source);
    if (success) {
        m_AttachedScripts.push_back(source);
        m_ScriptLanguages[source] = language;
        std::cout << "Added script in " << system->GetLanguageName() << std::endl;
    } else {
        std::cerr << "Failed to add script: " << system->GetLastError() << std::endl;
    }
    
    return success;
}

bool MultiLanguageScriptComponent::RemoveScript(const std::string& scriptPath)
{
    auto it = std::find(m_AttachedScripts.begin(), m_AttachedScripts.end(), scriptPath);
    if (it != m_AttachedScripts.end()) {
        m_AttachedScripts.erase(it);
        m_ScriptLanguages.erase(scriptPath);
        return true;
    }
    return false;
}

std::any MultiLanguageScriptComponent::CallFunction(const std::string& functionName,
                                                     const std::vector<std::any>& args)
{
    return ScriptLanguageRegistry::GetInstance().CallFunction(functionName, args);
}

std::any MultiLanguageScriptComponent::CallFunction(ScriptLanguage language,
                                                     const std::string& functionName,
                                                     const std::vector<std::any>& args)
{
    return ScriptLanguageRegistry::GetInstance().CallFunction(language, functionName, args);
}

void MultiLanguageScriptComponent::Init()
{
    // Call Init in all attached scripts if they support it
    for (const auto& script : m_AttachedScripts) {
        std::vector<std::any> args;
        CallFunction(script + "_init", args);
    }
}

void MultiLanguageScriptComponent::Update(float deltaTime)
{
    // Call Update in all attached scripts
    std::vector<std::any> args = {deltaTime};
    for (const auto& script : m_AttachedScripts) {
        std::vector<std::any> dtArgs = {deltaTime};
        CallFunction(script + "_update", dtArgs);
    }
}

void MultiLanguageScriptComponent::OnDestroy()
{
    // Call OnDestroy in all attached scripts
    for (const auto& script : m_AttachedScripts) {
        std::vector<std::any> args;
        CallFunction(script + "_destroy", args);
    }
}

void MultiLanguageScriptComponent::OnCollisionEnter(std::shared_ptr<GameObject> other)
{
    // Call OnCollisionEnter in all attached scripts
    std::vector<std::any> args = {other};
    for (const auto& script : m_AttachedScripts) {
        CallFunction(script + "_collision_enter", args);
    }
}

void MultiLanguageScriptComponent::OnCollisionExit(std::shared_ptr<GameObject> other)
{
    // Call OnCollisionExit in all attached scripts
    std::vector<std::any> args = {other};
    for (const auto& script : m_AttachedScripts) {
        CallFunction(script + "_collision_exit", args);
    }
}
