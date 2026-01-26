#include "ScriptLanguageRegistry.h"
#include "LuaScriptSystem.h"
#include "LuaJitScriptSystem.h"
#include "WrenScriptSystem.h"
#include "PythonScriptSystem.h"
#include "CSharpScriptSystem.h"
#include "CustomScriptSystem.h"
#include "TypeScriptScriptSystem.h"
#include "RustScriptSystem.h"
#include "SquirrelScriptSystem.h"
#include "GoScriptSystem.h"
#include "GDScriptSystem.h"
#include <iostream>
#include <algorithm>

ScriptLanguageRegistry::ScriptLanguageRegistry()
{
}

ScriptLanguageRegistry::~ScriptLanguageRegistry()
{
    Shutdown();
}

void ScriptLanguageRegistry::Init()
{
    InitializeExtensionMap();
    RegisterDefaultSystems();

    // Initialize all registered systems
    for (auto& pair : m_Systems) {
        try {
            pair.second->Init();
            std::cout << "Initialized: " << pair.second->GetLanguageName() << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Failed to initialize: " << pair.second->GetLanguageName()
                      << " - " << e.what() << std::endl;
        }
    }

    std::cout << "ScriptLanguageRegistry initialized with " << m_Systems.size()
              << " language systems" << std::endl;
}

void ScriptLanguageRegistry::Shutdown()
{
    // Shutdown all systems in reverse order
    for (auto it = m_Systems.rbegin(); it != m_Systems.rend(); ++it) {
        try {
            it->second->Shutdown();
        } catch (const std::exception& e) {
            std::cerr << "Error shutting down script system: " << e.what() << std::endl;
        }
    }
    m_Systems.clear();
    std::cout << "ScriptLanguageRegistry shutdown" << std::endl;
}

void ScriptLanguageRegistry::Update(float deltaTime)
{
    for (auto& pair : m_Systems) {
        try {
            pair.second->Update(deltaTime);
        } catch (const std::exception& e) {
            std::cerr << "Error updating " << pair.second->GetLanguageName()
                      << ": " << e.what() << std::endl;
        }
    }
}

bool ScriptLanguageRegistry::ExecuteScript(const std::string& filepath)
{
    ScriptLanguage language = DetectLanguage(filepath);
    return ExecuteScript(filepath, language);
}

bool ScriptLanguageRegistry::ExecuteScript(const std::string& filepath, ScriptLanguage language)
{
    IScriptSystem* system = GetScriptSystem(language);
    if (!system) {
        std::string error = "Script language not available: " + GetLanguageName(language);
        if (m_ErrorCallback) {
            m_ErrorCallback(language, error);
        }
        return false;
    }

    try {
        bool result = system->RunScript(filepath);
        if (result && m_SuccessCallback) {
            m_SuccessCallback(language, filepath);
        } else if (!result && m_ErrorCallback && system->HasErrors()) {
            m_ErrorCallback(language, system->GetLastError());
        }
        return result;
    } catch (const std::exception& e) {
        std::string error = std::string("Exception: ") + e.what();
        if (m_ErrorCallback) {
            m_ErrorCallback(language, error);
        }
        return false;
    }
}

bool ScriptLanguageRegistry::ExecuteString(const std::string& source, ScriptLanguage language)
{
    IScriptSystem* system = GetScriptSystem(language);
    if (!system) {
        std::string error = "Script language not available: " + GetLanguageName(language);
        if (m_ErrorCallback) {
            m_ErrorCallback(language, error);
        }
        return false;
    }

    try {
        bool result = system->ExecuteString(source);
        if (result && m_SuccessCallback) {
            m_SuccessCallback(language, "<string>");
        } else if (!result && m_ErrorCallback && system->HasErrors()) {
            m_ErrorCallback(language, system->GetLastError());
        }
        return result;
    } catch (const std::exception& e) {
        std::string error = std::string("Exception: ") + e.what();
        if (m_ErrorCallback) {
            m_ErrorCallback(language, error);
        }
        return false;
    }
}

IScriptSystem* ScriptLanguageRegistry::GetScriptSystem(ScriptLanguage language)
{
    auto it = m_Systems.find(language);
    return it != m_Systems.end() ? it->second.get() : nullptr;
}

void ScriptLanguageRegistry::RegisterScriptSystem(ScriptLanguage language,
                                                   std::shared_ptr<IScriptSystem> system)
{
    if (!system) {
        std::cerr << "Cannot register null script system" << std::endl;
        return;
    }

    m_Systems[language] = system;
    std::cout << "Registered script system: " << system->GetLanguageName() << std::endl;
}

std::any ScriptLanguageRegistry::CallFunction(const std::string& functionName,
                                              const std::vector<std::any>& args)
{
    // Try each system in a preferred order
    std::vector<ScriptLanguage> order = {
        ScriptLanguage::Lua,
        ScriptLanguage::Wren,
        ScriptLanguage::Python,
        ScriptLanguage::CSharp,
        ScriptLanguage::TypeScript,
        ScriptLanguage::Rust,
        ScriptLanguage::Squirrel,
        ScriptLanguage::Custom
    };

    for (ScriptLanguage lang : order) {
        IScriptSystem* system = GetScriptSystem(lang);
        if (system && system->HasType(functionName)) {
            return system->CallFunction(functionName, args);
        }
    }

    std::cerr << "Function not found in any language system: " << functionName << std::endl;
    return std::any();
}

std::any ScriptLanguageRegistry::CallFunction(ScriptLanguage language,
                                              const std::string& functionName,
                                              const std::vector<std::any>& args)
{
    IScriptSystem* system = GetScriptSystem(language);
    if (!system) {
        std::cerr << "Script language not initialized: " << GetLanguageName(language) << std::endl;
        return std::any();
    }

    return system->CallFunction(functionName, args);
}

ScriptLanguage ScriptLanguageRegistry::DetectLanguage(const std::string& filepath) const
{
    // Extract file extension
    size_t dotPos = filepath.rfind('.');
    if (dotPos == std::string::npos) {
        return ScriptLanguage::Custom;
    }

    std::string extension = filepath.substr(dotPos);
    return GetLanguageFromExtension(extension);
}

ScriptLanguage ScriptLanguageRegistry::GetLanguageFromExtension(const std::string& extension) const
{
    auto it = m_ExtensionMap.find(extension);
    return it != m_ExtensionMap.end() ? it->second : ScriptLanguage::Custom;
}

std::vector<ScriptLanguage> ScriptLanguageRegistry::GetSupportedLanguages() const
{
    std::vector<ScriptLanguage> languages;
    for (const auto& pair : m_Systems) {
        languages.push_back(pair.first);
    }
    return languages;
}

std::string ScriptLanguageRegistry::GetLanguageName(ScriptLanguage language) const
{
    IScriptSystem* system = GetScriptSystem(language);
    if (system) {
        return system->GetLanguageName();
    }

    switch (language) {
        case ScriptLanguage::Lua: return "Lua";
        case ScriptLanguage::LuaJIT: return "LuaJIT";
        case ScriptLanguage::Wren: return "Wren";
        case ScriptLanguage::Python: return "Python";
        case ScriptLanguage::CSharp: return "C#";
        case ScriptLanguage::Custom: return "Custom";
        case ScriptLanguage::TypeScript: return "TypeScript/JavaScript";
        case ScriptLanguage::Rust: return "Rust";
        case ScriptLanguage::Squirrel: return "Squirrel";
        case ScriptLanguage::Go: return "Go";
        case ScriptLanguage::GDScript: return "GDScript";
        default: return "Unknown";
    }
}

std::string ScriptLanguageRegistry::GetFileExtension(ScriptLanguage language) const
{
    IScriptSystem* system = GetScriptSystem(language);
    if (system) {
        return system->GetFileExtension();
    }
    return "";
}

bool ScriptLanguageRegistry::IsLanguageReady(ScriptLanguage language) const
{
    return GetScriptSystem(language) != nullptr;
}

uint64_t ScriptLanguageRegistry::GetTotalMemoryUsage() const
{
    uint64_t total = 0;
    for (const auto& pair : m_Systems) {
        total += pair.second->GetMemoryUsage();
    }
    return total;
}

double ScriptLanguageRegistry::GetLastExecutionTime(ScriptLanguage language) const
{
    IScriptSystem* system = GetScriptSystem(language);
    return system ? system->GetLastExecutionTime() : 0.0;
}

bool ScriptLanguageRegistry::HasErrors() const
{
    for (const auto& pair : m_Systems) {
        if (pair.second->HasErrors()) {
            return true;
        }
    }
    return false;
}

std::map<std::string, std::string> ScriptLanguageRegistry::GetAllErrors() const
{
    std::map<std::string, std::string> errors;
    for (const auto& pair : m_Systems) {
        if (pair.second->HasErrors()) {
            errors[pair.second->GetLanguageName()] = pair.second->GetLastError();
        }
    }
    return errors;
}

void ScriptLanguageRegistry::ClearErrors()
{
    // Note: Individual script systems don't have a ClearError method yet
    // Could be added to IScriptSystem interface if needed
}

bool ScriptLanguageRegistry::ReloadScript(const std::string& filepath)
{
    ScriptLanguage language = DetectLanguage(filepath);
    IScriptSystem* system = GetScriptSystem(language);

    if (!system) {
        std::cerr << "Cannot reload script: language not available: " << filepath << std::endl;
        return false;
    }

    if (!system->SupportsHotReload()) {
        std::cerr << "Language does not support hot-reload: " << GetLanguageName(language) << std::endl;
        return false;
    }

    system->ReloadScript(filepath);
    return true;
}

bool ScriptLanguageRegistry::SupportsHotReload(ScriptLanguage language) const
{
    IScriptSystem* system = GetScriptSystem(language);
    return system ? system->SupportsHotReload() : false;
}

void ScriptLanguageRegistry::SetErrorCallback(ErrorCallback callback)
{
    m_ErrorCallback = callback;
}

void ScriptLanguageRegistry::SetSuccessCallback(SuccessCallback callback)
{
    m_SuccessCallback = callback;
}

void ScriptLanguageRegistry::InitializeExtensionMap()
{
    m_ExtensionMap[".lua"] = ScriptLanguage::Lua;
    m_ExtensionMap[".wren"] = ScriptLanguage::Wren;
    m_ExtensionMap[".py"] = ScriptLanguage::Python;
    m_ExtensionMap[".cs"] = ScriptLanguage::CSharp;
    m_ExtensionMap[".js"] = ScriptLanguage::TypeScript;
    m_ExtensionMap[".ts"] = ScriptLanguage::TypeScript;
    m_ExtensionMap[".dll"] = ScriptLanguage::Rust;
    m_ExtensionMap[".so"] = ScriptLanguage::Rust;
    m_ExtensionMap[".dylib"] = ScriptLanguage::Rust;
    m_ExtensionMap[".nut"] = ScriptLanguage::Squirrel;
    m_ExtensionMap[".go"] = ScriptLanguage::Go;
    m_ExtensionMap[".gd"] = ScriptLanguage::GDScript;
    m_ExtensionMap[".asm"] = ScriptLanguage::Custom;
    m_ExtensionMap[".bc"] = ScriptLanguage::Custom;
}

void ScriptLanguageRegistry::RegisterDefaultSystems()
{
    // Register all available script systems
    // In practice, some may be optional based on build configuration
    
    // Register standard Lua
    RegisterScriptSystem(ScriptLanguage::Lua,
                        std::make_shared<LuaScriptSystem>());
    
    // Register LuaJIT for 10x+ performance
    // This uses the same .lua extension but with JIT compilation
    RegisterScriptSystem(ScriptLanguage::LuaJIT,
                        std::make_shared<LuaJitScriptSystem>());
    
    RegisterScriptSystem(ScriptLanguage::Wren,
                        std::make_shared<WrenScriptSystem>());
    
    RegisterScriptSystem(ScriptLanguage::Python,
                        std::make_shared<PythonScriptSystem>());
    
    RegisterScriptSystem(ScriptLanguage::CSharp,
                        std::make_shared<CSharpScriptSystem>());
    
    RegisterScriptSystem(ScriptLanguage::Custom,
                        std::make_shared<CustomScriptSystem>());
    
    RegisterScriptSystem(ScriptLanguage::TypeScript,
                        std::make_shared<TypeScriptScriptSystem>());
    
    RegisterScriptSystem(ScriptLanguage::Rust,
                        std::make_shared<RustScriptSystem>());
    
    RegisterScriptSystem(ScriptLanguage::Squirrel,
                        std::make_shared<SquirrelScriptSystem>());
    
    RegisterScriptSystem(ScriptLanguage::Go,
                        std::make_shared<GoScriptSystem>());
    
    RegisterScriptSystem(ScriptLanguage::GDScript,
                        std::make_shared<GDScriptSystem>());
}
