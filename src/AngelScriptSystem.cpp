#include "AngelScriptSystem.h"
#include "GameObject.h"
#include "Transform.h"
#include <iostream>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <chrono>
#include <glm/glm.hpp>

// AngelScript includes
#include "angelscript.h"

// ============================================================================
// Global Callbacks for AngelScript
// ============================================================================

static void MessageCallback(const asSMessageInfo* msg, void* param) {
    auto* system = static_cast<AngelScriptSystem*>(param);
    
    std::string message = "[";
    if (msg->type == asMSGTYPE_ERROR) {
        message += "ERROR";
    } else if (msg->type == asMSGTYPE_WARNING) {
        message += "WARNING";
    } else {
        message += "INFO";
    }
    message += "] " + std::string(msg->section) + " (" + 
               std::to_string(msg->row) + ", " + 
               std::to_string(msg->col) + "): " + 
               std::string(msg->message);

    system->GetLastError();  // Trigger error capture
}

void PrintString(asIScriptGeneric* gen) {
    std::string* str = reinterpret_cast<std::string*>(gen->GetArgAddress(0));
    std::cout << *str << std::endl;
}

// ============================================================================
// Lifecycle
// ============================================================================

AngelScriptSystem::AngelScriptSystem()
    : m_Engine(nullptr), m_Context(nullptr), m_ActiveModule(nullptr),
      m_PrintHandler(nullptr), m_ErrorHandler(nullptr), m_Initialized(false) {
}

AngelScriptSystem::~AngelScriptSystem() {
    Shutdown();
}

void AngelScriptSystem::Init() {
    if (m_Initialized) return;

    // Create the AngelScript engine
    m_Engine = asCreateScriptEngine();
    if (!m_Engine) {
        m_LastError = "Failed to create AngelScript engine";
        return;
    }

    // Set engine properties
    m_Engine->SetEngineProperty(asEP_SCRIPT_RECURSION_LEVEL, 100);
    m_Engine->SetEngineProperty(asEP_BUILD_WITHOUT_LINE_CUES, false);
    m_Engine->SetEngineProperty(asEP_OPTIMIZE_BYTECODE, m_OptimizeScripts ? 1 : 0);
    m_Engine->SetEngineProperty(asEP_INIT_GLOBAL_VARS, true);

    // Set message callback
    m_Engine->SetMessageCallback(asFUNCTION(MessageCallback), this, asCALL_CDECL);

    // Create context
    m_Context = m_Engine->CreateContext();
    if (!m_Context) {
        m_LastError = "Failed to create AngelScript context";
        m_Engine->Release();
        m_Engine = nullptr;
        return;
    }

    // Register engine functionality
    SetupCallbacks();
    RegisterTypes();

    // Create default module
    CreateModule("__default");
    SetActiveModule("__default");

    m_Initialized = true;
    std::cout << "AngelScript engine initialized (v" << ANGELSCRIPT_VERSION_STRING << ")" << std::endl;
}

void AngelScriptSystem::Shutdown() {
    if (!m_Initialized) return;

    // Discard all modules
    for (auto& [name, module] : m_Modules) {
        if (module) {
            m_Engine->DiscardModule(name.c_str());
        }
    }
    m_Modules.clear();
    m_ScriptPaths.clear();

    // Release context
    if (m_Context) {
        m_Context->Release();
        m_Context = nullptr;
    }

    // Release engine
    if (m_Engine) {
        m_Engine->Release();
        m_Engine = nullptr;
    }

    m_Initialized = false;
    std::cout << "AngelScript engine shutdown" << std::endl;
}

void AngelScriptSystem::Update(float deltaTime) {
    if (!m_Initialized) return;
    (void)deltaTime;
    // Engine updates happen per function call
}

// ============================================================================
// Script Execution
// ============================================================================

bool AngelScriptSystem::RunScript(const std::string& filepath) {
    if (!m_Initialized) {
        m_LastError = "AngelScript not initialized";
        return false;
    }

    // Read file
    std::ifstream file(filepath);
    if (!file.is_open()) {
        m_LastError = "Failed to open script: " + filepath;
        return false;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string source = buffer.str();

    if (!ExecuteString(source)) {
        return false;
    }

    // Store script path for potential hot-reload
    m_ScriptPaths[m_ActiveModule->GetName()] = filepath;
    return true;
}

bool AngelScriptSystem::ExecuteString(const std::string& source) {
    if (!m_Initialized) {
        m_LastError = "AngelScript not initialized";
        return false;
    }

    if (!m_ActiveModule) {
        m_LastError = "No active module set";
        return false;
    }

    // Add script section
    auto result = m_ActiveModule->AddScriptSection("script", source.c_str());
    if (result < 0) {
        m_LastError = "Failed to add script section";
        return false;
    }

    // Build module
    result = m_ActiveModule->Build();
    if (result < 0) {
        m_LastError = "Failed to build module";
        return false;
    }

    return true;
}

// ============================================================================
// Type Registration
// ============================================================================

void AngelScriptSystem::RegisterTypes() {
    if (!m_Engine) return;

    RegisterEngineTypes();
    RegisterGameObjectTypes();
    RegisterMathTypes();
    RegisterPhysicsTypes();
}

bool AngelScriptSystem::HasType(const std::string& typeName) const {
    if (!m_Engine) return false;
    return m_Engine->GetTypeIdByDecl(typeName.c_str()) > 0;
}

void AngelScriptSystem::RegisterEngineTypes() {
    if (!m_Engine) return;

    // Register string type
    m_Engine->RegisterGlobalFunction("void print(const string& in)",
                                    asFUNCTION(PrintString), asCALL_GENERIC);

    // Register basic math functions
    m_Engine->RegisterGlobalFunction("float sin(float)", 
                                    asFUNCTION(sinf), asCALL_CDECL);
    m_Engine->RegisterGlobalFunction("float cos(float)", 
                                    asFUNCTION(cosf), asCALL_CDECL);
    m_Engine->RegisterGlobalFunction("float sqrt(float)", 
                                    asFUNCTION(sqrtf), asCALL_CDECL);
    m_Engine->RegisterGlobalFunction("float abs(float)", 
                                    asFUNCTION(fabsf), asCALL_CDECL);
}

void AngelScriptSystem::RegisterGameObjectTypes() {
    if (!m_Engine) return;

    // Register basic object type handling
    // This would be expanded based on GameObject interface
    m_Engine->RegisterObjectType("Vec3", sizeof(glm::vec3), asOBJ_VALUE | asOBJ_POD | asOBJ_APP_CLASS_ALLINTS);
    m_Engine->RegisterObjectType("Transform", 0, asOBJ_REF);
    m_Engine->RegisterObjectType("GameObject", 0, asOBJ_REF);
}

void AngelScriptSystem::RegisterMathTypes() {
    if (!m_Engine) return;

    // Register Vec3 struct
    m_Engine->RegisterObjectProperty("Vec3", "float x", asOFFSET(glm::vec3, x));
    m_Engine->RegisterObjectProperty("Vec3", "float y", asOFFSET(glm::vec3, y));
    m_Engine->RegisterObjectProperty("Vec3", "float z", asOFFSET(glm::vec3, z));
}

void AngelScriptSystem::RegisterPhysicsTypes() {
    if (!m_Engine) return;

    // Physics type registrations would go here
}

// ============================================================================
// Function Calling
// ============================================================================

std::any AngelScriptSystem::CallFunction(const std::string& functionName,
                                        const std::vector<std::any>& args) {
    if (!m_Initialized || !m_Context) {
        m_LastError = "AngelScript not initialized";
        return std::any();
    }

    auto start = std::chrono::high_resolution_clock::now();

    // Get function
    asIScriptFunction* func = m_ActiveModule->GetFunctionByName(functionName.c_str());
    if (!func) {
        m_LastError = "Function not found: " + functionName;
        return std::any();
    }

    // Prepare context
    m_Context->Prepare(func);

    // Set arguments (simplified - would need proper type handling)
    (void)args;  // Suppress unused warning for now

    // Execute
    auto result = m_Context->Execute();
    
    auto end = std::chrono::high_resolution_clock::now();
    m_LastExecutionTime = std::chrono::duration<double>(end - start).count();

    if (result != asEXECUTION_FINISHED) {
        if (result == asEXECUTION_EXCEPTION) {
            m_LastError = m_Context->GetExceptionString();
        }
        return std::any();
    }

    return std::any();
}

std::any AngelScriptSystem::CallMethod(const std::string& objectName,
                                       const std::string& methodName,
                                       const std::vector<std::any>& args) {
    (void)objectName;
    (void)methodName;
    (void)args;
    // Implementation would go here
    return std::any();
}

// ============================================================================
// Variable Management
// ============================================================================

void AngelScriptSystem::SetGlobalVariable(const std::string& varName, const std::any& value) {
    if (!m_Engine) return;
    (void)varName;
    (void)value;
    // Implementation would go here
}

std::string AngelScriptSystem::GetGlobalVariable(const std::string& varName) const {
    if (!m_Engine) return "";
    (void)varName;
    // Implementation would go here
    return "";
}

// ============================================================================
// Module Management
// ============================================================================

bool AngelScriptSystem::CreateModule(const std::string& moduleName) {
    if (!m_Engine) return false;

    asIScriptModule* module = m_Engine->GetModule(moduleName.c_str());
    if (!module) {
        module = m_Engine->GetModule(moduleName.c_str(), asGM_CREATE_IF_NOT_EXISTS);
    }

    if (!module) {
        m_LastError = "Failed to create module: " + moduleName;
        return false;
    }

    m_Modules[moduleName] = module;
    return true;
}

bool AngelScriptSystem::BuildModule(const std::string& moduleName) {
    auto it = m_Modules.find(moduleName);
    if (it == m_Modules.end() || !it->second) {
        m_LastError = "Module not found: " + moduleName;
        return false;
    }

    auto result = it->second->Build();
    return result >= 0;
}

void AngelScriptSystem::DiscardModule(const std::string& moduleName) {
    auto it = m_Modules.find(moduleName);
    if (it != m_Modules.end()) {
        if (m_Engine && it->second) {
            m_Engine->DiscardModule(moduleName.c_str());
        }
        m_Modules.erase(it);
        m_ScriptPaths.erase(moduleName);
    }
}

asIScriptModule* AngelScriptSystem::GetModule(const std::string& moduleName) const {
    auto it = m_Modules.find(moduleName);
    return it != m_Modules.end() ? it->second : nullptr;
}

void AngelScriptSystem::SetActiveModule(const std::string& moduleName) {
    auto it = m_Modules.find(moduleName);
    if (it != m_Modules.end()) {
        m_ActiveModule = it->second;
    } else {
        m_LastError = "Module not found: " + moduleName;
    }
}

bool AngelScriptSystem::HasFunction(const std::string& functionName) const {
    if (!m_ActiveModule) return false;
    auto* func = m_ActiveModule->GetFunctionByName(functionName.c_str());
    return func != nullptr;
}

// ============================================================================
// Hot-Reload & Optimization
// ============================================================================

void AngelScriptSystem::ReloadScript(const std::string& filepath) {
    if (!m_Initialized) return;

    // Find module with this filepath
    std::string moduleToReload;
    for (const auto& [name, path] : m_ScriptPaths) {
        if (path == filepath) {
            moduleToReload = name;
            break;
        }
    }

    if (moduleToReload.empty()) {
        m_LastError = "Script not loaded: " + filepath;
        return;
    }

    // Discard and recreate module
    DiscardModule(moduleToReload);
    CreateModule(moduleToReload);
    SetActiveModule(moduleToReload);
    RunScript(filepath);
}

uint64_t AngelScriptSystem::GetMemoryUsage() const {
    if (!m_Engine) return 0;
    
    // AngelScript doesn't directly expose memory usage
    // Return approximate based on compilation statistics
    return 2000000;  // ~2MB baseline for engine
}

void AngelScriptSystem::ForceGarbageCollection() {
    if (!m_Engine) return;
    m_Engine->GarbageCollect(asGC_FULL_CYCLE);
}

void AngelScriptSystem::ClearState() {
    if (!m_Initialized) return;

    for (auto& [name, module] : m_Modules) {
        if (module && m_Engine) {
            m_Engine->DiscardModule(name.c_str());
        }
    }
    m_Modules.clear();
    m_ScriptPaths.clear();

    CreateModule("__default");
    SetActiveModule("__default");

    m_LastError = "";
    m_LastExecutionTime = 0.0;
}

// ============================================================================
// Compilation Statistics
// ============================================================================

AngelScriptSystem::CompileStats AngelScriptSystem::GetCompileStats() const {
    CompileStats stats;
    stats.totalModules = m_Modules.size();
    
    for (const auto& [name, module] : m_Modules) {
        if (module) {
            stats.totalFunctions += module->GetFunctionCount();
            // Additional counting would go here
        }
    }

    stats.memoryUsed = GetMemoryUsage();
    return stats;
}

// ============================================================================
// Callbacks & Helper Functions
// ============================================================================

void AngelScriptSystem::SetupCallbacks() {
    if (!m_Engine) return;

    // Setup any necessary callbacks here
}

void AngelScriptSystem::LogMessage(const std::string& message, bool isError) {
    if (isError) {
        m_LastError = message;
        if (m_ErrorHandler) {
            m_ErrorHandler(message);
        }
    } else {
        if (m_PrintHandler) {
            m_PrintHandler(message);
        }
    }
}

int AngelScriptSystem::ExecuteFunction(asIScriptContext* ctx, const std::string& funcName) {
    if (!ctx) return -1;
    
    asIScriptFunction* func = m_ActiveModule->GetFunctionByName(funcName.c_str());
    if (!func) {
        return -1;
    }

    ctx->Prepare(func);
    return ctx->Execute();
}
