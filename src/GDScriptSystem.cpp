#include "GDScriptSystem.h"
#include "GameObject.h"
#include "Transform.h"
#include <iostream>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <chrono>
#include <cstring>

// GDScript C API mock implementation
// In a real implementation, this would use the actual GDScript C API from Godot
namespace GDScriptAPI {
    struct Context {
        std::unordered_map<std::string, std::string> scripts;
        std::unordered_map<std::string, std::any> globals;
        bool initialized = false;
    };

    struct VM {
        Context* context = nullptr;
        uint64_t memoryUsage = 0;
    };

    GDScriptContext CreateContext() {
        auto ctx = new Context();
        ctx->initialized = true;
        return ctx;
    }

    void DestroyContext(GDScriptContext ctx) {
        delete static_cast<Context*>(ctx);
    }

    GDScriptVM CreateVM(GDScriptContext ctx) {
        auto vm = new VM();
        vm->context = static_cast<Context*>(ctx);
        return vm;
    }

    void DestroyVM(GDScriptVM vm) {
        delete static_cast<VM*>(vm);
    }

    bool LoadScript(GDScriptVM vm, const std::string& filepath, const std::string& source) {
        auto v = static_cast<VM*>(vm);
        if (!v || !v->context) return false;
        v->context->scripts[filepath] = source;
        v->memoryUsage += source.size();
        return true;
    }

    bool ExecuteScript(GDScriptVM vm, const std::string& filepath) {
        auto v = static_cast<VM*>(vm);
        if (!v || !v->context) return false;
        auto it = v->context->scripts.find(filepath);
        if (it == v->context->scripts.end()) {
            return false;
        }
        // In real implementation, parse and execute GDScript bytecode
        // For now, we simulate successful execution
        return true;
    }

    bool ExecuteString(GDScriptVM vm, const std::string& source) {
        auto v = static_cast<VM*>(vm);
        if (!v || !v->context) return false;
        // In real implementation, parse and execute GDScript source
        return true;
    }

    std::any CallFunction(GDScriptVM vm, const std::string& functionName, const std::vector<std::any>& args) {
        auto v = static_cast<VM*>(vm);
        if (!v || !v->context) return std::any();
        // In real implementation, lookup and call function in GDScript
        return std::any();
    }

    uint64_t GetMemoryUsage(GDScriptVM vm) {
        auto v = static_cast<VM*>(vm);
        if (!v) return 0;
        return v->memoryUsage;
    }
};

// ============================================================================
// Lifecycle
// ============================================================================

GDScriptSystem::GDScriptSystem() {
}

GDScriptSystem::~GDScriptSystem() {
    Shutdown();
}

void GDScriptSystem::Init() {
    if (m_Initialized) return;

    try {
        std::cout << "Initializing GDScriptSystem..." << std::endl;

        // Create GDScript context
        m_Context = GDScriptAPI::CreateContext();
        if (!m_Context) {
            throw std::runtime_error("Failed to create GDScript context");
        }

        // Create GDScript VM
        m_VM = GDScriptAPI::CreateVM(m_Context);
        if (!m_VM) {
            throw std::runtime_error("Failed to create GDScript VM");
        }

        // Setup built-in functions and classes
        SetupBuiltins();

        // Load standard library
        LoadStandardLibrary();

        // Register engine types
        RegisterTypes();

        m_Initialized = true;
        m_LastError.clear();

        std::cout << "GDScriptSystem initialized successfully" << std::endl;
    }
    catch (const std::exception& e) {
        m_LastError = std::string("Initialization error: ") + e.what();
        std::cerr << m_LastError << std::endl;
        m_Initialized = false;
    }
}

void GDScriptSystem::Shutdown() {
    if (!m_Initialized) return;

    try {
        std::cout << "Shutting down GDScriptSystem..." << std::endl;

        m_SignalCallbacks.clear();
        m_BoundFunctions.clear();
        m_LoadedScripts.clear();

        if (m_VM) {
            GDScriptAPI::DestroyVM(m_VM);
            m_VM = nullptr;
        }

        if (m_Context) {
            GDScriptAPI::DestroyContext(m_Context);
            m_Context = nullptr;
        }

        m_Initialized = false;
        std::cout << "GDScriptSystem shut down successfully" << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Error during shutdown: " << e.what() << std::endl;
    }
}

void GDScriptSystem::Update(float deltaTime) {
    if (!m_Initialized) return;

    // Call _process callbacks for loaded scripts
    // This would iterate through loaded GDScript objects and call their _process methods
    // For now, this is a placeholder for future integration
}

// ============================================================================
// Script Execution
// ============================================================================

bool GDScriptSystem::RunScript(const std::string& filepath) {
    if (!m_Initialized) {
        m_LastError = "GDScriptSystem not initialized";
        return false;
    }

    try {
        auto start = std::chrono::high_resolution_clock::now();

        // Check if file exists
        if (!std::filesystem::exists(filepath)) {
            m_LastError = std::string("Script file not found: ") + filepath;
            return false;
        }

        // Read script file
        std::ifstream file(filepath);
        if (!file.is_open()) {
            m_LastError = std::string("Failed to open script: ") + filepath;
            return false;
        }

        std::stringstream buffer;
        buffer << file.rdbuf();
        file.close();

        std::string source = buffer.str();

        // Load and execute script
        if (!GDScriptAPI::LoadScript(m_VM, filepath, source)) {
            m_LastError = std::string("Failed to load script: ") + filepath;
            return false;
        }

        if (!GDScriptAPI::ExecuteScript(m_VM, filepath)) {
            m_LastError = std::string("Failed to execute script: ") + filepath;
            return false;
        }

        m_LoadedScripts[filepath] = source;
        m_LastError.clear();

        auto end = std::chrono::high_resolution_clock::now();
        m_LastExecutionTime = std::chrono::duration<double, std::milli>(end - start).count();

        std::cout << "GDScript loaded: " << filepath << " (" << m_LastExecutionTime << "ms)" << std::endl;
        return true;
    }
    catch (const std::exception& e) {
        m_LastError = std::string("Script execution error: ") + e.what();
        std::cerr << m_LastError << std::endl;
        return false;
    }
}

bool GDScriptSystem::ExecuteString(const std::string& source) {
    if (!m_Initialized) {
        m_LastError = "GDScriptSystem not initialized";
        return false;
    }

    try {
        auto start = std::chrono::high_resolution_clock::now();

        if (!GDScriptAPI::ExecuteString(m_VM, source)) {
            m_LastError = "Failed to execute GDScript source";
            return false;
        }

        m_LastError.clear();

        auto end = std::chrono::high_resolution_clock::now();
        m_LastExecutionTime = std::chrono::duration<double, std::milli>(end - start).count();

        return true;
    }
    catch (const std::exception& e) {
        m_LastError = std::string("Execution error: ") + e.what();
        std::cerr << m_LastError << std::endl;
        return false;
    }
}

// ============================================================================
// Function Calling
// ============================================================================

std::any GDScriptSystem::CallFunction(const std::string& functionName,
                                       const std::vector<std::any>& args) {
    if (!m_Initialized) {
        m_LastError = "GDScriptSystem not initialized";
        return std::any();
    }

    try {
        auto start = std::chrono::high_resolution_clock::now();

        // Check if it's a bound C++ function
        auto it = m_BoundFunctions.find(functionName);
        if (it != m_BoundFunctions.end()) {
            return it->second(args);
        }

        // Otherwise, call GDScript function
        std::any result = GDScriptAPI::CallFunction(m_VM, functionName, args);

        auto end = std::chrono::high_resolution_clock::now();
        m_LastExecutionTime = std::chrono::duration<double, std::milli>(end - start).count();

        return result;
    }
    catch (const std::exception& e) {
        m_LastError = std::string("Function call error: ") + e.what();
        std::cerr << m_LastError << std::endl;
        return std::any();
    }
}

bool GDScriptSystem::HasFunction(const std::string& functionName) const {
    if (!m_Initialized) return false;

    // Check bound functions
    if (m_BoundFunctions.find(functionName) != m_BoundFunctions.end()) {
        return true;
    }

    // Check if function exists in GDScript
    // In real implementation, would check the GDScript context
    return false;
}

// ============================================================================
// Type Registration
// ============================================================================

void GDScriptSystem::RegisterTypes() {
    if (!m_Initialized) return;

    std::cout << "Registering engine types for GDScript..." << std::endl;

    // Register Vec3
    BindFunction("Vec3", [](const std::vector<std::any>& args) -> std::any {
        if (args.size() >= 3) {
            // Create Vec3 from arguments
            float x = std::any_cast<float>(args[0]);
            float y = std::any_cast<float>(args[1]);
            float z = std::any_cast<float>(args[2]);
            // Return as any (in real implementation, return Vec3 object)
            return std::any();
        }
        return std::any();
    });

    // Register Transform
    BindFunction("Transform", [](const std::vector<std::any>& args) -> std::any {
        // Create Transform object
        return std::any();
    });

    // Register GameObject
    BindFunction("GameObject", [](const std::vector<std::any>& args) -> std::any {
        // Create GameObject object
        return std::any();
    });

    std::cout << "Engine types registered" << std::endl;
}

// ============================================================================
// Class Registration
// ============================================================================

void GDScriptSystem::RegisterClass(const std::string& className,
                                    std::function<void(GDScriptObject)> initializer) {
    if (!m_Initialized) return;

    try {
        // In real implementation, register class with GDScript
        // and call initializer function
        initializer(nullptr);
    }
    catch (const std::exception& e) {
        std::cerr << "Error registering class " << className << ": " << e.what() << std::endl;
    }
}

void GDScriptSystem::BindFunction(const std::string& functionName,
                                   std::function<std::any(const std::vector<std::any>&)> callable) {
    if (!m_Initialized) return;

    m_BoundFunctions[functionName] = callable;
}

// ============================================================================
// Signal System
// ============================================================================

bool GDScriptSystem::ConnectSignal(const std::string& objectName,
                                    const std::string& signalName,
                                    std::function<void(const std::vector<std::any>&)> callback) {
    if (!m_Initialized) {
        m_LastError = "GDScriptSystem not initialized";
        return false;
    }

    try {
        std::string key = objectName + ":" + signalName;
        m_SignalCallbacks[key] = callback;
        return true;
    }
    catch (const std::exception& e) {
        m_LastError = std::string("Signal connection error: ") + e.what();
        return false;
    }
}

bool GDScriptSystem::EmitSignal(const std::string& objectName,
                                 const std::string& signalName,
                                 const std::vector<std::any>& args) {
    if (!m_Initialized) {
        m_LastError = "GDScriptSystem not initialized";
        return false;
    }

    try {
        std::string key = objectName + ":" + signalName;
        auto it = m_SignalCallbacks.find(key);
        if (it != m_SignalCallbacks.end()) {
            it->second(args);
            return true;
        }
        return false;
    }
    catch (const std::exception& e) {
        m_LastError = std::string("Signal emission error: ") + e.what();
        return false;
    }
}

// ============================================================================
// Hot-Reload Support
// ============================================================================

void GDScriptSystem::ReloadScript(const std::string& filepath) {
    if (!m_HotReloadEnabled || !m_Initialized) {
        return;
    }

    auto it = m_LoadedScripts.find(filepath);
    if (it == m_LoadedScripts.end()) {
        return;
    }

    std::cout << "Hot-reloading script: " << filepath << std::endl;
    RunScript(filepath);
}

// ============================================================================
// Memory Management
// ============================================================================

uint64_t GDScriptSystem::GetMemoryUsage() const {
    if (!m_Initialized || !m_VM) return 0;
    return GDScriptAPI::GetMemoryUsage(m_VM);
}

// ============================================================================
// Internal Helpers
// ============================================================================

void GDScriptSystem::InitializeVM() {
    // Implementation for VM setup beyond basic creation
    std::cout << "GDScript VM initialized" << std::endl;
}

void GDScriptSystem::ShutdownVM() {
    // Implementation for VM cleanup
    std::cout << "GDScript VM shut down" << std::endl;
}

void GDScriptSystem::SetupBuiltins() {
    if (!m_Initialized) return;

    std::cout << "Setting up GDScript built-in functions..." << std::endl;

    // Setup print function
    BindFunction("print", [](const std::vector<std::any>& args) -> std::any {
        for (const auto& arg : args) {
            if (arg.type() == typeid(std::string)) {
                std::cout << std::any_cast<std::string>(arg);
            }
            else if (arg.type() == typeid(float)) {
                std::cout << std::any_cast<float>(arg);
            }
            else if (arg.type() == typeid(int)) {
                std::cout << std::any_cast<int>(arg);
            }
            else if (arg.type() == typeid(bool)) {
                std::cout << (std::any_cast<bool>(arg) ? "true" : "false");
            }
        }
        std::cout << std::endl;
        return std::any();
    });

    // Setup debug function
    BindFunction("debug", [](const std::vector<std::any>& args) -> std::any {
        std::cout << "[DEBUG] ";
        for (const auto& arg : args) {
            if (arg.type() == typeid(std::string)) {
                std::cout << std::any_cast<std::string>(arg) << " ";
            }
        }
        std::cout << std::endl;
        return std::any();
    });

    std::cout << "Built-in functions set up" << std::endl;
}

void GDScriptSystem::LoadStandardLibrary() {
    if (!m_Initialized) return;

    std::cout << "Loading GDScript standard library..." << std::endl;

    // In a real implementation, this would load the GDScript standard library
    // For now, we just mark it as loaded
    std::cout << "Standard library loaded" << std::endl;
}
