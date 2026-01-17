#include "SquirrelScriptSystem.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>

// Note: Include sqrat.h or squirrel.h when available for Squirrel integration

SquirrelScriptSystem::SquirrelScriptSystem()
    : m_VM(nullptr)
{
}

SquirrelScriptSystem::~SquirrelScriptSystem()
{
    Shutdown();
}

void SquirrelScriptSystem::Init()
{
    // Initialize Squirrel VM
    // This is a stub - actual implementation requires Squirrel integration
    /*
    m_VM = sq_open(1024 * 10); // 10K stack size
    sq_setforeigndeleter(m_VM, -1, nullptr);
    
    // Open standard library
    sq_pushroottable(m_VM);
    sqstd_register_bloblib(m_VM);
    sqstd_register_iolib(m_VM);
    sqstd_register_systemlib(m_VM);
    sqstd_register_mathlib(m_VM);
    sqstd_register_stringlib(m_VM);
    sq_pop(m_VM, 1);
    */
    
    RegisterTypes();
    std::cout << "SquirrelScriptSystem Initialized" << std::endl;
}

void SquirrelScriptSystem::Shutdown()
{
    // Clean up Squirrel resources
    /*
    if (m_VM) {
        sq_close(m_VM);
        m_VM = nullptr;
    }
    */
    std::cout << "SquirrelScriptSystem Shutdown" << std::endl;
}

void SquirrelScriptSystem::Update(float deltaTime)
{
    // Squirrel is fully interpreted at execution time
    // No special update logic needed
}

bool SquirrelScriptSystem::RunScript(const std::string& filepath)
{
    std::ifstream file(filepath);
    if (!file.is_open()) {
        SetError("Failed to open file: " + filepath);
        return false;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    return ExecuteString(buffer.str());
}

bool SquirrelScriptSystem::ExecuteString(const std::string& source)
{
    auto start = std::chrono::high_resolution_clock::now();

    // Stub implementation
    // Real implementation would:
    // 1. Compile Squirrel source
    // 2. Execute in Squirrel VM
    // 3. Handle exceptions and errors
    
    /*
    if (!m_VM) {
        SetError("Squirrel VM not initialized");
        return false;
    }

    sq_pushroottable(m_VM);
    
    SQInteger result = sqstd_dostring(m_VM, source.c_str(), SQTrue);
    
    if (SQ_FAILED(result)) {
        const SQChar *err = nullptr;
        sq_getlasterror(m_VM);
        if (sq_gettype(m_VM, -1) == OT_STRING) {
            sq_getstring(m_VM, -1, &err);
            SetError(std::string(err));
        }
        sq_pop(m_VM, 1);
        return false;
    }
    
    sq_pop(m_VM, 1);
    */
    
    auto end = std::chrono::high_resolution_clock::now();
    m_LastExecutionTime = std::chrono::duration<double>(end - start).count();
    return true;
}

void SquirrelScriptSystem::RegisterTypes()
{
    RegisterGameObjectTypes();
    RegisterMathTypes();
    RegisterPhysicsTypes();
    RegisterAudioTypes();
    RegisterAssetTypes();
}

bool SquirrelScriptSystem::HasType(const std::string& typeName) const
{
    // Check if a type is registered in the Squirrel VM
    // Stub implementation
    return false;
}

std::any SquirrelScriptSystem::CallFunction(const std::string& functionName,
                                            const std::vector<std::any>& args)
{
    // Call a Squirrel function from C++
    /*
    if (!m_VM) {
        SetError("Squirrel VM not initialized");
        return std::any();
    }

    sq_pushroottable(m_VM);
    sq_pushstring(m_VM, functionName.c_str(), -1);
    if (sq_get(m_VM, -2) != SQ_OK) {
        SetError("Function not found: " + functionName);
        return std::any();
    }

    sq_pushroottable(m_VM);
    
    // Convert and push arguments
    for (const auto& arg : args) {
        // Convert std::any to Squirrel values
        // ...
    }

    if (sq_call(m_VM, args.size() + 1, SQTrue, SQTrue) != SQ_OK) {
        SetError("Error calling function: " + functionName);
        return std::any();
    }

    // Get return value
    std::any result = std::any();
    // ...
    
    sq_pop(m_VM, 4); // Pop function, this, result, root table
    */
    return std::any();
}

void SquirrelScriptSystem::ReloadScript(const std::string& filepath)
{
    std::cout << "Hot-reloading script: " << filepath << std::endl;
    RunScript(filepath);
}

uint64_t SquirrelScriptSystem::GetMemoryUsage() const
{
    // Query Squirrel VM memory statistics
    /*
    if (m_VM) {
        // Squirrel doesn't have a built-in memory query function
        // Would need to track manually or via OS-specific APIs
    }
    */
    return m_MemoryUsage;
}

double SquirrelScriptSystem::GetLastExecutionTime() const
{
    return m_LastExecutionTime;
}

bool SquirrelScriptSystem::HasErrors() const
{
    return !m_LastError.empty();
}

std::string SquirrelScriptSystem::GetLastError() const
{
    return m_LastError;
}

void SquirrelScriptSystem::RegisterFunction(const std::string& name,
                                            std::function<void(HSQUIRRELVM)> func)
{
    m_Functions[name] = func;
    
    // Register with Squirrel VM
    /*
    if (!m_VM) return;
    
    // Implementation details for registering native functions
    */
}

void SquirrelScriptSystem::RegisterClass(const std::string& className,
                                         std::function<void(HSQUIRRELVM)> constructor)
{
    // Register a C++ class as a Squirrel class
    /*
    if (!m_VM) return;
    
    // Create Squirrel class
    // Bind constructor and methods
    // Add to global scope
    */
}

void SquirrelScriptSystem::RegisterGameObjectTypes()
{
    // Register GameObject, Transform, Component types
    // Example pseudo-code:
    // RegisterClass("GameObject", [](HSQUIRRELVM vm) { ... });
}

void SquirrelScriptSystem::RegisterMathTypes()
{
    // Register Vec2, Vec3, Vec4, Quaternion, Matrix4, etc.
}

void SquirrelScriptSystem::RegisterPhysicsTypes()
{
    // Register RigidBody, Collider, PhysicsSystem, etc.
}

void SquirrelScriptSystem::RegisterAudioTypes()
{
    // Register AudioSource, AudioListener, AudioSystem, etc.
}

void SquirrelScriptSystem::RegisterAssetTypes()
{
    // Register Texture, Material, Model, Animation, etc.
}

void SquirrelScriptSystem::SetError(const std::string& error)
{
    m_LastError = error;
    std::cerr << "SquirrelScriptSystem Error: " << error << std::endl;
}
