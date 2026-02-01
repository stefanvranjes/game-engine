#include "RustScriptSystem.h"
#include <iostream>
#include <fstream>
#include <chrono>

#ifdef _WIN32
    #include <windows.h>
#else
    #include <dlfcn.h>
#endif

RustScriptSystem::RustScriptSystem()
    : m_MemoryUsage(0)
{
}

RustScriptSystem::~RustScriptSystem()
{
    Shutdown();
}

void RustScriptSystem::Init()
{
    std::cout << "RustScriptSystem Initialized (Native Compiled)" << std::endl;
}

void RustScriptSystem::Shutdown()
{
    // Unload all loaded libraries
    for (auto& pair : m_LoadedLibraries) {
        UnloadNativeLibrary(pair.first);
    }
    m_LoadedLibraries.clear();
    std::cout << "RustScriptSystem Shutdown" << std::endl;
}

void RustScriptSystem::Update(float deltaTime)
{
    // Rust scripts are native compiled, no per-frame interpretation needed
    // However, this could be used to call Rust-side update functions if registered
}

bool RustScriptSystem::RunScript(const std::string& filepath)
{
    return LoadNativeLibrary(filepath);
}

bool RustScriptSystem::LoadNativeLibrary(const std::string& libPath)
{
    // Unload previous version if it exists (for hot-reload)
    auto it = m_LoadedLibraries.find(libPath);
    if (it != m_LoadedLibraries.end()) {
        UnloadNativeLibrary(libPath);
    }

#ifdef _WIN32
    HMODULE handle = LoadLibraryA(libPath.c_str());
    if (!handle) {
        DWORD error = ::GetLastError();
        SetError("Failed to load Rust library: " + libPath + " (error: " + std::to_string(error) + ")");
        return false;
    }
#else
    void* handle = dlopen(libPath.c_str(), RTLD_LAZY | RTLD_GLOBAL);
    if (!handle) {
        SetError(std::string("Failed to load Rust library: ") + dlerror());
        return false;
    }
#endif

    m_LoadedLibraries[libPath] = handle;

    // Try to call init function if it exists
    auto init_func = GetFunction<RustInitFunc>(handle, "rust_init");
    if (init_func) {
        init_func();
    }

    std::cout << "Loaded Rust library: " << libPath << std::endl;
    return true;
}

bool RustScriptSystem::ExecuteString(const std::string& source)
{
    SetError("Rust scripting does not support runtime string execution. Use compiled libraries.");
    return false;
}

bool RustScriptSystem::HasType(const std::string& typeName) const
{
    // Rust types are opaque to C++, so we can only know about types that are explicitly exported
    return false;
}

std::any RustScriptSystem::CallFunction(const std::string& functionName,
                                        const std::vector<std::any>& args)
{
    // For simplicity, we'll return null
    // In practice, you would:
    // 1. Look up the function across all loaded libraries
    // 2. Convert std::any arguments to appropriate C types
    // 3. Call the Rust function
    // 4. Convert return value back to std::any
    
    std::cout << "Calling Rust function: " << functionName << " with " << args.size() << " arguments" << std::endl;
    return std::any();
}

void RustScriptSystem::ReloadScript(const std::string& filepath)
{
    std::cout << "Hot-reloading Rust library: " << filepath << std::endl;
    LoadNativeLibrary(filepath);
}

uint64_t RustScriptSystem::GetMemoryUsage() const
{
    // For native compiled code, memory usage would need to be tracked
    // via OS-specific APIs or by the Rust code itself
    return m_MemoryUsage;
}

double RustScriptSystem::GetLastExecutionTime() const
{
    return m_LastExecutionTime;
}

bool RustScriptSystem::HasErrors() const
{
    return !m_LastError.empty();
}

std::string RustScriptSystem::GetLastError() const
{
    return m_LastError;
}

void* RustScriptSystem::GetLibraryHandle(const std::string& libPath) const
{
    auto it = m_LoadedLibraries.find(libPath);
    if (it != m_LoadedLibraries.end()) {
        return it->second;
    }
    return nullptr;
}

bool RustScriptSystem::UnloadNativeLibrary(const std::string& libPath)
{
    auto it = m_LoadedLibraries.find(libPath);
    if (it == m_LoadedLibraries.end()) {
        return false;
    }

    void* handle = it->second;

    // Try to call shutdown function if it exists
    auto shutdown_func = GetFunction<RustShutdownFunc>(handle, "rust_shutdown");
    if (shutdown_func) {
        shutdown_func();
    }

#ifdef _WIN32
    if (!FreeLibrary((HMODULE)handle)) {
        SetError("Failed to unload Rust library: " + libPath);
        return false;
    }
#else
    if (dlclose(handle) != 0) {
        SetError(std::string("Failed to unload Rust library: ") + dlerror());
        return false;
    }
#endif

    m_LoadedLibraries.erase(it);
    return true;
}

void RustScriptSystem::SetError(const std::string& error)
{
    m_LastError = error;
    std::cerr << "RustScriptSystem Error: " << error << std::endl;
}
