#include "CSharpScriptSystem.h"
#include "GameObject.h"
#include "Transform.h"
#include <iostream>
#include <filesystem>
#include <fstream>

// Internal Calls
// void CS_Log(string message)
static void CS_Log(MonoString* message) {
    char* str = mono_string_to_utf8(message);
    std::cout << "[C#] " << str << std::endl;
    mono_free(str);
}

// Internal Calls for GameObject/Transform would go here
// ...

CSharpScriptSystem::CSharpScriptSystem() {}

CSharpScriptSystem::~CSharpScriptSystem() {
    Shutdown();
}

void CSharpScriptSystem::Init() {
    if (m_RootDomain) return;

    // Set path to mono libraries if needed (or rely on system path)
    // mono_set_dirs("path/to/lib", "path/to/etc");
    
    m_RootDomain = mono_jit_init("GameEngineDomain");
    if (!m_RootDomain) {
        std::cerr << "Failed to initialize Mono JIT" << std::endl;
        return;
    }
    
    // Create specific app domain
    m_Domain = mono_domain_create_appdomain("GameScriptRuntime", nullptr);
    mono_domain_set(m_Domain, true);

    std::cout << "CSharpScriptSystem Initialized (Mono)" << std::endl;
    
    RegisterInternalCalls();
}

void CSharpScriptSystem::Shutdown() {
    if (m_Domain) {
        // mono_domain_unload(m_Domain); // Can cause issues if unloaded incorrectly, often skip in game engines
        m_Domain = nullptr;
    }
    
    if (m_RootDomain) {
        mono_jit_cleanup(m_RootDomain);
        m_RootDomain = nullptr;
    }
}

void CSharpScriptSystem::Update(float deltaTime) {
    // Optional
}

bool CSharpScriptSystem::RunScript(const std::string& filepath) {
    // In C#, "running a script" usually means loading an assembly.
    // If filepath ends in .dll, load it.
    if (filepath.find(".dll") != std::string::npos) {
        LoadAssembly(filepath);
        return m_GameAssembly != nullptr;
    }
    return false;
}

void CSharpScriptSystem::LoadAssembly(const std::string& path) {
    if (!std::filesystem::exists(path)) {
        std::cerr << "Assembly not found: " << path << std::endl;
        return;
    }

    std::vector<char> fileData;
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) return;
    
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    fileData.resize(size);
    if (!file.read(fileData.data(), size)) return;

    // Load assembly from image data
    MonoImageOpenStatus status;
    MonoImage* image = mono_image_open_from_data_with_name(
        fileData.data(), (uint32_t)size, 
        true, // need copy
        &status, 
        false, // ref only 
        path.c_str()
    );

    if (status != MONO_IMAGE_OK || !image) {
        std::cerr << "Failed to load Mono Image" << std::endl;
        return;
    }

    m_GameAssembly = mono_assembly_load_from_full(image, path.c_str(), &status, 0);
    mono_image_close(image);
    
    if (m_GameAssembly) {
        m_GameAssemblyImage = mono_assembly_get_image(m_GameAssembly);
        std::cout << "Loaded Assembly: " << path << std::endl;
    }
}

void CSharpScriptSystem::RegisterInternalCalls() {
    mono_add_internal_call("GameEngine.Debug::Log", CS_Log);
}

MonoObject* CSharpScriptSystem::CreateObject(const std::string& namespaceName, const std::string& className) {
    if (!m_GameAssemblyImage) return nullptr;

    MonoClass* monoClass = mono_class_from_name(m_GameAssemblyImage, namespaceName.c_str(), className.c_str());
    if (!monoClass) return nullptr;

    MonoObject* obj = mono_object_new(m_Domain, monoClass);
    mono_runtime_object_init(obj);
    return obj;
}
