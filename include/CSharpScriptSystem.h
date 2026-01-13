#pragma once

#include "IScriptSystem.h"
#include <string>

#ifdef HAS_MONO
#include <mono/jit/jit.h>
#include <mono/metadata/assembly.h>
#include <mono/metadata/debug-helpers.h>
#endif

class CSharpScriptSystem : public IScriptSystem {
public:
    static CSharpScriptSystem& GetInstance() {
        static CSharpScriptSystem instance;
        return instance;
    }

    void Init() override;
    void Shutdown() override;
    void Update(float deltaTime) override;

    bool RunScript(const std::string& filepath) override;
    
#ifdef HAS_MONO
    // Create an object of a class from the loaded assembly
    MonoObject* CreateObject(const std::string& namespaceName, const std::string& className);
    
    // Get the main assembly (GameScript.dll)
    MonoAssembly* GetAssembly() const { return m_GameAssembly; }

    MonoDomain* GetDomain() const { return m_Domain; }
#endif

private:
    CSharpScriptSystem();
    ~CSharpScriptSystem();

#ifdef HAS_MONO
    MonoDomain* m_RootDomain = nullptr;
    MonoDomain* m_Domain = nullptr;
    MonoAssembly* m_GameAssembly = nullptr;
    MonoImage* m_GameAssemblyImage = nullptr;
#endif

    void LoadAssembly(const std::string& path);
    void RegisterInternalCalls();
};
