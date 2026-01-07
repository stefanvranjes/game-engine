#pragma once

#include <string>
#include <memory>
#include "ScriptSystem.h"

class GameObject;

class ScriptComponent {
public:
    ScriptComponent(std::weak_ptr<GameObject> owner);
    ~ScriptComponent();

    void LoadScript(const std::string& filepath);
    void Init();
    void Update(float deltaTime);

    void Reload();

private:
    std::weak_ptr<GameObject> m_Owner;
    std::string m_ScriptPath;
    
    // We could store a Lua table reference here for optimization
    // int m_SelfRef = LUA_NOREF;
};
