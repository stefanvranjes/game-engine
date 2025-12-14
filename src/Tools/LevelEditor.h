#pragma once

#include <string>
#include <vector>
#include <memory>
#include <stack>
#include "../Core/GameObject.h"
#include "../Particles/ParticleSystem.h"

namespace Tools {

class LevelEditor {
public:
    LevelEditor();
    ~LevelEditor();
    
    // Scene management
    void NewScene();
    void LoadScene(const std::string& filepath);
    void SaveScene(const std::string& filepath);
    void ExportScene(const std::string& filepath, const std::string& format = "json");
    
    // GameObject manipulation
    std::shared_ptr<GameObject> CreateGameObject(const std::string& name);
    void DeleteGameObject(std::shared_ptr<GameObject> obj);
    void DuplicateGameObject(std::shared_ptr<GameObject> obj);
    
    // Hierarchy management
    void SetParent(std::shared_ptr<GameObject> child, std::shared_ptr<GameObject> parent);
    void MoveToFront(std::shared_ptr<GameObject> obj);
    void MoveToBack(std::shared_ptr<GameObject> obj);
    
    // Selection & undo/redo
    void SelectGameObject(std::shared_ptr<GameObject> obj);
    std::shared_ptr<GameObject> GetSelectedGameObject() const { return m_SelectedObject; }
    
    void Undo();
    void Redo();
    
    // Particle system integration
    void CreateParticleEmitter(const std::string& name);
    
    // Gizmo support
    void EnableGizmo(bool enable, const std::string& mode = "translate");
    
    // Get all objects in scene
    const std::vector<std::shared_ptr<GameObject>>& GetSceneObjects() const { return m_SceneObjects; }

private:
    std::vector<std::shared_ptr<GameObject>> m_SceneObjects;
    std::shared_ptr<GameObject> m_SelectedObject;
    std::stack<std::string> m_UndoStack;
    std::stack<std::string> m_RedoStack;
    bool m_GizmoEnabled;
    std::string m_GizmoMode;
    
    void SerializeScene(const std::string& filepath);
    std::shared_ptr<GameObject> DeserializeGameObject(const std::string& data);
    void SaveUndoState();
};

} // namespace Tools