#include "LevelEditor.h"
#include <iostream>
#include <fstream>
#include <sstream>

namespace Tools {

LevelEditor::LevelEditor() 
    : m_SelectedObject(nullptr), 
      m_GizmoEnabled(false),
      m_GizmoMode("translate") {
}

LevelEditor::~LevelEditor() {
    m_SceneObjects.clear();
}

void LevelEditor::NewScene() {
    std::cout << "Creating new scene..." << std::endl;
    m_SceneObjects.clear();
    m_SelectedObject = nullptr;
    while (!m_UndoStack.empty()) m_UndoStack.pop();
    while (!m_RedoStack.empty()) m_RedoStack.pop();
}

void LevelEditor::LoadScene(const std::string& filepath) {
    std::cout << "Loading scene from: " << filepath << std::endl;
    NewScene();
    
    // TODO: Load scene from JSON/YAML file
    std::ifstream file(filepath);
    if (file.is_open()) {
        std::string line;
        while (std::getline(file, line)) {
            // TODO: Parse and deserialize game objects
        }
        file.close();
        std::cout << "Scene loaded successfully" << std::endl;
    }
}

void LevelEditor::SaveScene(const std::string& filepath) {
    std::cout << "Saving scene to: " << filepath << std::endl;
    SerializeScene(filepath);
}

void LevelEditor::ExportScene(const std::string& filepath, const std::string& format) {
    std::cout << "Exporting scene as " << format << " to: " << filepath << std::endl;
    
    // TODO: Export scene in specified format (json, yaml, etc)
}

std::shared_ptr<GameObject> LevelEditor::CreateGameObject(const std::string& name) {
    auto gameObj = std::make_shared<GameObject>(name);
    m_SceneObjects.push_back(gameObj);
    
    std::cout << "Created GameObject: " << name << std::endl;
    SaveUndoState();
    
    return gameObj;
}

void LevelEditor::DeleteGameObject(std::shared_ptr<GameObject> obj) {
    if (!obj) return;
    
    std::cout << "Deleting GameObject: " << obj->GetName() << std::endl;
    
    auto it = std::find(m_SceneObjects.begin(), m_SceneObjects.end(), obj);
    if (it != m_SceneObjects.end()) {
        m_SceneObjects.erase(it);
    }
    
    if (m_SelectedObject == obj) {
        m_SelectedObject = nullptr;
    }
    
    SaveUndoState();
}

void LevelEditor::DuplicateGameObject(std::shared_ptr<GameObject> obj) {
    if (!obj) return;
    
    std::cout << "Duplicating GameObject: " << obj->GetName() << std::endl;
    
    // TODO: Create copy of GameObject with all components
    auto duplicate = std::make_shared<GameObject>(obj->GetName() + "_Copy");
    m_SceneObjects.push_back(duplicate);
    
    SaveUndoState();
}

void LevelEditor::SetParent(std::shared_ptr<GameObject> child, std::shared_ptr<GameObject> parent) {
    if (!child) return;
    
    std::cout << "Setting parent of " << child->GetName() 
              << " to " << (parent ? parent->GetName() : "nullptr") << std::endl;
    
    // TODO: Update hierarchy
    SaveUndoState();
}

void LevelEditor::MoveToFront(std::shared_ptr<GameObject> obj) {
    auto it = std::find(m_SceneObjects.begin(), m_SceneObjects.end(), obj);
    if (it != m_SceneObjects.end()) {
        m_SceneObjects.erase(it);
        m_SceneObjects.push_back(obj);
        SaveUndoState();
    }
}

void LevelEditor::MoveToBack(std::shared_ptr<GameObject> obj) {
    auto it = std::find(m_SceneObjects.begin(), m_SceneObjects.end(), obj);
    if (it != m_SceneObjects.end()) {
        m_SceneObjects.erase(it);
        m_SceneObjects.insert(m_SceneObjects.begin(), obj);
        SaveUndoState();
    }
}

void LevelEditor::SelectGameObject(std::shared_ptr<GameObject> obj) {
    m_SelectedObject = obj;
    std::cout << "Selected: " << (obj ? obj->GetName() : "None") << std::endl;
}

void LevelEditor::Undo() {
    if (!m_UndoStack.empty()) {
        m_RedoStack.push(m_UndoStack.top());
        m_UndoStack.pop();
        std::cout << "Undo performed" << std::endl;
    }
}

void LevelEditor::Redo() {
    if (!m_RedoStack.empty()) {
        m_UndoStack.push(m_RedoStack.top());
        m_RedoStack.pop();
        std::cout << "Redo performed" << std::endl;
    }
}

void LevelEditor::CreateParticleEmitter(const std::string& name) {
    std::cout << "Creating particle emitter: " << name << std::endl;
    
    // TODO: Create particle system and attach to game object
}

void LevelEditor::EnableGizmo(bool enable, const std::string& mode) {
    m_GizmoEnabled = enable;
    m_GizmoMode = mode;
    std::cout << "Gizmo " << (enable ? "enabled" : "disabled") 
              << " (mode: " << mode << ")" << std::endl;
}

void LevelEditor::SerializeScene(const std::string& filepath) {
    std::ofstream file(filepath);
    if (file.is_open()) {
        file << "{\n  \"gameObjects\": [\n";
        
        for (size_t i = 0; i < m_SceneObjects.size(); ++i) {
            // TODO: Serialize each game object
            if (i < m_SceneObjects.size() - 1) {
                file << "    {},\n";
            } else {
                file << "    {}\n";
            }
        }
        
        file << "  ]\n}\n";
        file.close();
    }
}

std::shared_ptr<GameObject> LevelEditor::DeserializeGameObject(const std::string& data) {
    // TODO: Parse JSON/YAML and create GameObject
    return std::make_shared<GameObject>("Deserialized");
}

void LevelEditor::SaveUndoState() {
    // TODO: Serialize current scene state
    m_UndoStack.push("state");
}

} // namespace Tools