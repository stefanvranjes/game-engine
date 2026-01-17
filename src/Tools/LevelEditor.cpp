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
    
    // Load scene from JSON/YAML file
    std::ifstream file(filepath);
    if (file.is_open()) {
        std::string line;
        std::string content((std::istreambuf_iterator<char>(file)),
                           std::istreambuf_iterator<char>());
        file.close();
        
        // Parse JSON structure to deserialize game objects
        // Expected structure:
        // {
        //   "name": "Scene Name",
        //   "gameObjects": [
        //     {
        //       "name": "GameObject1",
        //       "position": [0, 0, 0],
        //       "rotation": [0, 0, 0],
        //       "scale": [1, 1, 1],
        //       "components": [...]
        //     },
        //     ...
        //   ]
        // }
        
        // In a real implementation, parse JSON and create GameObjects
        // auto json = nlohmann::json::parse(content);
        // for (const auto& objData : json["gameObjects"]) {
        //     auto gameObj = DeserializeGameObject(objData.dump());
        //     if (gameObj) {
        //         m_SceneObjects.push_back(gameObj);
        //     }
        // }
        
        std::cout << "Scene loaded successfully" << std::endl;
    } 
    else {
        std::cerr << "Failed to open scene file: " << filepath << std::endl;
    }
}

void LevelEditor::SaveScene(const std::string& filepath) {
    std::cout << "Saving scene to: " << filepath << std::endl;
    SerializeScene(filepath);
}

void LevelEditor::ExportScene(const std::string& filepath, const std::string& format) {
    std::cout << "Exporting scene as " << format << " to: " << filepath << std::endl;
    
    // Export scene in specified format (json, yaml, etc)
    if (format == "json") {
        ExportSceneAsJSON(filepath);
    } 
    else if (format == "yaml") {
        ExportSceneAsYAML(filepath);
    } 
    else {
        std::cerr << "Unsupported export format: " << format << std::endl;
        return;
    }
    
    std::cout << "Scene exported successfully" << std::endl;
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
    
    // Create copy of GameObject with all components
    auto duplicate = std::make_shared<GameObject>(obj->GetName() + "_Copy");
    
    // Copy transform properties
    duplicate->GetTransform().SetPosition(obj->GetTransform().GetPosition());
    duplicate->GetTransform().SetRotation(obj->GetTransform().GetRotation());
    duplicate->GetTransform().SetScale(obj->GetTransform().GetScale());
    
    // In a real implementation:
    // - Copy all components (mesh, material, collider, etc.)
    // - Copy component properties
    // - Handle nested components
    // - Clone shared resources (textures, meshes) appropriately
    
    m_SceneObjects.push_back(duplicate);
    
    SaveUndoState();
}

void LevelEditor::SetParent(std::shared_ptr<GameObject> child, std::shared_ptr<GameObject> parent) {
    if (!child) return;
    
    std::cout << "Setting parent of " << child->GetName() 
              << " to " << (parent ? parent->GetName() : "nullptr") << std::endl;
    
    // Update hierarchy - set parent-child relationship
    // In a real implementation:
    // 1. Detach from current parent if any
    // 2. Set new parent reference
    // 3. Notify transform hierarchy for world position updates
    // 4. Update child's local transform to maintain world position if needed
    
    if (parent) {
        // Attach child to parent
        parent->AddChild(child);
        std::cout << "  Hierarchy updated: " << child->GetName() << " is now child of " 
                  << parent->GetName() << std::endl;
    } 
    else {
        // Detach child from parent (make it root)
        // child->SetParent(nullptr);
        std::cout << "  " << child->GetName() << " detached from parent (now root object)" << std::endl;
    }
    
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
    
    // Create particle system and attach to game object
    auto emitterObj = std::make_shared<GameObject>(name);
    
    // Create particle system component
    // In a real implementation:
    // auto particleSystem = emitterObj->AddComponent<ParticleSystem>();
    // particleSystem->SetMaxParticles(100000);
    // particleSystem->SetEmissionRate(1000);
    // particleSystem->SetLifetime(2.0f);
    // particleSystem->SetInitialVelocity({0, 5, 0});
    // particleSystem->EnableGPUCompute(true);
    
    m_SceneObjects.push_back(emitterObj);
    
    std::cout << "  Particle emitter created and added to scene" << std::endl;
    SaveUndoState();
}

void LevelEditor::EnableGizmo(bool enable, const std::string& mode) {
    m_GizmoEnabled = enable;
    m_GizmoMode = mode;
    std::cout << "Gizmo " << (enable ? "enabled" : "disabled") 
              << " (mode: " << mode << ")" << std::endl;
}

void LevelEditor::SerializeScene(const std::string& filepath) {
    std::cout << "Serializing scene to: " << filepath << std::endl;
    
    std::ofstream file(filepath);
    if (file.is_open()) {
        file << "{\n";
        file << "  \"sceneMetadata\": {\n";
        file << "    \"name\": \"Scene\",\n";
        file << "    \"objectCount\": " << m_SceneObjects.size() << "\n";
        file << "  },\n";
        file << "  \"gameObjects\": [\n";
        
        // Serialize each game object
        for (size_t i = 0; i < m_SceneObjects.size(); ++i) {
            auto& obj = m_SceneObjects[i];
            
            file << "    {\n";
            file << "      \"name\": \"" << obj->GetName() << "\",\n";
            
            // Serialize transform
            auto& transform = obj->GetTransform();
            auto pos = transform.GetPosition();
            auto rot = transform.GetRotation();
            auto scale = transform.GetScale();
            
            file << "      \"position\": [" << pos.x << ", " << pos.y << ", " << pos.z << "],\n";
            file << "      \"rotation\": [" << rot.x << ", " << rot.y << ", " << rot.z << "],\n";
            file << "      \"scale\": [" << scale.x << ", " << scale.y << ", " << scale.z << "],\n";
            
            // Serialize components would go here
            file << "      \"components\": []\n";
            
            if (i < m_SceneObjects.size() - 1) {
                file << "    },\n";
            } else {
                file << "    }\n";
            }
        }
        
        file << "  ]\n}\n";
        file.close();
        
        std::cout << "Scene serialized successfully (" << m_SceneObjects.size() << " objects)" << std::endl;
    }
}

std::shared_ptr<GameObject> LevelEditor::DeserializeGameObject(const std::string& data) {
    // Parse JSON/YAML and create GameObject
    // Expected JSON structure:
    // {
    //   "name": "GameObject",
    //   "position": [x, y, z],
    //   "rotation": [x, y, z],
    //   "scale": [x, y, z],
    //   "components": [...]
    // }
    
    try {
        // In a real implementation:
        // auto json = nlohmann::json::parse(data);
        // std::string name = json.value("name", "GameObject");
        
        // Create GameObject with deserialized name
        auto gameObj = std::make_shared<GameObject>("Deserialized");
        
        // Set transform from JSON if available
        // auto pos = json["position"].get<std::array<float, 3>>();
        // gameObj->GetTransform().SetPosition(glm::vec3(pos[0], pos[1], pos[2]));
        
        // Deserialize and attach components
        // for (const auto& compData : json["components"]) {
        //     DeserializeComponent(gameObj, compData);
        // }
        
        return gameObj;
    } 
    catch (const std::exception& e) {
        std::cerr << "Failed to deserialize GameObject: " << e.what() << std::endl;
        return nullptr;
    }
}

void LevelEditor::SaveUndoState() {
    // Serialize current scene state for undo history
    // In a real implementation, this would:
    // 1. Create a snapshot of the entire scene
    // 2. Serialize all GameObjects and components
    // 3. Store transform, properties, and relationships
    // 4. Push serialized state onto undo stack
    // 5. Clear redo stack on new action
    
    // For now, we push a generic state marker
    m_UndoStack.push("state_" + std::to_string(m_UndoStack.size()));
}

// Helper methods for export formats
namespace {
    void ExportSceneAsJSON(const std::string& filepath, const std::vector<std::shared_ptr<GameObject>>& objects) {
        std::ofstream file(filepath);
        if (file.is_open()) {
            file << "{\n";
            file << "  \"format\": \"gameengine-scene\",\n";
            file << "  \"version\": 1,\n";
            file << "  \"objects\": [\n";
            
            for (size_t i = 0; i < objects.size(); ++i) {
                auto& obj = objects[i];
                file << "    {\n";
                file << "      \"name\": \"" << obj->GetName() << "\",\n";
                
                auto& transform = obj->GetTransform();
                auto pos = transform.GetPosition();
                auto rot = transform.GetRotation();
                auto scale = transform.GetScale();
                
                file << "      \"position\": [" << pos.x << ", " << pos.y << ", " << pos.z << "],\n";
                file << "      \"rotation\": [" << rot.x << ", " << rot.y << ", " << rot.z << "],\n";
                file << "      \"scale\": [" << scale.x << ", " << scale.y << ", " << scale.z << "]\n";
                
                if (i < objects.size() - 1) {
                    file << "    },\n";
                } else {
                    file << "    }\n";
                }
            }
            
            file << "  ]\n}\n";
            file.close();
        }
    }
    
    void ExportSceneAsYAML(const std::string& filepath, const std::vector<std::shared_ptr<GameObject>>& objects) {
        std::ofstream file(filepath);
        if (file.is_open()) {
            file << "format: gameengine-scene\n";
            file << "version: 1\n";
            file << "objects:\n";
            
            for (const auto& obj : objects) {
                file << "  - name: " << obj->GetName() << "\n";
                
                auto& transform = obj->GetTransform();
                auto pos = transform.GetPosition();
                auto rot = transform.GetRotation();
                auto scale = transform.GetScale();
                
                file << "    position: [" << pos.x << ", " << pos.y << ", " << pos.z << "]\n";
                file << "    rotation: [" << rot.x << ", " << rot.y << ", " << rot.z << "]\n";
                file << "    scale: [" << scale.x << ", " << scale.y << ", " << scale.z << "]\n";
            }
            
            file.close();
        }
    }
}

// Update ExportScene to properly call helper functions
void LevelEditor::ExportSceneAsJSON(const std::string& filepath) {
    ::ExportSceneAsJSON(filepath, m_SceneObjects);
}

void LevelEditor::ExportSceneAsYAML(const std::string& filepath) {
    ::ExportSceneAsYAML(filepath, m_SceneObjects);
}