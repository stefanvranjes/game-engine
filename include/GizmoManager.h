#pragma once

#include "Gizmo.h"
#include "TranslationGizmo.h"
#include "RotationGizmo.h"
#include "ScaleGizmo.h"
#include <memory> 

class GameObject;
class Camera;
struct GLFWwindow;

class GizmoManager {
public:
    GizmoManager();
    
    void Update(float deltaTime);
    void Render(Shader* shader, const Camera& camera);
    
    // Input Handling
    void ProcessInput(GLFWwindow* window, float deltaTime, const Camera& camera);
    bool OnAccessoryInput(int key, int action); // Returns true if consumed
    
    void SetSelectedObject(std::shared_ptr<GameObject> object);
    std::shared_ptr<GameObject> GetSelectedObject() const { return m_SelectedObject; }
    
    void SetGizmoType(GizmoType type);
    GizmoType GetGizmoType() const { return m_CurrentType; }
    
    // Returns true if a gizmo is currently being dragged (to block camera input)
    bool IsDragging() const;

private:
    GizmoType m_CurrentType = GizmoType::Translation;
    std::shared_ptr<GameObject> m_SelectedObject;
    
    std::unique_ptr<TranslationGizmo> m_TranslationGizmo;
    std::unique_ptr<RotationGizmo> m_RotationGizmo;
    std::unique_ptr<ScaleGizmo> m_ScaleGizmo;
    
    Gizmo* GetActiveGizmo();
    
    // Input state
    bool m_MousePressed = false;
};
