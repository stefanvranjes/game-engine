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

    // Snapping
    void SetSnappingEnabled(bool enabled) { m_SnappingEnabled = enabled; }
    bool IsSnappingEnabled() const { return m_SnappingEnabled; }
    
    void SetTranslationSnap(float snap) { m_TranslationSnap = snap; }
    float GetTranslationSnap() const { return m_TranslationSnap; }
    
    void SetRotationSnap(float snap) { m_RotationSnap = snap; }
    float GetRotationSnap() const { return m_RotationSnap; }
    
    void SetScaleSnap(float snap) { m_ScaleSnap = snap; }
    float GetScaleSnap() const { return m_ScaleSnap; }

private:
    GizmoType m_CurrentType = GizmoType::Translation;
    std::shared_ptr<GameObject> m_SelectedObject;
    
    std::unique_ptr<TranslationGizmo> m_TranslationGizmo;
    std::unique_ptr<RotationGizmo> m_RotationGizmo;
    std::unique_ptr<ScaleGizmo> m_ScaleGizmo;
    
    bool m_SnappingEnabled = false;
    float m_TranslationSnap = 1.0f;
    float m_RotationSnap = 15.0f;
    float m_ScaleSnap = 0.5f;
    
    Gizmo* GetActiveGizmo();
    
    // Input state
    bool m_MousePressed = false;
    
public:
    enum class EditMode {
        Object,
        Vertex,
        Edge,
        Face // Triangle
    };
    
    void SetEditMode(EditMode mode) { m_EditMode = mode; }
    EditMode GetEditMode() const { return m_EditMode; }
    
    // Selection Sets (Indices)
    const std::vector<int>& GetSelectedVertices() const { return m_SelectedVertices; }
    const std::vector<int>& GetSelectedFaces() const { return m_SelectedFaces; }
    
    void ClearSubObjectSelection();

private:
    EditMode m_EditMode = EditMode::Object;
    std::vector<int> m_SelectedVertices;
    std::vector<int> m_SelectedFaces; // Stores triangle index (index in indices array / 3)
    
    void PickSubObject(const Ray& ray, const Camera& camera);
    void UpdateMeshSelection(const Vec3& delta);
    
    // Cached original positions for dragging
    std::vector<Vec3> m_OriginalVertexPositions;
};
