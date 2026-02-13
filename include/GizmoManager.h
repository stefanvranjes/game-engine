#pragma once

#include "Gizmo.h"
#include "TranslationGizmo.h"
#include "RotationGizmo.h"
#include "ScaleGizmo.h"
#include <glm/glm.hpp>
#ifdef USE_PHYSX
#include "FractureLineGizmo.h"
#endif
#include <memory>
#include <string>

class GameObject;
class Camera;
struct GLFWwindow;

class GizmoManager {
public:
    GizmoManager();
    ~GizmoManager();
    
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
    
    // Local/World Space Toggle
    void SetUseLocalSpace(bool local) { m_UseLocalSpace = local; }
    bool IsUsingLocalSpace() const { return m_UseLocalSpace; }
    void ToggleLocalSpace() { m_UseLocalSpace = !m_UseLocalSpace; }
    
    // Gizmo Size Control
    void SetGizmoSize(float size) { m_GizmoSize = glm::clamp(size, 0.1f, 5.0f); }
    float GetGizmoSize() const { return m_GizmoSize; }
    
    // Reset to defaults
    void ResetGizmoSettings() {
        m_UseLocalSpace = false;
        m_GizmoSize = 1.0f;
        m_SnappingEnabled = false;
    }

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
    
    bool m_UseLocalSpace = false;
    float m_GizmoSize = 1.0f;
    
    Gizmo* GetActiveGizmo();
    
    // Input state
    bool m_MousePressed = false;
    
public:
    enum class EditMode {
        Object,
        Vertex,
        Edge,
        Face, // Triangle
#ifdef USE_PHYSX
        FractureLine
#endif
    };
    
    void SetEditMode(EditMode mode) { m_EditMode = mode; }
    EditMode GetEditMode() const { return m_EditMode; }
    
    // Selection Sets (Indices)
    const std::vector<int>& GetSelectedVertices() const { return m_SelectedVertices; }
    const std::vector<int>& GetSelectedFaces() const { return m_SelectedFaces; }
    
    void ClearSubObjectSelection();
    
#ifdef USE_PHYSX
    // Fracture Line Gizmo
    /**
     * @brief Get fracture line gizmo
     */
    FractureLineGizmo* GetFractureLineGizmo();
    
    /**
     * @brief Create new fracture line for selected soft body
     */
    void CreateNewFractureLine();
    
    /**
     * @brief Delete selected fracture line
     */
    void DeleteSelectedFractureLine();
    
    // Pattern Library
    /**
     * @brief Get pattern library
     */
    class FractureLinePatternLibrary* GetPatternLibrary() { return m_PatternLibrary.get(); }
    
    /**
     * @brief Save selected fracture line as preset
     */
    bool SaveFractureLinePreset(const std::string& name, const std::string& desc = "");
    
    /**
     * @brief Load fracture line preset
     */
    bool LoadFractureLinePreset(const std::string& name);
    
    /**
     * @brief Save pattern library to file
     */
    bool SavePatternLibrary(const std::string& filename = "");
    
    /**
     * @brief Load pattern library from file
     */
    bool LoadPatternLibrary(const std::string& filename = "");
#endif

private:
    EditMode m_EditMode = EditMode::Object;
    std::vector<int> m_SelectedVertices;
    std::vector<int> m_SelectedFaces; // Stores triangle index (index in indices array / 3)
    
    void PickSubObject(const Ray& ray, const Camera& camera);
    void UpdateMeshSelection(const Vec3& delta);
    
    // Cached original positions for dragging
    std::vector<Vec3> m_OriginalVertexPositions;
    
    // Fracture line gizmo
#ifdef USE_PHYSX
    std::unique_ptr<FractureLineGizmo> m_FractureLineGizmo;
    
    // Pattern library
    std::unique_ptr<class FractureLinePatternLibrary> m_PatternLibrary;
#endif
};
