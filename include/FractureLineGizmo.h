#pragma once

#include "Gizmo.h"
#include "FractureLine.h"
#include <vector>
#include <memory>

class PhysXSoftBody;

/**
 * @brief Interactive gizmo for creating and editing fracture lines
 * 
 * Allows visual manipulation of fracture line control points and properties.
 * Supports multiple editing modes: add, edit, delete, and insert points.
 */
class FractureLineGizmo : public Gizmo {
public:
    enum class EditMode {
        Add,      // Click to add new points
        Edit,     // Drag to move points
        Delete,   // Click to delete points
        Insert    // Click on segments to insert points
    };

    FractureLineGizmo();
    ~FractureLineGizmo() override = default;

    void Update(float deltaTime) override;
    void Draw(Shader* shader, const Camera& camera) override;
    
    bool OnMousePress(const Ray& ray) override;
    void OnMouseRelease() override;
    void OnMouseDrag(const Ray& ray, const Camera& camera) override;
    bool OnMouseMove(const Ray& ray) override;

    // Soft body binding
    void SetSoftBody(PhysXSoftBody* softBody) { m_SoftBody = softBody; }
    PhysXSoftBody* GetSoftBody() const { return m_SoftBody; }

    // Edit mode
    void SetEditMode(EditMode mode) { m_EditMode = mode; }
    EditMode GetEditMode() const { return m_EditMode; }

    // Fracture line management
    void CreateNewFractureLine();
    void DeleteSelectedFractureLine();
    void SelectFractureLine(int index);
    void DeselectAll();
    int GetSelectedFractureLineIndex() const { return m_SelectedLineIndex; }
    
    // Point manipulation
    void AddPoint(const Vec3& point);
    void RemovePoint(int pointIndex);
    void InsertPoint(int segmentIndex, const Vec3& point);
    void MovePoint(int pointIndex, const Vec3& newPosition);

    // Property editing
    void SetSelectedLineWeakness(float weakness);
    void SetSelectedLineWidth(float width);
    float GetSelectedLineWeakness() const;
    float GetSelectedLineWidth() const;

    // Visual settings
    void SetPointSize(float size) { m_PointSize = size; }
    void SetLineThickness(float thickness) { m_LineThickness = thickness; }
    void SetShowWidthVisualization(bool show) { m_ShowWidthVisualization = show; }
    
    // Pattern Library Integration
    /**
     * @brief Save selected fracture line as preset
     */
    bool SaveAsPreset(const std::string& name, const std::string& description = "");
    
    /**
     * @brief Load preset and create new fracture line
     */
    bool LoadPreset(const std::string& name);
    
    /**
     * @brief Get pattern library
     */
    class FractureLinePatternLibrary* GetPatternLibrary() { return m_PatternLibrary; }
    
    /**
     * @brief Set pattern library
     */
    void SetPatternLibrary(class FractureLinePatternLibrary* library) { m_PatternLibrary = library; }
    
    // Tear Pattern Integration
    /**
     * @brief Convert selected fracture line to tear pattern
     */
    std::unique_ptr<SoftBodyTearPattern> ConvertToPattern(
        SoftBodyTearPattern::PatternType type
    );
    
    /**
     * @brief Execute tear along selected fracture line
     */
    void ExecuteTearAlongLine();
    
    /**
     * @brief Show tear preview for selected line
     */
    void ShowTearPreview(bool show);
    
    /**
     * @brief Check if tear preview is active
     */
    bool IsTearPreviewActive() const { return m_ShowTearPreview; }

private:
    PhysXSoftBody* m_SoftBody = nullptr;
    EditMode m_EditMode = EditMode::Edit;
    
    // Selection state
    int m_SelectedLineIndex = -1;
    int m_SelectedPointIndex = -1;
    int m_HoveredPointIndex = -1;
    int m_HoveredSegmentIndex = -1;
    
    // Dragging state
    Vec3 m_DragStartPoint;
    Vec3 m_DragPlaneNormal;
    Vec3 m_OriginalPointPosition;
    
    // Visual settings
    float m_PointSize = 0.1f;
    float m_LineThickness = 0.05f;
    bool m_ShowWidthVisualization = true;
    
    // Colors
    Vec3 m_ColorNormal = Vec3(0.8f, 0.4f, 0.2f);
    Vec3 m_ColorHovered = Vec3(1.0f, 0.8f, 0.2f);
    Vec3 m_ColorSelected = Vec3(1.0f, 1.0f, 0.0f);
    Vec3 m_ColorWeak = Vec3(0.9f, 0.2f, 0.2f);
    Vec3 m_ColorStrong = Vec3(0.2f, 0.9f, 0.2f);
    
    // Pattern library
    class FractureLinePatternLibrary* m_PatternLibrary = nullptr;
    
    // Tear preview state
    bool m_ShowTearPreview = false;
    std::vector<int> m_PreviewAffectedTets;
    
    // Rendering helpers
    void DrawFractureLine(Shader* shader, const Camera& camera, const FractureLine& line, int lineIndex);
    void DrawControlPoint(Shader* shader, const Vec3& position, const Vec3& color, float size);
    void DrawLineSegment(Shader* shader, const Vec3& start, const Vec3& end, const Vec3& color, float thickness);
    void DrawWidthVisualization(Shader* shader, const Camera& camera, const FractureLine& line);
    
    // Picking helpers
    bool PickPoint(const Ray& ray, int lineIndex, int& outPointIndex, float& outDistance);
    bool PickSegment(const Ray& ray, int lineIndex, int& outSegmentIndex, float& outDistance);
    Vec3 GetColorForWeakness(float weakness) const;
    
    // Geometry buffers
    static unsigned int s_SphereVAO, s_SphereVBO, s_SphereEBO;
    static unsigned int s_CylinderVAO, s_CylinderVBO, s_CylinderEBO;
    static bool s_GeometryInitialized;
    static void InitGeometry();
};
