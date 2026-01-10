#pragma once

#include "Gizmo.h"
#include "SoftBodyTearPattern.h"
#include <memory>
#include <vector>

class PhysXSoftBody;

/**
 * @brief Interactive gizmo for tear pattern visualization and execution
 */
class TearPatternGizmo : public Gizmo {
public:
    TearPatternGizmo();
    ~TearPatternGizmo() override = default;

    void Update(float deltaTime) override;
    void Draw(Shader* shader, const Camera& camera) override;
    
    bool OnMousePress(const Ray& ray) override;
    void OnMouseRelease() override;
    void OnMouseDrag(const Ray& ray, const Camera& camera) override;
    bool OnMouseMove(const Ray& ray) override;

    /**
     * @brief Set pattern to visualize
     */
    void SetPattern(std::unique_ptr<SoftBodyTearPattern> pattern);
    
    /**
     * @brief Get current pattern
     */
    SoftBodyTearPattern* GetPattern() const { return m_Pattern.get(); }
    
    /**
     * @brief Set soft body for pattern application
     */
    void SetSoftBody(PhysXSoftBody* softBody) { m_SoftBody = softBody; }
    
    /**
     * @brief Set start and end points for pattern
     */
    void SetPoints(const Vec3& startPoint, const Vec3& endPoint);
    
    /**
     * @brief Preview affected tetrahedra
     */
    void ShowPreview(bool show);
    
    /**
     * @brief Check if preview is active
     */
    bool IsPreviewActive() const { return m_ShowPreview; }
    
    /**
     * @brief Execute tear along pattern
     */
    void ExecuteTear();
    
    /**
     * @brief Update preview (recalculate affected tetrahedra)
     */
    void UpdatePreview();
    
    // Visual settings
    void SetPreviewColor(const Vec3& color) { m_PreviewColor = color; }
    void SetPatternColor(const Vec3& color) { m_PatternColor = color; }

private:
    std::unique_ptr<SoftBodyTearPattern> m_Pattern;
    PhysXSoftBody* m_SoftBody = nullptr;
    
    // Pattern points
    Vec3 m_StartPoint;
    Vec3 m_EndPoint;
    bool m_StartPointSet = false;
    bool m_EndPointSet = false;
    
    // Preview state
    bool m_ShowPreview = false;
    std::vector<int> m_AffectedTetrahedra;
    
    // Interaction state
    bool m_DraggingStart = false;
    bool m_DraggingEnd = false;
    Vec3 m_DragPlaneNormal;
    
    // Visual settings
    Vec3 m_PreviewColor = Vec3(1.0f, 0.3f, 0.3f);  // Red for affected tets
    Vec3 m_PatternColor = Vec3(1.0f, 1.0f, 0.0f);  // Yellow for pattern path
    float m_PointSize = 0.15f;
    
    // Rendering helpers
    void DrawPatternPath(Shader* shader, const Camera& camera);
    void DrawAffectedTetrahedra(Shader* shader, const Camera& camera);
    void DrawControlPoints(Shader* shader, const Camera& camera);
    
    // Picking helpers
    bool PickStartPoint(const Ray& ray, float& distance);
    bool PickEndPoint(const Ray& ray, float& distance);
};
