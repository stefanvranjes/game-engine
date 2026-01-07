#pragma once

#include "Gizmo.h"

class ScaleGizmo : public Gizmo {
public:
    ScaleGizmo() = default;

    void Draw(Shader* shader, const Camera& camera) override;
    
    bool OnMousePress(const Ray& ray) override;
    void OnMouseRelease() override;
    void OnMouseDrag(const Ray& ray, const Camera& camera) override;
    bool OnMouseMove(const Ray& ray) override;

private:
    float m_Scale = 1.0f;
    Vec3 m_DragOriginalScale;
    Vec3 m_DragStartPoint; // 2D mouse position or 3D point?
    // For scale, we track mouse delta along the axis on screen or in world?
    // Let's use World projection similar to translation.
    
    void GetAxes(Vec3& x, Vec3& y, Vec3& z);
};
