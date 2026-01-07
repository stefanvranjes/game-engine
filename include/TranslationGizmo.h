#pragma once

#include "Gizmo.h"

class TranslationGizmo : public Gizmo {
public:
    TranslationGizmo() = default;

    void Draw(Shader* shader, const Camera& camera) override;
    
    bool OnMousePress(const Ray& ray) override;
    void OnMouseRelease() override;
    void OnMouseDrag(const Ray& ray, const Camera& camera) override;
    bool OnMouseMove(const Ray& ray) override;

private:
    float m_Scale = 1.0f;
    Vec3 m_DragOriginalPos;
    Vec3 m_DragOffset; // Not used actually due to plane logic, but kept for state
    Vec3 m_DragStartPoint; // Point on the drag axis/plane where drag started
    
    // Helper to get axes directions in world space (currently just world axes, but for local mode it changes)
    void GetAxes(Vec3& x, Vec3& y, Vec3& z);
};
