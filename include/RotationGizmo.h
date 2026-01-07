#pragma once

#include "Gizmo.h"

class RotationGizmo : public Gizmo {
public:
    RotationGizmo() = default;

    void Draw(Shader* shader, const Camera& camera) override;
    
    bool OnMousePress(const Ray& ray) override;
    void OnMouseRelease() override;
    void OnMouseDrag(const Ray& ray, const Camera& camera) override;
    bool OnMouseMove(const Ray& ray) override;

private:
    float m_Scale = 1.0f;
    Vec3 m_DragOriginalRot; // Euler angles
    Vec3 m_DragStartVec; // Vector from center to click point on sphere
    
    // Get axis vectors
    void GetAxes(Vec3& x, Vec3& y, Vec3& z);
};
