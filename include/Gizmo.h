#pragma once

#include "Math/Vec3.h"
#include "Math/Mat4.h"
#include "Math/Vec2.h"
#include "Transform.h"
#include <memory>
#include <vector>

class Camera;
class Shader;
struct GLFWwindow;

enum class GizmoType {
    None,
    Translation,
    Rotation,
    Scale
};

enum class GizmoAxis {
    None,
    X,
    Y,
    Z,
    XY,
    XZ,
    YZ,
    XYZ // Uniform scale or view-aligned rotation
};

#include "Ray.h"

class Gizmo {
public:
    Gizmo();
    virtual ~Gizmo() = default;

    virtual void Update(float /*deltaTime*/) {}
    virtual void Draw(Shader* shader, const Camera& camera);
    
    // Returns true if the gizmo is currently being interacted with (dragged)
    virtual bool OnMousePress(const Ray& ray) = 0;
    virtual void OnMouseRelease() = 0;
    virtual void OnMouseDrag(const Ray& ray, const Camera& camera) = 0;
    virtual bool OnMouseMove(const Ray& ray) = 0; // For hover effects, returns true if hovering

    void SetTransform(Transform* transform) { m_Transform = transform; }
    void SetEnabled(bool enabled) { m_Enabled = enabled; }
    bool IsEnabled() const { return m_Enabled; }
    bool IsDragging() const { return m_IsDragging; }
    GizmoAxis GetHoverAxis() const { return m_HoverAxis; }

    void SetSnapping(bool enabled, float value) {
        m_SnappingEnabled = enabled;
        m_SnapValue = value;
    }

protected:
    // Helper to calculate screen scale to keep gizmo size constant
    float GetScreenScale(const Vec3& position, const Camera& camera);

    // Ray intersection helpers
    float RayIntersectPlane(const Ray& ray, const Vec3& planeNormal, const Vec3& planePoint);
    bool RayIntersectBox(const Ray& ray, const Vec3& boxMin, const Vec3& boxMax, float& t);
    bool RayIntersectTriangle(const Ray& ray, const Vec3& v0, const Vec3& v1, const Vec3& v2, float& t, Vec3& intersectionPoint);
    float RayClosestPoint(const Ray& ray, const Vec3& point);

    // Common Rendering Helpers
    void DrawArrow(Shader* shader, const Vec3& start, const Vec3& end, const Vec3& color, float scale);
    void DrawCube(Shader* shader, const Vec3& center, const Vec3& size, const Vec3& color);
    void DrawCircle(Shader* shader, const Vec3& center, const Vec3& normal, float radius, const Vec3& color);
    void DrawQuad(Shader* shader, const Vec3& center, const Vec3& right, const Vec3& up, const Vec3& color);

    Transform* m_Transform = nullptr;
    bool m_Enabled = true;
    bool m_IsDragging = false;
    
    bool m_SnappingEnabled = false;
    float m_SnapValue = 0.0f;

    GizmoType m_Type = GizmoType::None;
    GizmoAxis m_HoverAxis = GizmoAxis::None;
    GizmoAxis m_DragAxis = GizmoAxis::None;
    
    // Visual settings
    float m_GizmoSize = 1.0f;
    Vec3 m_ColorX = Vec3(0.8f, 0.2f, 0.2f);
    Vec3 m_ColorY = Vec3(0.2f, 0.8f, 0.2f);
    Vec3 m_ColorZ = Vec3(0.2f, 0.2f, 0.8f);
    Vec3 m_ColorSelected = Vec3(1.0f, 1.0f, 0.0f);
    
    // Shared geometry buffers could be static
    static unsigned int s_ArrowVAO, s_ArrowVBO;
    static unsigned int s_CubeVAO, s_CubeVBO;
    static unsigned int s_CircleVAO, s_CircleVBO;
    static unsigned int s_QuadVAO, s_QuadVBO;
    static bool s_Initialized;
    
    static void InitGizmoResources();
};
