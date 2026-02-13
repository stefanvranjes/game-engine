#include "RotationGizmo.h"
#include "Camera.h"
#include "Transform.h"
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

void RotationGizmo::GetAxes(Vec3& x, Vec3& y, Vec3& z) {
    if (m_UseLocalSpace && m_Transform) {
        // Transform world axes by the object's rotation
        const auto& quat = m_Transform->rotation;
        glm::mat4 rotMatrix = glm::mat4_cast(quat);
        x = Vec3(rotMatrix[0][0], rotMatrix[1][0], rotMatrix[2][0]);
        y = Vec3(rotMatrix[0][1], rotMatrix[1][1], rotMatrix[2][1]);
        z = Vec3(rotMatrix[0][2], rotMatrix[1][2], rotMatrix[2][2]);
    } else {
        // World space (default)
        x = Vec3(1, 0, 0);
        y = Vec3(0, 1, 0);
        z = Vec3(0, 0, 1);
    }
}

void RotationGizmo::Draw(Shader* shader, const Camera& camera) {
    if (!m_Enabled || !m_Transform) return;
    
    Vec3 pos = m_Transform->position;
    m_Scale = GetScreenScale(pos, camera) * m_GizmoSize;
    
    Vec3 x, y, z;
    GetAxes(x, y, z);
    
    float radius = m_Scale;
    
    // Draw 3 Circles
    DrawCircle(shader, pos, x, radius, 
        (m_HoverAxis == GizmoAxis::X || m_DragAxis == GizmoAxis::X) ? m_ColorSelected : m_ColorX);
        
    DrawCircle(shader, pos, y, radius, 
        (m_HoverAxis == GizmoAxis::Y || m_DragAxis == GizmoAxis::Y) ? m_ColorSelected : m_ColorY);
        
    DrawCircle(shader, pos, z, radius, 
        (m_HoverAxis == GizmoAxis::Z || m_DragAxis == GizmoAxis::Z) ? m_ColorSelected : m_ColorZ);
}

bool RotationGizmo::OnMousePress(const Ray& ray) {
    if (m_HoverAxis != GizmoAxis::None) {
        m_IsDragging = true;
        m_DragAxis = m_HoverAxis;
        m_DragOriginalRot = m_Transform->rotation;
        
        Vec3 pos = m_Transform->position;
        
        // Find intersection point on the plane of the rotation circle
        Vec3 normal;
        if (m_DragAxis == GizmoAxis::X) normal = Vec3(1,0,0);
        else if (m_DragAxis == GizmoAxis::Y) normal = Vec3(0,1,0);
        else normal = Vec3(0,0,1);
        
        float t = RayIntersectPlane(ray, normal, pos);
        if (t > 0) {
            Vec3 hit = ray.origin + ray.direction * t;
            m_DragStartVec = (hit - pos).Normalized();
        } else {
            m_IsDragging = false; // Failed to hit plane
            return false;
        }
        
        return true;
    }
    return false;
}

void RotationGizmo::OnMouseRelease() {
    m_IsDragging = false;
    m_DragAxis = GizmoAxis::None;
}

void RotationGizmo::OnMouseDrag(const Ray& ray, const Camera& camera) {
    if (!m_IsDragging || !m_Transform) return;
    
    Vec3 pos = m_Transform->position;
    
    // Project ray onto rotation plane
    Vec3 normal;
    if (m_DragAxis == GizmoAxis::X) normal = Vec3(1,0,0);
    else if (m_DragAxis == GizmoAxis::Y) normal = Vec3(0,1,0);
    else normal = Vec3(0,0,1);
    
    float t = RayIntersectPlane(ray, normal, pos);
    if (t > 0) {
        Vec3 hit = ray.origin + ray.direction * t;
        Vec3 currentVec = (hit - pos).Normalized();
        
        // Calculate angle between startVec and currentVec
        // Cross product gives us sin(theta) and axis
        // Dot product gives cos(theta)
        
        Vec3 start = m_DragStartVec;
        Vec3 current = currentVec;
        
        // Project start and current onto the 2D plane defined by normal?
        // Actually since we are working in 3D aligned to axes, it's easier to use atan2.
        
        auto GetAngle = [&](const Vec3& v) -> float {
            if (m_DragAxis == GizmoAxis::X) return atan2(v.z, v.y); // Y-Z plane
            if (m_DragAxis == GizmoAxis::Y) return atan2(v.x, v.z); // Z-X plane (Check handedness)
            return atan2(v.y, v.x); // X-Y plane
        };
        
        float angleStart = GetAngle(start);
        float angleCurrent = GetAngle(current);
        
        float deltaAngle = angleCurrent - angleStart;
        
        // Handle wrapping
        if (deltaAngle > 3.14159f) deltaAngle -= 2.0f * 3.14159f;
        if (deltaAngle < -3.14159f) deltaAngle += 2.0f * 3.14159f;
        
        // Convert to degrees
        float degrees = deltaAngle * 180.0f / 3.14159f;
        
        // Snapping Logic (Relative)
        if (m_SnappingEnabled && m_SnapValue > 0.001f) {
            // We want to snap the *delta* or the *total angle*?
            // "Discrete steps" feel: Accumulate 'degrees' and only apply when it crosses threshold?
            // Or Round(NewRot) = Round(Start + Delta).
            
            // Let's do Absolute snapping for now, it's often more useful for precise alignment (e.g. 90 deg).
            // But if user starts at 45.5, snapping to 15 might jump to 45.0. That's acceptable.
            
            // Actually, for rotation, snapping the "Delta" is tricky if we want to land on exactly 90.
            // Let's accumulate delta and snap the resulting target rotation?
            
            // Simpler: Just Snap the *New Rotation Value* for the active axis.
            
            float* targetVal = nullptr;
            if (m_DragAxis == GizmoAxis::X) targetVal = &m_DragOriginalRot.x;
            else if (m_DragAxis == GizmoAxis::Y) targetVal = &m_DragOriginalRot.y;
            else targetVal = &m_DragOriginalRot.z;
            
            float rawNewVal = *targetVal + degrees;
            float snappedVal = std::round(rawNewVal / m_SnapValue) * m_SnapValue;
            
            // Apply diff
            degrees = snappedVal - *targetVal;
        }

        Vec3 newRot = m_DragOriginalRot;
        if (m_DragAxis == GizmoAxis::X) { newRot.x += degrees; } 
        else if (m_DragAxis == GizmoAxis::Y) { newRot.y += degrees; }
        else { newRot.z += degrees; }
        
        m_Transform->rotation = newRot;
    }
}

bool RotationGizmo::OnMouseMove(const Ray& ray) {
    if (!m_Transform) return false;
    
    Vec3 pos = m_Transform->position;
    float scale = m_Scale;
    float radius = scale;
    float threshold = scale * 0.1f;
    
    // Ray-Torus intersection is hard.
    // Approximate with Ray-Plane intersection + Distance check to circle ring.
    
    auto CheckCircle = [&](GizmoAxis axis, const Vec3& normal) -> bool {
        float t = RayIntersectPlane(ray, normal, pos);
        if (t > 0) {
            Vec3 hit = ray.origin + ray.direction * t;
            float dist = (hit - pos).Length();
            if (std::abs(dist - radius) < threshold) {
                m_HoverAxis = axis;
                return true;
            }
        }
        return false;
    };
    
    if (CheckCircle(GizmoAxis::X, Vec3(1,0,0))) return true;
    if (CheckCircle(GizmoAxis::Y, Vec3(0,1,0))) return true;
    if (CheckCircle(GizmoAxis::Z, Vec3(0,0,1))) return true;
    
    m_HoverAxis = GizmoAxis::None;
    return false;
}
