#include "TranslationGizmo.h"
#include "Camera.h"
#include "Transform.h"
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include "Math/Mat4.h"


void TranslationGizmo::GetAxes(Vec3& x, Vec3& y, Vec3& z) {
    if (m_UseLocalSpace && m_Transform) {
        // Transform world axes by the object's rotation
        // Convert Euler angles to rotation matrix using the same order as Transform::GetModelMatrix
        float radX = m_Transform->rotation.x * 3.14159f / 180.0f;
        float radY = m_Transform->rotation.y * 3.14159f / 180.0f;
        float radZ = m_Transform->rotation.z * 3.14159f / 180.0f;

        Mat4 rotMatrix = Mat4::RotateY(radY) * Mat4::RotateX(radX) * Mat4::RotateZ(radZ);

        // Extract columns for X, Y, Z axes (Column-major storage)
        x = Vec3(rotMatrix.m[0], rotMatrix.m[1], rotMatrix.m[2]);
        y = Vec3(rotMatrix.m[4], rotMatrix.m[5], rotMatrix.m[6]);
        z = Vec3(rotMatrix.m[8], rotMatrix.m[9], rotMatrix.m[10]);

    } else {
        // World space (default)
        x = Vec3(1, 0, 0);
        y = Vec3(0, 1, 0);
        z = Vec3(0, 0, 1);
    }
}

void TranslationGizmo::Draw(Shader* shader, const Camera& camera) {
    if (!m_Enabled || !m_Transform) return;
    
    Vec3 pos = m_Transform->position;
    m_Scale = GetScreenScale(pos, camera) * m_GizmoSize;
    
    Vec3 x, y, z;
    GetAxes(x, y, z);
    
    // Draw Axis Arrows
    // X Axis (Red)
    DrawArrow(shader, pos, pos + x * m_Scale, 
        (m_HoverAxis == GizmoAxis::X || m_DragAxis == GizmoAxis::X) ? m_ColorSelected : m_ColorX, 
        m_Scale * 0.1f);
        
    // Y Axis (Green)
    DrawArrow(shader, pos, pos + y * m_Scale, 
        (m_HoverAxis == GizmoAxis::Y || m_DragAxis == GizmoAxis::Y) ? m_ColorSelected : m_ColorY,
        m_Scale * 0.1f);
        
    // Z Axis (Blue)
    DrawArrow(shader, pos, pos + z * m_Scale, 
        (m_HoverAxis == GizmoAxis::Z || m_DragAxis == GizmoAxis::Z) ? m_ColorSelected : m_ColorZ,
        m_Scale * 0.1f);
        
    // Draw Planar Handles (little quads)
    float quadOffset = m_Scale * 0.3f;
    float quadSize = m_Scale * 0.2f;
    
    // XY Plane (Blue, because it's perpendicular to Z)
    DrawQuad(shader, pos + (x + y) * quadOffset, x * quadSize, y * quadSize,
        (m_HoverAxis == GizmoAxis::XY || m_DragAxis == GizmoAxis::XY) ? m_ColorSelected : m_ColorZ);
        
    // XZ Plane (Green, perp to Y)
    DrawQuad(shader, pos + (x + z) * quadOffset, x * quadSize, z * quadSize,
        (m_HoverAxis == GizmoAxis::XZ || m_DragAxis == GizmoAxis::XZ) ? m_ColorSelected : m_ColorY);
        
    // YZ Plane (Red, perp to X)
    DrawQuad(shader, pos + (y + z) * quadOffset, y * quadSize, z * quadSize,
        (m_HoverAxis == GizmoAxis::YZ || m_DragAxis == GizmoAxis::YZ) ? m_ColorSelected : m_ColorX);
}

bool TranslationGizmo::OnMousePress(const Ray& ray) {
    if (m_HoverAxis != GizmoAxis::None) {
        m_IsDragging = true;
        m_DragAxis = m_HoverAxis;
        m_DragOriginalPos = m_Transform->position;

        // Calculate initial intersection point to establish offset
        Vec3 x, y, z;
        GetAxes(x, y, z);
        Vec3 pos = m_Transform->position;
        
        // Logic: Find the point on the gizmo geometry (or virtual plane) where the ray hits.
        // This 'HitPoint' is used to calculate the offset from the object center.
        // During drag, we find the new HitPoint and apply the offset.

        Vec3 hitPoint;
        bool hit = false;
        
        // Planar Drag
        if (m_DragAxis == GizmoAxis::XY || m_DragAxis == GizmoAxis::XZ || m_DragAxis == GizmoAxis::YZ) {
             Vec3 normal;
             if (m_DragAxis == GizmoAxis::XY) normal = z;
             else if (m_DragAxis == GizmoAxis::XZ) normal = y;
             else normal = x;
             
             float t = RayIntersectPlane(ray, normal, pos);
             if (t > 0) {
                 hitPoint = ray.origin + ray.direction * t;
                 hit = true;
             }
        } 
        // Axis Drag (X, Y, Z)
        else {
             Vec3 axis;
             if (m_DragAxis == GizmoAxis::X) axis = x;
             else if (m_DragAxis == GizmoAxis::Y) axis = y;
             else axis = z;
             
             // Project ray onto axis line to find closest point
             // Point on Axis = pos + axis * t
             // Ray = origin + dir * s
             // We want to minimize distance(pos + axis*t, origin + dir*s)
             
             Vec3 w0 = pos - ray.origin;
             float a = axis.Dot(axis);
             float b = axis.Dot(ray.direction);
             float c = ray.direction.Dot(ray.direction);
             float d = axis.Dot(w0);
             float e = ray.direction.Dot(w0);
             
             float denom = a*c - b*b;
             if (denom > 1e-6f) {
                 float sc = (b*e - c*d) / denom; // Distance along axis
                 hitPoint = pos + axis * sc;
                 hit = true;
             } else {
                 // Parallel, just use projection of ray origin?
                 // Or closest point to pos.
                 hitPoint = pos; // Fallback
                 hit = true;
             }
        }
        
        if (hit) {
             m_DragOffset = hitPoint - m_Transform->position;
        } else {
             m_DragOffset = Vec3(0,0,0);
        }
        
        return true;
    }
    return false;
}

void TranslationGizmo::OnMouseRelease() {
    m_IsDragging = false;
    m_DragAxis = GizmoAxis::None;
}

void TranslationGizmo::OnMouseDrag(const Ray& ray, const Camera& camera) {
    if (!m_IsDragging || !m_Transform) return;
    
    Vec3 x, y, z;
    GetAxes(x, y, z);
    
    Vec3 newPos = m_Transform->position; // Start with current
    
    // Calculate new hit point based on ray
    Vec3 hitPoint;
    bool hit = false;
    
    if (m_DragAxis == GizmoAxis::XY || m_DragAxis == GizmoAxis::XZ || m_DragAxis == GizmoAxis::YZ) {
        Vec3 normal;
        if (m_DragAxis == GizmoAxis::XY) normal = z;
        else if (m_DragAxis == GizmoAxis::XZ) normal = y;
        else normal = x;
        
        float t = RayIntersectPlane(ray, normal, m_Transform->position); 
        // Note: we intersect with plane passing through CURRENT position to get delta? 
        // OR better: passing through ORIGINAL position + offset?
        
        // Correct way:
        // We want newPosition such that (newPosition + offset) lies on the ray (approx).
        // Actually simplest is: Ray intersects the plane defined by "Original Obj Pos" and "Axis Normal".
        // The hit point tells us where the mouse is in that plane.
        // Then NewObjPos = HitPoint - Offset.
        // Wait, if we move the object, the plane moves? 
        // Standard behavior: The drag plane is infinite and fixed at the start of drag? 
        // YES. Fixed at start of drag is best.
        
        t = RayIntersectPlane(ray, normal, m_DragOriginalPos);
        if (t > 0) {
            hitPoint = ray.origin + ray.direction * t;
            hit = true;
        }
    }
    else {
        // Axis Drag
        Vec3 axis;
        if (m_DragAxis == GizmoAxis::X) axis = x;
        else if (m_DragAxis == GizmoAxis::Y) axis = y;
        else axis = z;
        
        // Define a plane for the ray to intersect.
        // Best plane contains the axis and faces the camera.
        Vec3 camDir = (m_DragOriginalPos - camera.GetPosition()).Normalized();
        Vec3 planeNormal = axis.Cross(camDir).Cross(axis).Normalized();
        
        float t = RayIntersectPlane(ray, planeNormal, m_DragOriginalPos);
        if (t > 0) {
            Vec3 planeHit = ray.origin + ray.direction * t;
            // Project this point onto the axis line passing through Original Pos
            Vec3 toHit = planeHit - m_DragOriginalPos;
            float projection = toHit.Dot(axis);
            hitPoint = m_DragOriginalPos + axis * projection;
            hit = true;
        }
    }
    
    if (hit) {
        // Apply offset
        newPos = hitPoint - m_DragOffset;
        
        // Snapping Logic
        if (m_SnappingEnabled && m_SnapValue > 0.001f) {
             // Snap to nearest grid increment
             auto Snap = [&](float val) -> float {
                 return std::round(val / m_SnapValue) * m_SnapValue;
             };
             
             newPos.x = Snap(newPos.x);
             newPos.y = Snap(newPos.y);
             newPos.z = Snap(newPos.z);
        }
        
        m_Transform->position = newPos;
    }
}

// I will rewrite OnMousePress/Drag to be consistent with a simpler logic
// 1. OnMousePress(ray): Check intersections. If hit, store m_DragAxis.
//    Also calculate "HitPoint" on the virtual plane/line based on that ray.
//    Store m_DragOffset = HitPoint - m_Transform->position.
//    WAIT: For Axis drag, calculating HitPoint without Camera orientation (for the virtual plane) is hard.
//    But we can pick a default plane: e.g. if X axis, use XY or XZ depending on which is more facing the ray?
//    Standard: Use two planes and pick best.
    
// Refined Draw/Trace:
bool TranslationGizmo::OnMouseMove(const Ray& ray) {
    if (!m_Transform) return false;
    
    Vec3 pos = m_Transform->position;
    float scale = m_Scale; // Use last frame scale (approx ok)
    
    Vec3 x, y, z;
    GetAxes(x, y, z);
    
    // Bounds for checks
    float arrowLen = scale;
    float tipRadius = scale * 0.1f; // Rough cylinder radius
    
    // Check Planar Handles first (they are inside)
    float quadOffset = scale * 0.3f;
    float quadSize = scale * 0.2f;
    
    // Quad Plane intersections
    float t;
    
    // XY Quad
    // Center: pos + (x+y)*quadOffset
    // Check intersection with plane Z
    // And check if localized point is within quad range
    
    // It's easier to check Ray vs Quad directly?
    // Let's use simple Box approximation for quads, or distance checks.
    
    // ... Implementation of robust "Ray vs Oriented Box" or similar ...
    
    // Using RayIntersectPlane for Quads:
    // XY Plane:
    float t_xy = RayIntersectPlane(ray, z, pos);
    if (t_xy > 0) {
        Vec3 hit = ray.origin + ray.direction * t_xy;
        Vec3 local = hit - pos;
        if (local.x > quadOffset && local.x < quadOffset + quadSize &&
            local.y > quadOffset && local.y < quadOffset + quadSize) {
            m_HoverAxis = GizmoAxis::XY;
            return true;
        }
    }
    
    // XZ Plane:
    float t_xz = RayIntersectPlane(ray, y, pos);
    if (t_xz > 0) {
        Vec3 hit = ray.origin + ray.direction * t_xz;
        Vec3 local = hit - pos;
        if (local.x > quadOffset && local.x < quadOffset + quadSize &&
            local.z > quadOffset && local.z < quadOffset + quadSize) {
            m_HoverAxis = GizmoAxis::XZ;
            return true;
        }
    }
    
    // YZ Plane:
    float t_yz = RayIntersectPlane(ray, x, pos);
    if (t_yz > 0) {
        Vec3 hit = ray.origin + ray.direction * t_yz;
        Vec3 local = hit - pos;
        if (local.y > quadOffset && local.y < quadOffset + quadSize &&
            local.z > quadOffset && local.z < quadOffset + quadSize) {
            m_HoverAxis = GizmoAxis::YZ;
            return true;
        }
    }
    
    // Check Arrows (Cylinders/Lines)
    // Distance from Ray to Line Segment
    float closeDist = scale * 0.1f; // Threshold
    
    // X Axis
    // Closest point on Ray to Segment (pos, pos+x*len)
    // Simplified: Distance between two infinite lines, check if point on Axis is within 0..len
    
    // Helper for "Ray - Segment" distance not in base class. 
    // Implementing inline robust check for hover:
    
    auto CheckAxis = [&](const Vec3& axis, GizmoAxis type) -> bool {
        Vec3 pA = pos;
        Vec3 pB = pos + axis * arrowLen;
        
        // Vector from ray origin to pA
        Vec3 w0 = ray.origin - pA;
        float a = ray.direction.Dot(ray.direction);
        float b = ray.direction.Dot(axis);
        float c = axis.Dot(axis);
        float d = ray.direction.Dot(w0);
        float e = axis.Dot(w0);
        
        float denom = a*c - b*b;
        if (denom < 1e-5f) return false; // Parallel
        
        // Closest point on Axis line parameter sc
        float sc = (b*d - a*e) / denom;
        
        if (sc < 0.0f || sc > 1.0f) return false; // Out of segment
        
        // Find closest point on ray
        float tc = (b*e - c*d) / denom;
        if (tc < 0) return false; // Behind camera
        
        Vec3 rayP = ray.origin + ray.direction * tc;
        Vec3 axisP = pA + axis * (sc * arrowLen); // sc is normalized? No, math above depends on axis length if c is length squared.
        // Let's normalize axis first in math?
        // Let's stick to normalized axis for math:
        // axis is (1,0,0) so length 1.
        // pB = pA + axis * arrowLen.
        // Math above assumes vectors 'ray.direction' and 'axis'.
        // ray.dir is normalized? Yes. 'axis' is normalized? Yes. arrowLength is separate.
        // So 'sc' is distance along axis.
        
        if (sc < 0.0f || sc > arrowLen) return false;
        
        if ((rayP - (pA + axis*sc)).Length() < closeDist) return true;
        return false;
    };
    
    if (CheckAxis(x, GizmoAxis::X)) { m_HoverAxis = GizmoAxis::X; return true; }
    if (CheckAxis(y, GizmoAxis::Y)) { m_HoverAxis = GizmoAxis::Y; return true; }
    if (CheckAxis(z, GizmoAxis::Z)) { m_HoverAxis = GizmoAxis::Z; return true; }
    
    m_HoverAxis = GizmoAxis::None;
    return false;
}

// Redoing OnMouseDrag to be complete and correct
// I'll put the "HitPoint" calculation into a helper in the .cpp only or usage in Drag
