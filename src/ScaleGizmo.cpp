#include "ScaleGizmo.h"
#include "Camera.h"
#include "Transform.h"

void ScaleGizmo::GetAxes(Vec3& x, Vec3& y, Vec3& z) {
    x = Vec3(1, 0, 0);
    y = Vec3(0, 1, 0);
    z = Vec3(0, 0, 1);
}

void ScaleGizmo::Draw(Shader* shader, const Camera& camera) {
    if (!m_Enabled || !m_Transform) return;
    
    Vec3 pos = m_Transform->position;
    m_Scale = GetScreenScale(pos, camera);
    
    Vec3 x, y, z;
    GetAxes(x, y, z);
    
    // Draw Axis Lines (same as translation arrows but with cubes at end)
    DrawArrow(shader, pos, pos + x * m_Scale, 
        (m_HoverAxis == GizmoAxis::X || m_DragAxis == GizmoAxis::X) ? m_ColorSelected : m_ColorX, 
        m_Scale * 0.1f);
        
    DrawArrow(shader, pos, pos + y * m_Scale, 
        (m_HoverAxis == GizmoAxis::Y || m_DragAxis == GizmoAxis::Y) ? m_ColorSelected : m_ColorY,
        m_Scale * 0.1f);
        
    DrawArrow(shader, pos, pos + z * m_Scale, 
        (m_HoverAxis == GizmoAxis::Z || m_DragAxis == GizmoAxis::Z) ? m_ColorSelected : m_ColorZ,
        m_Scale * 0.1f);
        
    // Draw Center Box for Uniform Scale
    DrawCube(shader, pos, Vec3(m_Scale * 0.15f, m_Scale * 0.15f, m_Scale * 0.15f),
        (m_HoverAxis == GizmoAxis::XYZ || m_DragAxis == GizmoAxis::XYZ) ? m_ColorSelected : Vec3(0.5f, 0.5f, 0.5f));
}

bool ScaleGizmo::OnMousePress(const Ray& ray) {
    if (m_HoverAxis != GizmoAxis::None) {
        m_IsDragging = true;
        m_DragAxis = m_HoverAxis;
        m_DragOriginalScale = m_Transform->scale;
        
        // For Scale, we can use the exact same logic as Translation to find "Distance dragged along axis"
        // And for Uniform, drag along camera up/right?
        
        // Let's store the hit point on the axis/plane
        Vec3 pos = m_Transform->position;
        Vec3 x, y, z;
        GetAxes(x, y, z);
        
        // Similar to TranslationGizmo logic to find start point
        // ... (Simplified: use same logic for axis hit)
        
        if (m_DragAxis == GizmoAxis::XYZ) {
             // Center drag
             // Intersect with plane facing camera
             // ...
             // Just store simple projection?
             m_DragStartPoint = ray.origin + ray.direction * 1.0f; // Placeholder
             // Better: Project ray closest to object center?
             float dist = RayClosestPoint(ray, pos);
             m_DragStartPoint = ray.origin + ray.direction * dist;
        } else {
             // Axis drag
             Vec3 axis;
             if (m_DragAxis == GizmoAxis::X) axis = x;
             else if (m_DragAxis == GizmoAxis::Y) axis = y;
             else axis = z;
             
             Vec3 w0 = pos - ray.origin;
             float a = axis.Dot(axis);
             float b = axis.Dot(ray.direction);
             float c = ray.direction.Dot(ray.direction);
             float d = axis.Dot(w0);
             float e = ray.direction.Dot(w0);
             
             float denom = a*c - b*b;
             if (denom > 1e-6f) {
                 float sc = (b*e - c*d) / denom; 
                 m_DragStartPoint = pos + axis * sc;
             } else {
                 m_DragStartPoint = pos;
             }
        }
        
        return true;
    }
    return false;
}

void ScaleGizmo::OnMouseRelease() {
    m_IsDragging = false;
    m_DragAxis = GizmoAxis::None;
}

void ScaleGizmo::OnMouseDrag(const Ray& ray, const Camera& camera) {
    if (!m_IsDragging || !m_Transform) return;
    
    Vec3 pos = m_Transform->position;
    Vec3 x, y, z;
    GetAxes(x, y, z);
    
    Vec3 currentPoint;
    bool valid = false;
    
    if (m_DragAxis == GizmoAxis::XYZ) {
        // Uniform drag
         float dist = RayClosestPoint(ray, pos);
         currentPoint = ray.origin + ray.direction * dist;
         
         // Delta from start
         Vec3 delta = currentPoint - m_DragStartPoint;
         // Project delta onto Camera Up+Right or just Up?
         // Let's say dragging UP/RIGHT increases scale.
         
         Vec3 camUp = camera.GetPosition().y > pos.y ? Vec3(0,1,0) : Vec3(0,1,0); 
         // Just use Y axis of screen?
         // Actually, simple distance based?
         
         float scaleChange = delta.y + delta.x; // Very rough
         // Better: Dot product with camera Up and Right?
         
         float factor = 1.0f + scaleChange;
         m_Transform->scale = m_DragOriginalScale * factor;
         
    } else {
        // Axis drag
        Vec3 axis;
        if (m_DragAxis == GizmoAxis::X) axis = x;
        else if (m_DragAxis == GizmoAxis::Y) axis = y;
        else axis = z;
        
        // Similar to Translation logic
        Vec3 camToObj = pos - camera.GetPosition();
        Vec3 planeNormal = axis.Cross(camToObj).Cross(axis).Normalized();
        
        float t = RayIntersectPlane(ray, planeNormal, pos);
        if (t > 0) {
            Vec3 planeHit = ray.origin + ray.direction * t;
            Vec3 toHit = planeHit - pos;
            float projection = toHit.Dot(axis);
            currentPoint = pos + axis * projection;
            
            // Calculate scale factor based on distance from center relative to start?
            // Or delta?
            
            // Delta approach
            // StartPoint was at 'sc' distance from center?
            // No, StartPoint was the 3D world intersection.
            
            Vec3 startVector = m_DragStartPoint - pos;
            float startDist = startVector.Dot(axis);
            
            Vec3 currentVector = currentPoint - pos;
            float currentDist = currentVector.Dot(axis);
            
            // Sensitivity
            float delta = currentDist - startDist;
            // Scale factor = 1 + delta / InitialSize?
            // Assume initial size of gizmo handled is m_Scale?
            
            float factor = 1.0f + (delta / m_Scale); // Sensitivity depends on gizmo visual size
            
            Vec3 newScale = m_DragOriginalScale;
            if (m_DragAxis == GizmoAxis::X) newScale.x *= factor;
            else if (m_DragAxis == GizmoAxis::Y) newScale.y *= factor;
            else newScale.z *= factor;
            
            m_Transform->scale = newScale;
        }
    }
}

bool ScaleGizmo::OnMouseMove(const Ray& ray) {
    if (!m_Transform) return false;
    
    // Reuse TranslationGizmo logic for axes
    // Add logic for Center Box
    
    Vec3 pos = m_Transform->position;
    float scale = m_Scale;
    float boxSize = scale * 0.15f;
    
    // Check Center Box
    Vec3 boxMin = pos - Vec3(boxSize, boxSize, boxSize) * 0.5f;
    Vec3 boxMax = pos + Vec3(boxSize, boxSize, boxSize) * 0.5f;
    
    float t;
    if (RayIntersectBox(ray, boxMin, boxMax, t)) {
        m_HoverAxis = GizmoAxis::XYZ;
        return true;
    }
    
    // Helper for "Ray - Segment" distance matches TranslationGizmo
    Vec3 x, y, z;
    GetAxes(x, y, z);
     float arrowLen = scale;
     float closeDist = scale * 0.1f;
    
     auto CheckAxis = [&](const Vec3& axis, GizmoAxis type) -> bool {
        Vec3 pA = pos;
        Vec3 pB = pos + axis * arrowLen;
        
        Vec3 w0 = ray.origin - pA;
        float a = ray.direction.Dot(ray.direction);
        float b = ray.direction.Dot(axis);
        float c = axis.Dot(axis);
        float d = ray.direction.Dot(w0);
        float e = axis.Dot(w0);
        
        float denom = a*c - b*b;
        if (denom < 1e-5f) return false; 
        
        float sc = (b*d - a*e) / denom;
        
        if (sc < 0.0f || sc > 1.0f) return false; 
        
        float tc = (b*e - c*d) / denom;
        if (tc < 0) return false; 
        
        Vec3 rayP = ray.origin + ray.direction * tc;
        
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
