#include "TearPatternGizmo.h"
#include "PhysXSoftBody.h"
#include "Camera.h"
#include "Shader.h"
#include "GLExtensions.h"

TearPatternGizmo::TearPatternGizmo() {
    m_Type = GizmoType::None;
}

void TearPatternGizmo::Update(float deltaTime) {
    // Update logic if needed
}

void TearPatternGizmo::Draw(Shader* shader, const Camera& camera) {
    if (!m_Enabled || !m_Pattern || !m_SoftBody) return;
    
    // Draw pattern path
    DrawPatternPath(shader, camera);
    
    // Draw affected tetrahedra if preview is active
    if (m_ShowPreview) {
        DrawAffectedTetrahedra(shader, camera);
    }
    
    // Draw control points
    if (m_StartPointSet || m_EndPointSet) {
        DrawControlPoints(shader, camera);
    }
}

void TearPatternGizmo::SetPattern(std::unique_ptr<SoftBodyTearPattern> pattern) {
    m_Pattern = std::move(pattern);
    
    // Reset preview when pattern changes
    m_ShowPreview = false;
    m_AffectedTetrahedra.clear();
}

void TearPatternGizmo::SetPoints(const Vec3& startPoint, const Vec3& endPoint) {
    m_StartPoint = startPoint;
    m_EndPoint = endPoint;
    m_StartPointSet = true;
    m_EndPointSet = true;
    
    // Update preview if active
    if (m_ShowPreview) {
        UpdatePreview();
    }
}

void TearPatternGizmo::ShowPreview(bool show) {
    m_ShowPreview = show;
    
    if (show && m_StartPointSet && m_EndPointSet) {
        UpdatePreview();
    } else if (!show) {
        m_AffectedTetrahedra.clear();
    }
}

void TearPatternGizmo::ExecuteTear() {
    if (!m_Pattern || !m_SoftBody || !m_StartPointSet || !m_EndPointSet) {
        return;
    }
    
    // Execute tear along pattern
    m_SoftBody->TearAlongPattern(*m_Pattern, m_StartPoint, m_EndPoint);
    
    // Clear preview after execution
    m_ShowPreview = false;
    m_AffectedTetrahedra.clear();
}

void TearPatternGizmo::UpdatePreview() {
    if (!m_Pattern || !m_SoftBody || !m_StartPointSet || !m_EndPointSet) {
        m_AffectedTetrahedra.clear();
        return;
    }
    
    // Get soft body mesh data
    int vertexCount = m_SoftBody->GetVertexCount();
    std::vector<Vec3> vertices(vertexCount);
    m_SoftBody->GetVertexPositions(vertices.data());
    
    // Get tetrahedral mesh data (we need access to this)
    // For now, we'll just clear the preview
    // TODO: Add method to PhysXSoftBody to get tetrahedral indices
    m_AffectedTetrahedra.clear();
}

bool TearPatternGizmo::OnMousePress(const Ray& ray) {
    if (!m_Enabled) return false;
    
    // Try to pick start or end point
    float startDist, endDist;
    bool hitStart = PickStartPoint(ray, startDist);
    bool hitEnd = PickEndPoint(ray, endDist);
    
    if (hitStart && (!hitEnd || startDist < endDist)) {
        m_DraggingStart = true;
        m_IsDragging = true;
        m_DragPlaneNormal = Vec3(0, 0, 1); // TODO: Calculate from camera
        return true;
    } else if (hitEnd) {
        m_DraggingEnd = true;
        m_IsDragging = true;
        m_DragPlaneNormal = Vec3(0, 0, 1); // TODO: Calculate from camera
        return true;
    }
    
    // If no point hit, set new start point
    if (!m_StartPointSet) {
        Vec3 planeNormal(0, 0, 1);
        Vec3 planePoint(0, 0, 0);
        float t = RayIntersectPlane(ray, planeNormal, planePoint);
        if (t > 0) {
            m_StartPoint = ray.origin + ray.direction * t;
            m_StartPointSet = true;
            return true;
        }
    } else if (!m_EndPointSet) {
        Vec3 planeNormal(0, 0, 1);
        float t = RayIntersectPlane(ray, planeNormal, m_StartPoint);
        if (t > 0) {
            m_EndPoint = ray.origin + ray.direction * t;
            m_EndPointSet = true;
            
            // Auto-update preview
            if (m_ShowPreview) {
                UpdatePreview();
            }
            return true;
        }
    }
    
    return false;
}

void TearPatternGizmo::OnMouseRelease() {
    m_IsDragging = false;
    m_DraggingStart = false;
    m_DraggingEnd = false;
}

void TearPatternGizmo::OnMouseDrag(const Ray& ray, const Camera& camera) {
    if (!m_IsDragging) return;
    
    Vec3 dragPoint = m_DraggingStart ? m_StartPoint : m_EndPoint;
    float t = RayIntersectPlane(ray, m_DragPlaneNormal, dragPoint);
    
    if (t > 0) {
        Vec3 newPoint = ray.origin + ray.direction * t;
        
        if (m_DraggingStart) {
            m_StartPoint = newPoint;
        } else if (m_DraggingEnd) {
            m_EndPoint = newPoint;
        }
        
        // Update preview if active
        if (m_ShowPreview && m_StartPointSet && m_EndPointSet) {
            UpdatePreview();
        }
    }
}

bool TearPatternGizmo::OnMouseMove(const Ray& ray) {
    // Hover feedback could be added here
    return false;
}

void TearPatternGizmo::DrawPatternPath(Shader* shader, const Camera& camera) {
    if (!m_StartPointSet || !m_EndPointSet) return;
    
    // Draw line from start to end
    // TODO: For curved patterns, draw the actual curve
    // For now, just draw a straight line
    
    // Use existing line drawing from Gizmo base class or implement simple line rendering
}

void TearPatternGizmo::DrawAffectedTetrahedra(Shader* shader, const Camera& camera) {
    if (m_AffectedTetrahedra.empty()) return;
    
    // Draw highlighted tetrahedra
    // This would require access to tetrahedral mesh data
    // For now, placeholder
}

void TearPatternGizmo::DrawControlPoints(Shader* shader, const Camera& camera) {
    if (m_StartPointSet) {
        Vec3 color = m_DraggingStart ? m_ColorSelected : m_PatternColor;
        DrawCube(shader, m_StartPoint, Vec3(m_PointSize, m_PointSize, m_PointSize), color);
    }
    
    if (m_EndPointSet) {
        Vec3 color = m_DraggingEnd ? m_ColorSelected : m_PatternColor;
        DrawCube(shader, m_EndPoint, Vec3(m_PointSize, m_PointSize, m_PointSize), color);
    }
}

bool TearPatternGizmo::PickStartPoint(const Ray& ray, float& distance) {
    if (!m_StartPointSet) return false;
    
    float t = RayClosestPoint(ray, m_StartPoint);
    if (t > 0) {
        Vec3 closestOnRay = ray.origin + ray.direction * t;
        float dist = (closestOnRay - m_StartPoint).Length();
        
        if (dist < m_PointSize * 2.0f) {
            distance = t;
            return true;
        }
    }
    
    return false;
}

bool TearPatternGizmo::PickEndPoint(const Ray& ray, float& distance) {
    if (!m_EndPointSet) return false;
    
    float t = RayClosestPoint(ray, m_EndPoint);
    if (t > 0) {
        Vec3 closestOnRay = ray.origin + ray.direction * t;
        float dist = (closestOnRay - m_EndPoint).Length();
        
        if (dist < m_PointSize * 2.0f) {
            distance = t;
            return true;
        }
    }
    
    return false;
}
