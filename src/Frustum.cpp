#include "Frustum.h"

Frustum::Frustum() {
    // Initialize with default planes
}

void Frustum::ExtractFromMatrix(const Mat4& viewProj) {
    // Extract the 6 frustum planes from the view-projection matrix
    // Based on Gribb & Hartmann method
    
    const float* m = viewProj.m;
    
    // Left plane: m[3] + m[0]
    m_Planes[LEFT].normal.x = m[3] + m[0];
    m_Planes[LEFT].normal.y = m[7] + m[4];
    m_Planes[LEFT].normal.z = m[11] + m[8];
    m_Planes[LEFT].distance = m[15] + m[12];
    m_Planes[LEFT].Normalize();
    
    // Right plane: m[3] - m[0]
    m_Planes[RIGHT].normal.x = m[3] - m[0];
    m_Planes[RIGHT].normal.y = m[7] - m[4];
    m_Planes[RIGHT].normal.z = m[11] - m[8];
    m_Planes[RIGHT].distance = m[15] - m[12];
    m_Planes[RIGHT].Normalize();
    
    // Bottom plane: m[3] + m[1]
    m_Planes[BOTTOM].normal.x = m[3] + m[1];
    m_Planes[BOTTOM].normal.y = m[7] + m[5];
    m_Planes[BOTTOM].normal.z = m[11] + m[9];
    m_Planes[BOTTOM].distance = m[15] + m[13];
    m_Planes[BOTTOM].Normalize();
    
    // Top plane: m[3] - m[1]
    m_Planes[TOP].normal.x = m[3] - m[1];
    m_Planes[TOP].normal.y = m[7] - m[5];
    m_Planes[TOP].normal.z = m[11] - m[9];
    m_Planes[TOP].distance = m[15] - m[13];
    m_Planes[TOP].Normalize();
    
    // Near plane: m[3] + m[2]
    m_Planes[PLANE_NEAR].normal.x = m[3] + m[2];
    m_Planes[PLANE_NEAR].normal.y = m[7] + m[6];
    m_Planes[PLANE_NEAR].normal.z = m[11] + m[10];
    m_Planes[PLANE_NEAR].distance = m[15] + m[14];
    m_Planes[PLANE_NEAR].Normalize();
    
    // Far plane: m[3] - m[2]
    m_Planes[PLANE_FAR].normal.x = m[3] - m[2];
    m_Planes[PLANE_FAR].normal.y = m[7] - m[6];
    m_Planes[PLANE_FAR].normal.z = m[11] - m[10];
    m_Planes[PLANE_FAR].distance = m[15] - m[14];
    m_Planes[PLANE_FAR].Normalize();
}

bool Frustum::ContainsAABB(const AABB& aabb) const {
    // Test AABB against all 6 frustum planes
    // If the AABB is completely outside any plane, it's not visible
    
    for (int i = 0; i < 6; ++i) {
        const Plane& plane = m_Planes[i];
        
        // Get the positive vertex (farthest point in the direction of the plane normal)
        Vec3 positiveVertex(
            plane.normal.x > 0 ? aabb.max.x : aabb.min.x,
            plane.normal.y > 0 ? aabb.max.y : aabb.min.y,
            plane.normal.z > 0 ? aabb.max.z : aabb.min.z
        );
        
        // If the positive vertex is behind the plane, the AABB is completely outside
        if (plane.DistanceToPoint(positiveVertex) < 0) {
            return false;
        }
    }
    
    // AABB intersects or is inside the frustum
    return true;
}
