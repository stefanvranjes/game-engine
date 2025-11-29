#pragma once

#include "Math/Vec3.h"
#include "Math/Mat4.h"
#include "Math/AABB.h"

struct Plane {
    Vec3 normal;
    float distance;
    
    Plane() : normal(0, 0, 0), distance(0) {}
    Plane(const Vec3& n, float d) : normal(n), distance(d) {}
    
    // Calculate signed distance from point to plane
    float DistanceToPoint(const Vec3& point) const {
        return normal.Dot(point) + distance;
    }
    
    // Normalize the plane equation
    void Normalize() {
        float length = normal.Length();
        if (length > 0.0f) {
            normal = normal / length;
            distance /= length;
        }
    }
};

class Frustum {
public:
    Frustum();
    
    // Extract frustum planes from a view-projection matrix
    void ExtractFromMatrix(const Mat4& viewProj);
    
    // Test if an AABB intersects or is inside the frustum
    bool ContainsAABB(const AABB& aabb) const;
    
private:
    Plane m_Planes[6]; // Left, Right, Bottom, Top, Near, Far
    
    enum PlaneIndex {
        LEFT,
        RIGHT,
        BOTTOM,
        TOP,
        PLANE_NEAR,
        PLANE_FAR
    };
};
