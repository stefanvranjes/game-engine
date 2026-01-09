#pragma once

#include "Math/Vec3.h"

/**
 * @brief Represents a ray in 3D space
 * 
 * Used for ray casting and intersection tests.
 */
class Ray {
public:
    Vec3 origin;
    Vec3 direction;  // Should be normalized
    
    Ray() : origin(0, 0, 0), direction(0, 0, -1) {}
    
    Ray(const Vec3& origin, const Vec3& direction)
        : origin(origin)
        , direction(direction)
    {
        this->direction.Normalize();
    }
    
    /**
     * @brief Get point along ray at distance t
     */
    Vec3 GetPoint(float t) const {
        return origin + direction * t;
    }
    
    /**
     * @brief Calculate closest distance from ray to a point
     * @param point Point in 3D space
     * @return Distance from ray to point
     */
    float DistanceToPoint(const Vec3& point) const {
        Vec3 toPoint = point - origin;
        float t = toPoint.Dot(direction);
        
        // If point is behind ray origin
        if (t < 0) {
            return toPoint.Length();
        }
        
        Vec3 closestPoint = origin + direction * t;
        return (point - closestPoint).Length();
    }
    
    /**
     * @brief Get parameter t for closest point on ray to a point
     */
    float GetClosestT(const Vec3& point) const {
        Vec3 toPoint = point - origin;
        return toPoint.Dot(direction);
    }
};
