#pragma once

#include "Math/Vec3.h"
#include "QuickHull.h" // For ConvexHull struct
#include <vector>
#include <functional>

/**
 * @brief Gift Wrapping (Jarvis March) algorithm for computing 3D convex hulls
 * 
 * Implements the O(nh) Gift Wrapping algorithm where n is the number of points
 * and h is the number of hull faces. Simpler than QuickHull but typically slower
 * for large point sets. Best used as a robust fallback or for verification.
 */
class GiftWrapping {
public:
    /**
     * @brief Construct GiftWrapping with optional epsilon tolerance
     * @param epsilon Tolerance for floating-point comparisons (default: 1e-6)
     */
    explicit GiftWrapping(float epsilon = 1e-6f);
    
    /**
     * @brief Compute convex hull from point cloud
     * @param points Array of input points
     * @param count Number of points
     * @return ConvexHull structure containing hull geometry and surface area
     */
    ConvexHull ComputeHull(const Vec3* points, int count);
    
    /**
     * @brief Set epsilon tolerance for numerical comparisons
     * @param epsilon Tolerance value (should be > 0)
     */
    void SetEpsilon(float epsilon);

private:
    float m_Epsilon;
    const Vec3* m_Points;
    int m_PointCount;
    
    // Internal edge structure handling uniqueness for open edges set
    struct DirectedEdge {
        int v0, v1; // Directed: v0 -> v1
        
        DirectedEdge(int a, int b) : v0(a), v1(b) {}
        
        bool operator==(const DirectedEdge& other) const {
            return v0 == other.v0 && v1 == other.v1;
        }
    };
    
    struct DirectedEdgeHash {
        size_t operator()(const DirectedEdge& e) const {
            return std::hash<int>()(e.v0) ^ (std::hash<int>()(e.v1) << 1);
        }
    };
    
    // Helper methods
    int FindExtremePoint() const;
    int FindSecondPoint(int p0) const;
    int FindBestPointForEdge(int p0, int p1, const Vec3& refNormal) const;
    float CalculateTriangleArea(const Vec3& p0, const Vec3& p1, const Vec3& p2) const;
};
