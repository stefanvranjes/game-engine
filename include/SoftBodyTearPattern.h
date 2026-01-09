#pragma once

#include "Math/Vec3.h"
#include <vector>
#include <memory>

/**
 * @brief Base class for soft body tear patterns
 */
class SoftBodyTearPattern {
public:
    enum class PatternType {
        Straight,    // Linear tear
        Curved,      // Bezier curve tear
        Radial,      // Burst from center
        Custom       // User-defined path
    };

    virtual ~SoftBodyTearPattern() = default;

    /**
     * @brief Select tetrahedra along pattern path
     * 
     * @param vertices Vertex positions
     * @param vertexCount Number of vertices
     * @param tetrahedra Tetrahedral indices (4 per tet)
     * @param tetrahedronCount Number of tetrahedra
     * @param startPoint Pattern start point
     * @param endPoint Pattern end point (or direction for radial)
     * @return Indices of selected tetrahedra
     */
    virtual std::vector<int> SelectTetrahedra(
        const Vec3* vertices,
        int vertexCount,
        const int* tetrahedra,
        int tetrahedronCount,
        const Vec3& startPoint,
        const Vec3& endPoint
    ) const = 0;

    /**
     * @brief Get pattern type
     */
    virtual PatternType GetType() const = 0;

protected:
    /**
     * @brief Calculate tetrahedron center
     */
    static Vec3 CalculateTetrahedronCenter(
        const Vec3& v0, const Vec3& v1,
        const Vec3& v2, const Vec3& v3
    );

    /**
     * @brief Calculate distance from point to line segment
     */
    static float DistanceToLineSegment(
        const Vec3& point,
        const Vec3& lineStart,
        const Vec3& lineEnd
    );
};
