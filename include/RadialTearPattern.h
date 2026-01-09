#pragma once

#include "SoftBodyTearPattern.h"

/**
 * @brief Radial burst tear pattern
 */
class RadialTearPattern : public SoftBodyTearPattern {
public:
    /**
     * @brief Constructor
     * @param rayCount Number of rays emanating from center
     * @param rayWidth Width of each ray
     */
    RadialTearPattern(int rayCount = 8, float rayWidth = 0.05f);

    std::vector<int> SelectTetrahedra(
        const Vec3* vertices,
        int vertexCount,
        const int* tetrahedra,
        int tetrahedronCount,
        const Vec3& center,
        const Vec3& direction  // Up vector for orientation
    ) const override;

    PatternType GetType() const override { return PatternType::Radial; }

    /**
     * @brief Set number of rays
     */
    void SetRayCount(int count) { m_RayCount = count; }

    /**
     * @brief Set radius of burst
     */
    void SetRadius(float radius) { m_Radius = radius; }

    /**
     * @brief Set width of each ray
     */
    void SetRayWidth(float width) { m_RayWidth = width; }

private:
    int m_RayCount;
    float m_RayWidth;
    float m_Radius;

    /**
     * @brief Generate ray endpoints around center
     */
    std::vector<Vec3> GenerateRayEndpoints(
        const Vec3& center,
        const Vec3& upVector
    ) const;
};
