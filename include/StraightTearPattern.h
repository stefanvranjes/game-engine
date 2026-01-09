#pragma once

#include "SoftBodyTearPattern.h"

/**
 * @brief Straight line tear pattern
 */
class StraightTearPattern : public SoftBodyTearPattern {
public:
    /**
     * @brief Constructor
     * @param width Width of tear line
     */
    explicit StraightTearPattern(float width = 0.1f);

    std::vector<int> SelectTetrahedra(
        const Vec3* vertices,
        int vertexCount,
        const int* tetrahedra,
        int tetrahedronCount,
        const Vec3& startPoint,
        const Vec3& endPoint
    ) const override;

    PatternType GetType() const override { return PatternType::Straight; }

    /**
     * @brief Set tear line width
     */
    void SetWidth(float width) { m_Width = width; }

    /**
     * @brief Get tear line width
     */
    float GetWidth() const { return m_Width; }

private:
    float m_Width;  // Tear line width
};
