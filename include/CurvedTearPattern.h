#pragma once

#include "SoftBodyTearPattern.h"

/**
 * @brief Curved tear pattern using Bezier curves
 */
class CurvedTearPattern : public SoftBodyTearPattern {
public:
    /**
     * @brief Constructor
     * @param width Width of tear curve
     * @param curvature Curve intensity (0 = straight, 1 = max curve)
     */
    CurvedTearPattern(float width = 0.1f, float curvature = 0.5f);

    std::vector<int> SelectTetrahedra(
        const Vec3* vertices,
        int vertexCount,
        const int* tetrahedra,
        int tetrahedronCount,
        const Vec3& startPoint,
        const Vec3& endPoint
    ) const override;

    PatternType GetType() const override { return PatternType::Curved; }

    /**
     * @brief Set curvature amount
     */
    void SetCurvature(float curvature) { m_Curvature = curvature; }

    /**
     * @brief Set explicit control point for Bezier curve
     */
    void SetControlPoint(const Vec3& point) { 
        m_ControlPoint = point;
        m_UseExplicitControl = true;
    }

    /**
     * @brief Use auto-generated control point
     */
    void UseAutoControl() { m_UseExplicitControl = false; }

private:
    float m_Width;
    float m_Curvature;
    Vec3 m_ControlPoint;
    bool m_UseExplicitControl;

    /**
     * @brief Evaluate quadratic Bezier curve at parameter t
     */
    Vec3 EvaluateBezier(float t, const Vec3& p0, const Vec3& p1, const Vec3& p2) const;

    /**
     * @brief Auto-generate control point
     */
    Vec3 GenerateControlPoint(const Vec3& start, const Vec3& end) const;

    /**
     * @brief Find closest point on Bezier curve
     */
    float FindClosestPointOnCurve(
        const Vec3& point,
        const Vec3& p0, const Vec3& p1, const Vec3& p2,
        int samples = 20
    ) const;
};
