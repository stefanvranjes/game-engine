#include "CurvedTearPattern.h"
#include <iostream>
#include <cmath>
#include <algorithm>

CurvedTearPattern::CurvedTearPattern(float width, float curvature)
    : m_Width(width)
    , m_Curvature(curvature)
    , m_ControlPoint(0, 0, 0)
    , m_UseExplicitControl(false)
{
}

std::vector<int> CurvedTearPattern::SelectTetrahedra(
    const Vec3* vertices,
    int vertexCount,
    const int* tetrahedra,
    int tetrahedronCount,
    const Vec3& startPoint,
    const Vec3& endPoint) const
{
    std::vector<int> selectedTets;
    float halfWidth = m_Width * 0.5f;

    // Determine control point
    Vec3 controlPoint = m_UseExplicitControl ? 
        m_ControlPoint : GenerateControlPoint(startPoint, endPoint);

    // For each tetrahedron
    for (int tetIdx = 0; tetIdx < tetrahedronCount; ++tetIdx) {
        // Get tetrahedron vertices
        int v0 = tetrahedra[tetIdx * 4 + 0];
        int v1 = tetrahedra[tetIdx * 4 + 1];
        int v2 = tetrahedra[tetIdx * 4 + 2];
        int v3 = tetrahedra[tetIdx * 4 + 3];

        // Calculate tetrahedron center
        Vec3 center = CalculateTetrahedronCenter(
            vertices[v0], vertices[v1],
            vertices[v2], vertices[v3]
        );

        // Find closest point on curve
        float closestT = FindClosestPointOnCurve(center, startPoint, controlPoint, endPoint);
        Vec3 closestPoint = EvaluateBezier(closestT, startPoint, controlPoint, endPoint);

        // Calculate distance
        float distance = (center - closestPoint).Length();

        // Select if within width
        if (distance <= halfWidth) {
            selectedTets.push_back(tetIdx);
        }
    }

    std::cout << "Curved pattern selected " << selectedTets.size() 
              << " tetrahedra" << std::endl;

    return selectedTets;
}

Vec3 CurvedTearPattern::EvaluateBezier(float t, const Vec3& p0, const Vec3& p1, const Vec3& p2) const
{
    float u = 1.0f - t;
    return u * u * p0 + 2.0f * u * t * p1 + t * t * p2;
}

Vec3 CurvedTearPattern::GenerateControlPoint(const Vec3& start, const Vec3& end) const
{
    // Midpoint
    Vec3 mid = (start + end) * 0.5f;

    // Direction from start to end
    Vec3 dir = (end - start).Normalized();

    // Perpendicular vector (assume Y-up)
    Vec3 up(0, 1, 0);
    Vec3 perpendicular = dir.Cross(up);

    // If parallel to up, use different perpendicular
    if (perpendicular.LengthSquared() < 0.01f) {
        perpendicular = dir.Cross(Vec3(1, 0, 0));
    }

    perpendicular = perpendicular.Normalized();

    // Offset midpoint by curvature
    float distance = (end - start).Length();
    return mid + perpendicular * (distance * m_Curvature);
}

float CurvedTearPattern::FindClosestPointOnCurve(
    const Vec3& point,
    const Vec3& p0, const Vec3& p1, const Vec3& p2,
    int samples) const
{
    float closestT = 0.0f;
    float minDistance = std::numeric_limits<float>::max();

    // Sample curve at intervals
    for (int i = 0; i <= samples; ++i) {
        float t = static_cast<float>(i) / samples;
        Vec3 curvePoint = EvaluateBezier(t, p0, p1, p2);
        float distance = (point - curvePoint).LengthSquared();

        if (distance < minDistance) {
            minDistance = distance;
            closestT = t;
        }
    }

    return closestT;
}
