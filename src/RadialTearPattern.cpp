#include "RadialTearPattern.h"
#include <iostream>
#include <cmath>
#include <unordered_set>

RadialTearPattern::RadialTearPattern(int rayCount, float rayWidth)
    : m_RayCount(rayCount)
    , m_RayWidth(rayWidth)
    , m_Radius(1.0f)
{
}

std::vector<int> RadialTearPattern::SelectTetrahedra(
    const Vec3* vertices,
    int vertexCount,
    const int* tetrahedra,
    int tetrahedronCount,
    const Vec3& center,
    const Vec3& direction) const
{
    std::unordered_set<int> selectedSet;  // Use set to avoid duplicates

    // Generate ray endpoints
    std::vector<Vec3> rayEndpoints = GenerateRayEndpoints(center, direction);

    float halfWidth = m_RayWidth * 0.5f;

    // For each ray
    for (const Vec3& endpoint : rayEndpoints) {
        // For each tetrahedron
        for (int tetIdx = 0; tetIdx < tetrahedronCount; ++tetIdx) {
            // Get tetrahedron vertices
            int v0 = tetrahedra[tetIdx * 4 + 0];
            int v1 = tetrahedra[tetIdx * 4 + 1];
            int v2 = tetrahedra[tetIdx * 4 + 2];
            int v3 = tetrahedra[tetIdx * 4 + 3];

            // Calculate tetrahedron center
            Vec3 tetCenter = CalculateTetrahedronCenter(
                vertices[v0], vertices[v1],
                vertices[v2], vertices[v3]
            );

            // Calculate distance to ray line
            float distance = DistanceToLineSegment(tetCenter, center, endpoint);

            // Select if within width
            if (distance <= halfWidth) {
                selectedSet.insert(tetIdx);
            }
        }
    }

    // Convert set to vector
    std::vector<int> selectedTets(selectedSet.begin(), selectedSet.end());

    std::cout << "Radial pattern selected " << selectedTets.size() 
              << " tetrahedra (" << m_RayCount << " rays)" << std::endl;

    return selectedTets;
}

std::vector<Vec3> RadialTearPattern::GenerateRayEndpoints(
    const Vec3& center,
    const Vec3& upVector) const
{
    std::vector<Vec3> endpoints;

    // Normalize up vector
    Vec3 up = upVector.Normalized();

    // Create perpendicular vectors for the plane
    Vec3 right = up.Cross(Vec3(0, 0, 1));
    if (right.LengthSquared() < 0.01f) {
        right = up.Cross(Vec3(1, 0, 0));
    }
    right = right.Normalized();

    Vec3 forward = up.Cross(right).Normalized();

    // Generate rays in a circle
    for (int i = 0; i < m_RayCount; ++i) {
        float angle = 2.0f * 3.14159265359f * i / m_RayCount;
        float cosAngle = std::cos(angle);
        float sinAngle = std::sin(angle);

        // Ray direction in plane perpendicular to up
        Vec3 rayDir = right * cosAngle + forward * sinAngle;

        // Endpoint at radius distance
        Vec3 endpoint = center + rayDir * m_Radius;

        endpoints.push_back(endpoint);
    }

    return endpoints;
}
