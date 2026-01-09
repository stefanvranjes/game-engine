#include "StraightTearPattern.h"
#include <iostream>

StraightTearPattern::StraightTearPattern(float width)
    : m_Width(width)
{
}

std::vector<int> StraightTearPattern::SelectTetrahedra(
    const Vec3* vertices,
    int vertexCount,
    const int* tetrahedra,
    int tetrahedronCount,
    const Vec3& startPoint,
    const Vec3& endPoint) const
{
    std::vector<int> selectedTets;
    float halfWidth = m_Width * 0.5f;

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

        // Calculate distance to line
        float distance = DistanceToLineSegment(center, startPoint, endPoint);

        // Select if within width
        if (distance <= halfWidth) {
            selectedTets.push_back(tetIdx);
        }
    }

    std::cout << "Straight pattern selected " << selectedTets.size() 
              << " tetrahedra" << std::endl;

    return selectedTets;
}
