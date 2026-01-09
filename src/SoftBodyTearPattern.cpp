#include "SoftBodyTearPattern.h"
#include <algorithm>
#include <cmath>

Vec3 SoftBodyTearPattern::CalculateTetrahedronCenter(
    const Vec3& v0, const Vec3& v1,
    const Vec3& v2, const Vec3& v3)
{
    return (v0 + v1 + v2 + v3) * 0.25f;
}

float SoftBodyTearPattern::DistanceToLineSegment(
    const Vec3& point,
    const Vec3& lineStart,
    const Vec3& lineEnd)
{
    Vec3 lineDir = lineEnd - lineStart;
    float lineLength = lineDir.Length();
    
    if (lineLength < 0.0001f) {
        return (point - lineStart).Length();
    }
    
    lineDir = lineDir * (1.0f / lineLength);
    
    Vec3 toPoint = point - lineStart;
    float projection = toPoint.Dot(lineDir);
    
    // Clamp to segment
    projection = std::max(0.0f, std::min(lineLength, projection));
    
    Vec3 closestPoint = lineStart + lineDir * projection;
    return (point - closestPoint).Length();
}
