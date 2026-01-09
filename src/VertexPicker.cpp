#include "VertexPicker.h"
#include "PhysXSoftBody.h"
#include <algorithm>

VertexPicker::VertexPicker()
    : m_PickRadius(0.5f)
{
}

VertexPicker::PickResult VertexPicker::PickVertex(
    const Ray& ray,
    PhysXSoftBody* softBody,
    float maxDistance)
{
    PickResult result;
    
    if (!softBody) {
        return result;
    }
    
    // Get vertex positions
    int vertexCount = softBody->GetVertexCount();
    std::vector<Vec3> positions(vertexCount);
    softBody->GetVertexPositions(positions.data());
    
    float closestDistance = maxDistance;
    
    // Find closest vertex to ray
    for (int i = 0; i < vertexCount; ++i) {
        float distance = ray.DistanceToPoint(positions[i]);
        
        if (distance < closestDistance) {
            closestDistance = distance;
            result.hit = true;
            result.vertexIndex = i;
            result.distance = distance;
            result.worldPosition = positions[i];
        }
    }
    
    return result;
}

std::vector<int> VertexPicker::PickVerticesInRadius(
    const Vec3& center,
    float radius,
    PhysXSoftBody* softBody)
{
    std::vector<int> pickedVertices;
    
    if (!softBody) {
        return pickedVertices;
    }
    
    // Get vertex positions
    int vertexCount = softBody->GetVertexCount();
    std::vector<Vec3> positions(vertexCount);
    softBody->GetVertexPositions(positions.data());
    
    float radiusSq = radius * radius;
    
    // Find all vertices within radius
    for (int i = 0; i < vertexCount; ++i) {
        float distSq = (positions[i] - center).LengthSquared();
        
        if (distSq <= radiusSq) {
            pickedVertices.push_back(i);
        }
    }
    
    return pickedVertices;
}

float VertexPicker::CalculateDistanceToVertex(const Ray& ray, const Vec3& vertex) {
    return ray.DistanceToPoint(vertex);
}
