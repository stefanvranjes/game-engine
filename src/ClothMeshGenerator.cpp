#include "ClothMeshGenerator.h"
#include <cmath>

ClothDesc ClothMeshGenerator::CreateRectangularCloth(
    float width,
    float height,
    int segmentsX,
    int segmentsY,
    const Vec3& position)
{
    ClothDesc desc;
    
    // Calculate particle count
    int particlesX = segmentsX + 1;
    int particlesY = segmentsY + 1;
    desc.particleCount = particlesX * particlesY;
    
    // Allocate particle positions
    desc.particlePositions = new Vec3[desc.particleCount];
    
    // Generate particle grid
    float stepX = width / segmentsX;
    float stepY = height / segmentsY;
    float startX = position.x - width * 0.5f;
    float startY = position.y + height * 0.5f; // Start from top
    
    for (int y = 0; y < particlesY; ++y) {
        for (int x = 0; x < particlesX; ++x) {
            int index = y * particlesX + x;
            desc.particlePositions[index] = Vec3(
                startX + x * stepX,
                startY - y * stepY,
                position.z
            );
        }
    }
    
    // Generate triangles (2 per quad)
    desc.triangleCount = segmentsX * segmentsY * 2;
    desc.triangleIndices = new int[desc.triangleCount * 3];
    
    int triIndex = 0;
    for (int y = 0; y < segmentsY; ++y) {
        for (int x = 0; x < segmentsX; ++x) {
            int i0 = y * particlesX + x;
            int i1 = y * particlesX + (x + 1);
            int i2 = (y + 1) * particlesX + x;
            int i3 = (y + 1) * particlesX + (x + 1);
            
            // Triangle 1
            desc.triangleIndices[triIndex++] = i0;
            desc.triangleIndices[triIndex++] = i2;
            desc.triangleIndices[triIndex++] = i1;
            
            // Triangle 2
            desc.triangleIndices[triIndex++] = i1;
            desc.triangleIndices[triIndex++] = i2;
            desc.triangleIndices[triIndex++] = i3;
        }
    }
    
    desc.particleMass = 0.1f; // 100g per particle
    desc.gravity = Vec3(0, -9.81f, 0);
    
    return desc;
}

ClothDesc ClothMeshGenerator::CreateFromMesh(const Mesh& mesh) {
    ClothDesc desc;
    
    const auto& vertices = mesh.GetVertices();
    const auto& indices = mesh.GetIndices();
    
    desc.particleCount = static_cast<int>(vertices.size());
    desc.particlePositions = new Vec3[desc.particleCount];
    
    for (size_t i = 0; i < vertices.size(); ++i) {
        desc.particlePositions[i] = vertices[i].Position;
    }
    
    desc.triangleCount = static_cast<int>(indices.size()) / 3;
    desc.triangleIndices = new int[indices.size()];
    
    for (size_t i = 0; i < indices.size(); ++i) {
        desc.triangleIndices[i] = indices[i];
    }
    
    desc.particleMass = 0.1f;
    desc.gravity = Vec3(0, -9.81f, 0);
    
    return desc;
}

ClothDesc ClothMeshGenerator::CreateCircularCloth(
    float radius,
    int segments,
    const Vec3& position)
{
    ClothDesc desc;
    
    // Center particle + ring particles
    int rings = segments / 2;
    desc.particleCount = 1 + (rings * segments);
    desc.particlePositions = new Vec3[desc.particleCount];
    
    // Center particle
    desc.particlePositions[0] = position;
    
    // Generate rings
    int particleIndex = 1;
    for (int ring = 1; ring <= rings; ++ring) {
        float ringRadius = radius * (static_cast<float>(ring) / rings);
        for (int seg = 0; seg < segments; ++seg) {
            float angle = (static_cast<float>(seg) / segments) * 2.0f * 3.14159f;
            desc.particlePositions[particleIndex++] = Vec3(
                position.x + ringRadius * std::cos(angle),
                position.y,
                position.z + ringRadius * std::sin(angle)
            );
        }
    }
    
    // Generate triangles
    std::vector<int> triangles;
    
    // Center triangles
    for (int seg = 0; seg < segments; ++seg) {
        int next = (seg + 1) % segments;
        triangles.push_back(0);
        triangles.push_back(1 + seg);
        triangles.push_back(1 + next);
    }
    
    // Ring triangles
    for (int ring = 0; ring < rings - 1; ++ring) {
        int ringStart = 1 + ring * segments;
        int nextRingStart = 1 + (ring + 1) * segments;
        
        for (int seg = 0; seg < segments; ++seg) {
            int next = (seg + 1) % segments;
            
            // Triangle 1
            triangles.push_back(ringStart + seg);
            triangles.push_back(nextRingStart + seg);
            triangles.push_back(ringStart + next);
            
            // Triangle 2
            triangles.push_back(ringStart + next);
            triangles.push_back(nextRingStart + seg);
            triangles.push_back(nextRingStart + next);
        }
    }
    
    desc.triangleCount = static_cast<int>(triangles.size()) / 3;
    desc.triangleIndices = new int[triangles.size()];
    for (size_t i = 0; i < triangles.size(); ++i) {
        desc.triangleIndices[i] = triangles[i];
    }
    
    desc.particleMass = 0.1f;
    desc.gravity = Vec3(0, -9.81f, 0);
    
    return desc;
}

void ClothMeshGenerator::FreeClothDesc(ClothDesc& desc) {
    if (desc.particlePositions) {
        delete[] desc.particlePositions;
        desc.particlePositions = nullptr;
    }
    
    if (desc.triangleIndices) {
        delete[] desc.triangleIndices;
        desc.triangleIndices = nullptr;
    }
    
    desc.particleCount = 0;
    desc.triangleCount = 0;
}

void ClothMeshGenerator::GenerateUVCoordinates(
    std::vector<Vec3>& positions,
    std::vector<Vec2>& uvs,
    float width,
    float height)
{
    uvs.resize(positions.size());
    
    for (size_t i = 0; i < positions.size(); ++i) {
        uvs[i].x = (positions[i].x + width * 0.5f) / width;
        uvs[i].y = (positions[i].y + height * 0.5f) / height;
    }
}
