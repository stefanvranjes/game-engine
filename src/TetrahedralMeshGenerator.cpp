#include "TetrahedralMeshGenerator.h"
#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <iostream>

TetrahedralMeshGenerator::TetrahedralMesh TetrahedralMeshGenerator::Generate(
    const Vec3* surfaceVertices,
    int surfaceVertexCount,
    const int* surfaceTriangles,
    int surfaceTriangleCount,
    float resolution)
{
    // Use voxel-based generation as default
    return GenerateVoxelBased(surfaceVertices, surfaceVertexCount, surfaceTriangles, surfaceTriangleCount, resolution);
}

TetrahedralMeshGenerator::TetrahedralMesh TetrahedralMeshGenerator::GenerateVoxelBased(
    const Vec3* surfaceVertices,
    int surfaceVertexCount,
    const int* surfaceTriangles,
    int surfaceTriangleCount,
    float voxelSize)
{
    TetrahedralMesh result;
    
    // Calculate bounding box
    AABB bounds = CalculateBounds(surfaceVertices, surfaceVertexCount);
    
    // Expand bounds slightly
    bounds.min = bounds.min - Vec3(voxelSize, voxelSize, voxelSize);
    bounds.max = bounds.max + Vec3(voxelSize, voxelSize, voxelSize);
    
    // Calculate grid dimensions
    int gridX = static_cast<int>(std::ceil((bounds.max.x - bounds.min.x) / voxelSize));
    int gridY = static_cast<int>(std::ceil((bounds.max.y - bounds.min.y) / voxelSize));
    int gridZ = static_cast<int>(std::ceil((bounds.max.z - bounds.min.z) / voxelSize));
    
    std::cout << "Generating tetrahedral mesh with grid: " << gridX << "x" << gridY << "x" << gridZ << std::endl;
    
    // Create voxel grid and test which voxels are inside the mesh
    std::vector<Voxel> voxels;
    voxels.reserve(gridX * gridY * gridZ);
    
    for (int z = 0; z < gridZ; ++z) {
        for (int y = 0; y < gridY; ++y) {
            for (int x = 0; x < gridX; ++x) {
                Vec3 voxelPos(
                    bounds.min.x + x * voxelSize + voxelSize * 0.5f,
                    bounds.min.y + y * voxelSize + voxelSize * 0.5f,
                    bounds.min.z + z * voxelSize + voxelSize * 0.5f
                );
                
                Voxel voxel;
                voxel.position = voxelPos;
                voxel.isInterior = IsPointInsideMesh(voxelPos, surfaceVertices, surfaceTriangles, surfaceTriangleCount);
                voxel.isBoundary = false; // Can be enhanced to detect boundary voxels
                
                if (voxel.isInterior) {
                    voxels.push_back(voxel);
                }
            }
        }
    }
    
    std::cout << "Found " << voxels.size() << " interior voxels" << std::endl;
    
    // Generate tetrahedra from interior voxels
    for (const auto& voxel : voxels) {
        CreateTetrahedraFromVoxel(voxel.position, voxelSize, result.vertices, result.indices);
    }
    
    result.tetrahedronCount = static_cast<int>(result.indices.size() / 4);
    
    // Create surface mapping (map each surface vertex to nearest tet vertex)
    result.surfaceMapping.resize(surfaceVertexCount);
    for (int i = 0; i < surfaceVertexCount; ++i) {
        const Vec3& surfaceVert = surfaceVertices[i];
        
        // Find nearest tetrahedral vertex
        float minDist = std::numeric_limits<float>::max();
        int nearestIdx = 0;
        
        for (size_t j = 0; j < result.vertices.size(); ++j) {
            float dist = (result.vertices[j] - surfaceVert).LengthSquared();
            if (dist < minDist) {
                minDist = dist;
                nearestIdx = static_cast<int>(j);
            }
        }
        
        result.surfaceMapping[i] = nearestIdx;
    }
    
    std::cout << "Generated " << result.tetrahedronCount << " tetrahedra with " 
              << result.vertices.size() << " vertices" << std::endl;
    
    return result;
}

TetrahedralMeshGenerator::TetrahedralMesh TetrahedralMeshGenerator::GenerateSimple(
    const Vec3* surfaceVertices,
    int surfaceVertexCount)
{
    TetrahedralMesh result;
    
    if (surfaceVertexCount < 4) {
        std::cerr << "Need at least 4 vertices to create tetrahedra" << std::endl;
        return result;
    }
    
    // Calculate center point
    Vec3 center(0, 0, 0);
    for (int i = 0; i < surfaceVertexCount; ++i) {
        center = center + surfaceVertices[i];
    }
    center = center * (1.0f / surfaceVertexCount);
    
    // Add center as first vertex
    result.vertices.push_back(center);
    
    // Add all surface vertices
    for (int i = 0; i < surfaceVertexCount; ++i) {
        result.vertices.push_back(surfaceVertices[i]);
    }
    
    // Create tetrahedra by connecting center to surface triangles
    // This is a very simple approach - just creates a star-shaped tet mesh
    // For better quality, would need proper Delaunay tetrahedralization
    
    // Simple approach: create tets from center to groups of 3 nearby vertices
    for (int i = 0; i < surfaceVertexCount - 2; i += 3) {
        result.indices.push_back(0); // Center
        result.indices.push_back(i + 1);
        result.indices.push_back(i + 2);
        result.indices.push_back((i + 3) % surfaceVertexCount + 1);
    }
    
    result.tetrahedronCount = static_cast<int>(result.indices.size() / 4);
    
    // Simple 1:1 mapping for surface vertices
    result.surfaceMapping.resize(surfaceVertexCount);
    for (int i = 0; i < surfaceVertexCount; ++i) {
        result.surfaceMapping[i] = i + 1; // +1 because center is at index 0
    }
    
    return result;
}

TetrahedralMeshGenerator::AABB TetrahedralMeshGenerator::CalculateBounds(const Vec3* vertices, int count) {
    AABB bounds;
    bounds.min = Vec3(std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
    bounds.max = Vec3(std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest());
    
    for (int i = 0; i < count; ++i) {
        const Vec3& v = vertices[i];
        bounds.min.x = std::min(bounds.min.x, v.x);
        bounds.min.y = std::min(bounds.min.y, v.y);
        bounds.min.z = std::min(bounds.min.z, v.z);
        bounds.max.x = std::max(bounds.max.x, v.x);
        bounds.max.y = std::max(bounds.max.y, v.y);
        bounds.max.z = std::max(bounds.max.z, v.z);
    }
    
    return bounds;
}

bool TetrahedralMeshGenerator::IsPointInsideMesh(
    const Vec3& point,
    const Vec3* vertices,
    const int* triangles,
    int triangleCount)
{
    // Simple ray casting algorithm
    // Cast a ray from the point in +X direction and count intersections
    int intersectionCount = 0;
    Vec3 rayDir(1, 0, 0);
    
    for (int i = 0; i < triangleCount; ++i) {
        const Vec3& v0 = vertices[triangles[i * 3 + 0]];
        const Vec3& v1 = vertices[triangles[i * 3 + 1]];
        const Vec3& v2 = vertices[triangles[i * 3 + 2]];
        
        // Möller-Trumbore ray-triangle intersection
        Vec3 edge1 = v1 - v0;
        Vec3 edge2 = v2 - v0;
        Vec3 h = rayDir.Cross(edge2);
        float a = edge1.Dot(h);
        
        if (std::abs(a) < 0.00001f) continue; // Ray parallel to triangle
        
        float f = 1.0f / a;
        Vec3 s = point - v0;
        float u = f * s.Dot(h);
        
        if (u < 0.0f || u > 1.0f) continue;
        
        Vec3 q = s.Cross(edge1);
        float v = f * rayDir.Dot(q);
        
        if (v < 0.0f || u + v > 1.0f) continue;
        
        float t = f * edge2.Dot(q);
        
        if (t > 0.00001f) { // Intersection ahead of ray origin
            intersectionCount++;
        }
    }
    
    // Odd number of intersections = inside
    return (intersectionCount % 2) == 1;
}

void TetrahedralMeshGenerator::CreateTetrahedraFromVoxel(
    const Vec3& voxelPos,
    float voxelSize,
    std::vector<Vec3>& outVertices,
    std::vector<int>& outIndices)
{
    // Create 8 corners of the voxel
    float half = voxelSize * 0.5f;
    Vec3 corners[8] = {
        voxelPos + Vec3(-half, -half, -half), // 0
        voxelPos + Vec3( half, -half, -half), // 1
        voxelPos + Vec3( half,  half, -half), // 2
        voxelPos + Vec3(-half,  half, -half), // 3
        voxelPos + Vec3(-half, -half,  half), // 4
        voxelPos + Vec3( half, -half,  half), // 5
        voxelPos + Vec3( half,  half,  half), // 6
        voxelPos + Vec3(-half,  half,  half)  // 7
    };
    
    // Add vertices and get their indices
    int indices[8];
    for (int i = 0; i < 8; ++i) {
        indices[i] = FindOrAddVertex(corners[i], outVertices);
    }
    
    // Decompose cube into 5 tetrahedra
    // Standard decomposition to avoid bias
    int tetIndices[5][4] = {
        {0, 1, 2, 5},
        {0, 2, 3, 7},
        {0, 5, 7, 4},
        {2, 5, 6, 7},
        {0, 2, 5, 7}
    };
    
    for (int i = 0; i < 5; ++i) {
        outIndices.push_back(indices[tetIndices[i][0]]);
        outIndices.push_back(indices[tetIndices[i][1]]);
        outIndices.push_back(indices[tetIndices[i][2]]);
        outIndices.push_back(indices[tetIndices[i][3]]);
    }
}

int TetrahedralMeshGenerator::FindOrAddVertex(const Vec3& vertex, std::vector<Vec3>& vertices, float epsilon) {
    // Check if vertex already exists
    for (size_t i = 0; i < vertices.size(); ++i) {
        if ((vertices[i] - vertex).LengthSquared() < epsilon * epsilon) {
            return static_cast<int>(i);
        }
    }
    
    // Add new vertex
    vertices.push_back(vertex);
    return static_cast<int>(vertices.size() - 1);
}
