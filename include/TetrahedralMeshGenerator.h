#pragma once

#include "Math/Vec3.h"
#include <vector>
#include <memory>

/**
 * @brief Utility class for generating tetrahedral meshes from surface meshes
 * 
 * Provides methods to convert triangle surface meshes into volumetric tetrahedral meshes
 * required for soft body physics simulation.
 */
class TetrahedralMeshGenerator {
public:
    /**
     * @brief Tetrahedral mesh data structure
     */
    struct TetrahedralMesh {
        std::vector<Vec3> vertices;           // Tetrahedral mesh vertices
        std::vector<int> indices;             // Tetrahedral indices (4 per tetrahedron)
        std::vector<int> surfaceMapping;      // Maps surface vertices to tet vertices
        int tetrahedronCount;                 // Number of tetrahedra
    };

    /**
     * @brief Generate tetrahedral mesh from surface mesh
     * 
     * @param surfaceVertices Surface mesh vertices
     * @param surfaceVertexCount Number of surface vertices
     * @param surfaceTriangles Surface mesh triangle indices
     * @param surfaceTriangleCount Number of surface triangles
     * @param resolution Voxel resolution for mesh generation (smaller = more detail)
     * @return Generated tetrahedral mesh
     */
    static TetrahedralMesh Generate(
        const Vec3* surfaceVertices,
        int surfaceVertexCount,
        const int* surfaceTriangles,
        int surfaceTriangleCount,
        float resolution = 0.1f
    );

    /**
     * @brief Generate tetrahedral mesh using simple voxel-based approach
     * 
     * Fast but lower quality. Good for real-time generation.
     * 
     * @param surfaceVertices Surface mesh vertices
     * @param surfaceVertexCount Number of surface vertices
     * @param surfaceTriangles Surface mesh triangle indices
     * @param surfaceTriangleCount Number of surface triangles
     * @param voxelSize Size of each voxel
     * @return Generated tetrahedral mesh
     */
    static TetrahedralMesh GenerateVoxelBased(
        const Vec3* surfaceVertices,
        int surfaceVertexCount,
        const int* surfaceTriangles,
        int surfaceTriangleCount,
        float voxelSize
    );

    /**
     * @brief Generate simple tetrahedral mesh for basic shapes
     * 
     * Creates a coarse tetrahedral mesh suitable for simple convex shapes.
     * Very fast, minimal quality.
     * 
     * @param surfaceVertices Surface mesh vertices
     * @param surfaceVertexCount Number of surface vertices
     * @return Generated tetrahedral mesh
     */
    static TetrahedralMesh GenerateSimple(
        const Vec3* surfaceVertices,
        int surfaceVertexCount
    );

private:
    struct Voxel {
        Vec3 position;
        bool isInterior;
        bool isBoundary;
    };

    struct AABB {
        Vec3 min;
        Vec3 max;
    };

    // Helper methods
    static AABB CalculateBounds(const Vec3* vertices, int count);
    static bool IsPointInsideMesh(const Vec3& point, const Vec3* vertices, const int* triangles, int triangleCount);
    static void CreateTetrahedraFromVoxel(const Vec3& voxelPos, float voxelSize, std::vector<Vec3>& outVertices, std::vector<int>& outIndices);
    static int FindOrAddVertex(const Vec3& vertex, std::vector<Vec3>& vertices, float epsilon = 0.001f);
};
