#pragma once

#include "Math/Vec3.h"
#include "SoftBodyTearSystem.h"
#include <vector>
#include <unordered_map>
#include <unordered_set>

/**
 * @brief Splits tetrahedral meshes along tear lines
 */
class TetrahedralMeshSplitter {
public:
    /**
     * @brief Result of mesh splitting operation
     */
    struct SplitResult {
        // Piece 1 (original, modified)
        std::vector<Vec3> vertices1;
        std::vector<int> tetrahedra1;
        std::vector<int> vertexMapping1;  // Maps new indices to original
        
        // Piece 2 (new piece)
        std::vector<Vec3> vertices2;
        std::vector<int> tetrahedra2;
        std::vector<int> vertexMapping2;
        
        // Tear information
        std::vector<int> tearEdges;       // Pairs of vertex indices
        std::vector<Vec3> tearBoundaryPositions;  // Positions along tear line
        std::vector<Vec3> tearBoundaryNormals;    // Normals for visual effects
        float tearEnergy;                 // Energy released during split
        
        int piece1VertexCount;
        int piece2VertexCount;
        bool splitSuccessful;
    };

    /**
     * @brief Split tetrahedral mesh along detected tears
     * 
     * @param vertices Original vertex positions
     * @param vertexCount Number of vertices
     * @param tetrahedra Tetrahedral indices (4 per tet)
     * @param tetrahedronCount Number of tetrahedra
     * @param tears Detected tears
     * @return Split result with two separate pieces
     */
    static SplitResult SplitAlongTear(
        const Vec3* vertices,
        int vertexCount,
        const int* tetrahedra,
        int tetrahedronCount,
        const std::vector<SoftBodyTearSystem::TearInfo>& tears
    );

    /**
     * @brief Split tetrahedral mesh with state transfer (velocities)
     * 
     * @param vertices Original vertex positions
     * @param vertexCount Number of vertices
     * @param tetrahedra Tetrahedral indices (4 per tet)
     * @param tetrahedronCount Number of tetrahedra
     * @param tears Detected tears
     * @param velocities Current vertex velocities
     * @param outVelocities1 Output velocities for piece 1
     * @param outVelocities2 Output velocities for piece 2
     * @return Split result with two separate pieces
     */
    static SplitResult SplitWithStateTransfer(
        const Vec3* vertices,
        int vertexCount,
        const int* tetrahedra,
        int tetrahedronCount,
        const std::vector<SoftBodyTearSystem::TearInfo>& tears,
        const Vec3* velocities,
        std::vector<Vec3>& outVelocities1,
        std::vector<Vec3>& outVelocities2
    );


private:
    /**
     * @brief Build connectivity graph of tetrahedra
     */
    static void BuildConnectivityGraph(
        const int* tetrahedra,
        int tetrahedronCount,
        std::vector<std::vector<int>>& outAdjacency
    );

    /**
     * @brief Partition tetrahedra into two groups using flood fill
     */
    static void PartitionTetrahedra(
        const std::vector<std::vector<int>>& adjacency,
        const std::unordered_set<int>& tornTets,
        std::vector<int>& outPartition1,
        std::vector<int>& outPartition2
    );

    /**
     * @brief Duplicate vertices along tear line
     */
    static void DuplicateTearVertices(
        const Vec3* vertices,
        int vertexCount,
        const std::vector<SoftBodyTearSystem::TearInfo>& tears,
        const std::vector<int>& partition1,
        const std::vector<int>& partition2,
        const int* tetrahedra,
        std::unordered_map<int, int>& outVertexDuplication
    );

    /**
     * @brief Determine which partition owns each duplicated vertex
     */
    static void DetermineVertexOwnership(
        const std::vector<int>& partition1,
        const std::vector<int>& partition2,
        const int* tetrahedra,
        const std::unordered_set<int>& tearVertices,
        std::unordered_map<int, int>& outOwnership  // vertex -> partition (0 or 1)
    );

    /**
     * @brief Extract mesh for a partition
     */
    static void ExtractPartitionMesh(
        const Vec3* vertices,
        int vertexCount,
        const int* tetrahedra,
        const std::vector<int>& partition,
        const std::unordered_map<int, int>& vertexDuplication,
        const std::unordered_map<int, int>& vertexOwnership,
        int partitionId,
        std::vector<Vec3>& outVertices,
        std::vector<int>& outTetrahedra,
        std::vector<int>& outVertexMapping
    );

    /**
     * @brief Check if two tetrahedra share an edge
     */
    static bool TetrahedraShareEdge(
        const int* tet1,
        const int* tet2,
        int& outV0,
        int& outV1
    );
};
