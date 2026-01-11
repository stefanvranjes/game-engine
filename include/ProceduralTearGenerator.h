#pragma once

#include "Math/Vec3.h"
#include "SoftBodyTearSystem.h"
#include <vector>

/**
 * @brief Generates procedural tear paths based on stress analysis
 * 
 * This class analyzes stress distribution in soft bodies and generates
 * realistic tear propagation paths that follow stress lines and principal
 * stress directions.
 */
class ProceduralTearGenerator {
public:
    /**
     * @brief Information about a generated tear path
     */
    struct TearPath {
        std::vector<int> tetrahedronSequence;  // Sequence of tets along path
        std::vector<Vec3> pathPoints;          // 3D points along tear path
        Vec3 propagationDirection;             // Current propagation direction
        float totalEnergy;                     // Total energy in this path
        bool isComplete;                       // Whether path has finished propagating
    };

    /**
     * @brief Generate tear path from initial tear
     * 
     * @param initialTear The initial tear that triggers propagation
     * @param currentPositions Current vertex positions
     * @param restPositions Rest (undeformed) vertex positions
     * @param tetrahedronIndices Tetrahedral mesh indices
     * @param tetrahedronCount Number of tetrahedra
     * @param stressData Stress data for all tetrahedra
     * @param energyThreshold Minimum energy required to continue propagation
     * @param maxPropagationSteps Maximum number of tets to propagate through
     * @return Generated tear path
     */
    static TearPath GenerateTearPath(
        const SoftBodyTearSystem::TearInfo& initialTear,
        const Vec3* currentPositions,
        const Vec3* restPositions,
        const int* tetrahedronIndices,
        int tetrahedronCount,
        const SoftBodyTearSystem::StressData* stressData,
        float energyThreshold,
        int maxPropagationSteps
    );

    /**
     * @brief Calculate principal stress direction for a tetrahedron
     * 
     * Computes the direction of maximum tensile stress, which indicates
     * the most likely direction for tear propagation.
     * 
     * @param tetrahedronIndex Index of the tetrahedron
     * @param currentPositions Current vertex positions
     * @param restPositions Rest vertex positions
     * @param tetrahedronIndices Tetrahedral mesh indices
     * @return Principal stress direction (normalized)
     */
    static Vec3 CalculatePrincipalStressDirection(
        int tetrahedronIndex,
        const Vec3* currentPositions,
        const Vec3* restPositions,
        const int* tetrahedronIndices
    );

    /**
     * @brief Find neighboring tetrahedron with highest stress
     * 
     * @param currentTetIndex Current tetrahedron index
     * @param tetrahedronIndices Tetrahedral mesh indices
     * @param tetrahedronCount Number of tetrahedra
     * @param stressData Stress data for all tetrahedra
     * @param visitedTets Set of already visited tetrahedra
     * @param preferredDirection Preferred propagation direction
     * @return Index of next tetrahedron, or -1 if none found
     */
    static int FindNextTetrahedron(
        int currentTetIndex,
        const int* tetrahedronIndices,
        int tetrahedronCount,
        const SoftBodyTearSystem::StressData* stressData,
        const std::vector<bool>& visitedTets,
        const Vec3& preferredDirection
    );

    /**
     * @brief Calculate energy available for tear propagation
     * 
     * @param tetIndex Tetrahedron index
     * @param stressData Stress data
     * @param currentPositions Current positions
     * @param restPositions Rest positions
     * @param tetrahedronIndices Tetrahedral indices
     * @return Available energy for propagation
     */
    static float CalculateTearEnergy(
        int tetIndex,
        const SoftBodyTearSystem::StressData& stressData,
        const Vec3* currentPositions,
        const Vec3* restPositions,
        const int* tetrahedronIndices
    );

private:
    /**
     * @brief Build adjacency list for tetrahedral mesh
     */
    static void BuildAdjacencyList(
        const int* tetrahedronIndices,
        int tetrahedronCount,
        std::vector<std::vector<int>>& outAdjacency
    );

    /**
     * @brief Calculate tetrahedron center
     */
    static Vec3 CalculateTetrahedronCenter(
        const Vec3& v0, const Vec3& v1,
        const Vec3& v2, const Vec3& v3
    );

    /**
     * @brief Check if two tetrahedra share a face
     */
    static bool TetrahedraShareFace(
        const int* tet1,
        const int* tet2
    );
};
