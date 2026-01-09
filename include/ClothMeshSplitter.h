#pragma once

#include "Math/Vec3.h"
#include <vector>
#include <memory>

// Forward declarations
class ClothTearPattern;
class SpatialHashGrid;

/**
 * @brief Utility for splitting cloth meshes
 * 
 * Provides algorithms for splitting cloth meshes into separate pieces
 * when tearing occurs. Handles particle duplication, triangle splitting,
 * and connected component detection.
 */
class ClothMeshSplitter {
public:
    /**
     * @brief Result of mesh splitting operation
     */
    struct SplitResult {
        // Piece 1
        std::vector<Vec3> piece1Positions;
        std::vector<int> piece1Indices;
        std::vector<int> piece1OriginalParticles; // Map to original indices
        
        // Piece 2
        std::vector<Vec3> piece2Positions;
        std::vector<int> piece2Indices;
        std::vector<int> piece2OriginalParticles;
        
        // Tear edge particles (for visualization)
        std::vector<int> tearEdgeParticles;
        
        bool success;
        
        SplitResult() : success(false) {}
    };

    /**
     * @brief Split mesh at a specific particle
     * @param positions Original particle positions
     * @param indices Original triangle indices (3 per triangle)
     * @param tearParticle Particle index to split at
     * @return Split result with two pieces
     */
    static SplitResult SplitAtParticle(
        const std::vector<Vec3>& positions,
        const std::vector<int>& indices,
        int tearParticle
    );

    /**
     * @brief Split mesh along a line
     * @param positions Original particle positions
     * @param indices Original triangle indices
     * @param start Line start position in world space
     * @param end Line end position in world space
     * @return Split result with two pieces
     */
    static SplitResult SplitAlongLine(
        const std::vector<Vec3>& positions,
        const std::vector<int>& indices,
        const Vec3& start,
        const Vec3& end
    );

    /**
     * @brief Split mesh using a tear pattern
     * @param positions Original particle positions
     * @param indices Original triangle indices
     * @param pattern Tear pattern to apply
     * @param position Pattern application position (world space)
     * @param direction Pattern orientation direction (normalized)
     * @param scale Pattern scale multiplier
     * @param spatialGrid Optional spatial grid for performance optimization
     * @return Split result with pieces
     */
    static SplitResult SplitWithPattern(
        const std::vector<Vec3>& positions,
        const std::vector<int>& indices,
        std::shared_ptr<ClothTearPattern> pattern,
        const Vec3& position,
        const Vec3& direction,
        float scale,
        const SpatialHashGrid* spatialGrid = nullptr
    );

private:
    /**
     * @brief Find connected components in mesh
     * @param indices Triangle indices
     * @param particleCount Total number of particles
     * @param removedParticles Particles to exclude from graph
     * @param components Output: list of connected components
     */
    static void FindConnectedComponents(
        const std::vector<int>& indices,
        int particleCount,
        const std::vector<int>& removedParticles,
        std::vector<std::vector<int>>& components
    );

    /**
     * @brief Build mesh piece from component
     * @param originalPositions Original particle positions
     * @param originalIndices Original triangle indices
     * @param component Particle indices in this component
     * @param outPositions Output particle positions
     * @param outIndices Output triangle indices
     * @param outOriginalMap Output mapping to original indices
     */
    static void BuildMeshPiece(
        const std::vector<Vec3>& originalPositions,
        const std::vector<int>& originalIndices,
        const std::vector<int>& component,
        std::vector<Vec3>& outPositions,
        std::vector<int>& outIndices,
        std::vector<int>& outOriginalMap
    );

    /**
     * @brief Remap triangle indices to new particle indices
     * @param originalIndices Original indices
     * @param particleMap Map from old to new indices
     * @param newIndices Output remapped indices
     */
    static void RemapIndices(
        const std::vector<int>& originalIndices,
        const std::vector<int>& particleMap,
        std::vector<int>& newIndices
    );
};
