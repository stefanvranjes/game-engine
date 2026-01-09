#pragma once

#include "Math/Vec3.h"
#include <vector>
#include <unordered_map>
#include <cstdint>

/**
 * @brief Spatial hash grid for efficient 3D spatial queries
 * 
 * Partitions 3D space into uniform grid cells to accelerate proximity queries.
 * Reduces particle lookup complexity from O(n) to O(k) where k is the number
 * of particles in the query region.
 * 
 * Performance characteristics:
 * - Build: O(n) where n = particle count
 * - Query: O(k) where k = particles in query region (typically << n)
 * - Memory: ~8 bytes per particle
 */
class SpatialHashGrid {
public:
    SpatialHashGrid();
    ~SpatialHashGrid();

    /**
     * @brief Build the spatial grid from particle positions
     * @param positions Particle positions to index
     * @param cellSize Size of each grid cell (should be ~2x typical query size)
     */
    void Build(const std::vector<Vec3>& positions, float cellSize);

    /**
     * @brief Clear the grid
     */
    void Clear();

    /**
     * @brief Query particles within an axis-aligned bounding box
     * @param min Minimum corner of AABB
     * @param max Maximum corner of AABB
     * @return Indices of particles within the AABB
     */
    std::vector<int> QueryAABB(const Vec3& min, const Vec3& max) const;

    /**
     * @brief Query particles within a sphere
     * @param center Center of sphere
     * @param radius Radius of sphere
     * @return Indices of particles within the sphere
     */
    std::vector<int> QuerySphere(const Vec3& center, float radius) const;

    /**
     * @brief Query particles near a line segment
     * @param start Start point of line segment
     * @param end End point of line segment
     * @param width Distance threshold from line
     * @return Indices of particles within width distance of the line segment
     */
    std::vector<int> QueryLineSegment(const Vec3& start, const Vec3& end, float width) const;

    /**
     * @brief Get the cell size used by this grid
     */
    float GetCellSize() const { return m_CellSize; }

    /**
     * @brief Get the number of particles in the grid
     */
    size_t GetParticleCount() const { return m_ParticleCount; }

    /**
     * @brief Check if the grid is empty
     */
    bool IsEmpty() const { return m_ParticleCount == 0; }

private:
    /**
     * @brief Compute hash key for a grid cell
     * @param x Grid cell X coordinate
     * @param y Grid cell Y coordinate
     * @param z Grid cell Z coordinate
     * @return Hash key for the cell
     */
    int64_t ComputeCellHash(int x, int y, int z) const;

    /**
     * @brief Convert world position to grid cell coordinates
     * @param position World position
     * @param outX Output grid X coordinate
     * @param outY Output grid Y coordinate
     * @param outZ Output grid Z coordinate
     */
    void WorldToGrid(const Vec3& position, int& outX, int& outY, int& outZ) const;

    /**
     * @brief Get particles in a specific grid cell
     * @param x Grid cell X coordinate
     * @param y Grid cell Y coordinate
     * @param z Grid cell Z coordinate
     * @return Pointer to particle list (nullptr if cell is empty)
     */
    const std::vector<int>* GetCell(int x, int y, int z) const;

private:
    // Grid cell storage: maps cell hash -> list of particle indices
    std::unordered_map<int64_t, std::vector<int>> m_Cells;
    
    // Grid parameters
    float m_CellSize;
    float m_InvCellSize;  // 1.0f / m_CellSize for faster division
    size_t m_ParticleCount;
    
    // Particle positions (stored for distance checks in queries)
    std::vector<Vec3> m_Positions;
};
