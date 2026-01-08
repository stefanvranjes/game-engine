#pragma once

#include "Math/Vec3.h"
#include <vector>
#include <unordered_map>
#include <cmath>
#include <algorithm>

/**
 * @brief Simple Spatial Grid for spatial partitioning
 * 
 * Partitions space into a grid of cells to optimize spatial queries.
 * Useful for culling collision shapes or finding nearby objects.
 */
template <typename T>
class SpatialGrid {
public:
    struct AABB {
        Vec3 min;
        Vec3 max;

        bool Intersects(const AABB& other) const {
            return (min.x <= other.max.x && max.x >= other.min.x) &&
                   (min.y <= other.max.y && max.y >= other.min.y) &&
                   (min.z <= other.max.z && max.z >= other.min.z);
        }
    };

    explicit SpatialGrid(float cellSize = 1.0f)
        : m_CellSize(cellSize)
    {
    }

    void SetCellSize(float size) {
        m_CellSize = size;
        Clear(); // Need to re-insert if cell size changes, but for now just clear
    }

    void Clear() {
        m_Grid.clear();
        m_Items.clear();
    }

    void Insert(const T& item, const Vec3& pos, float radius) {
        AABB bounds;
        bounds.min = pos - Vec3(radius, radius, radius);
        bounds.max = pos + Vec3(radius, radius, radius);
        Insert(item, bounds);
    }

    void Insert(const T& item, const AABB& bounds) {
        int id = static_cast<int>(m_Items.size());
        m_Items.push_back({item, bounds, id});

        // Determine grid cells
        int minX = static_cast<int>(std::floor(bounds.min.x / m_CellSize));
        int minY = static_cast<int>(std::floor(bounds.min.y / m_CellSize));
        int minZ = static_cast<int>(std::floor(bounds.min.z / m_CellSize));
        
        int maxX = static_cast<int>(std::floor(bounds.max.x / m_CellSize));
        int maxY = static_cast<int>(std::floor(bounds.max.y / m_CellSize));
        int maxZ = static_cast<int>(std::floor(bounds.max.z / m_CellSize));

        for (int x = minX; x <= maxX; ++x) {
            for (int y = minY; y <= maxY; ++y) {
                for (int z = minZ; z <= maxZ; ++z) {
                    uint64_t hash = HashCell(x, y, z);
                    m_Grid[hash].push_back(id);
                }
            }
        }
    }

    void Query(const AABB& area, std::vector<T>& outResults) const {
        int minX = static_cast<int>(std::floor(area.min.x / m_CellSize));
        int minY = static_cast<int>(std::floor(area.min.y / m_CellSize));
        int minZ = static_cast<int>(std::floor(area.min.z / m_CellSize));

        int maxX = static_cast<int>(std::floor(area.max.x / m_CellSize));
        int maxY = static_cast<int>(std::floor(area.max.y / m_CellSize));
        int maxZ = static_cast<int>(std::floor(area.max.z / m_CellSize));

        // Use a visited set to avoid duplicates? 
        // Or simpler: sort and unique at the end if generic T isn't identified by ID.
        // Assuming T is small/copyable.
        // Better: Query returns unique items.
        
        // Since we stored generic T, maybe we assume T has ID or we rely on m_Items check?
        // Let's use a temporary boolean vector or visited set if m_Items is large.
        // For simplicity: std::vector<int> collectedIndices;
        std::vector<int> candidates;
        
        for (int x = minX; x <= maxX; ++x) {
            for (int y = minY; y <= maxY; ++y) {
                for (int z = minZ; z <= maxZ; ++z) {
                    uint64_t hash = HashCell(x, y, z);
                    auto it = m_Grid.find(hash);
                    if (it != m_Grid.end()) {
                        candidates.insert(candidates.end(), it->second.begin(), it->second.end());
                    }
                }
            }
        }

        // Deduplicate
        std::sort(candidates.begin(), candidates.end());
        candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());

        // Check exact bounds and output
        for (int id : candidates) {
            if (m_Items[id].bounds.Intersects(area)) {
                outResults.push_back(m_Items[id].data);
            }
        }
    }

private:
    struct ItemEntry {
        T data;
        AABB bounds;
        int id;
    };

    std::vector<ItemEntry> m_Items;
    std::unordered_map<uint64_t, std::vector<int>> m_Grid;
    float m_CellSize;

    uint64_t HashCell(int x, int y, int z) const {
        // Simple hash for 3 ints
        // Use primes to reduce collisions
        const uint64_t p1 = 73856093;
        const uint64_t p2 = 19349663;
        const uint64_t p3 = 83492791;
        return (static_cast<uint64_t>(x) * p1) ^ 
               (static_cast<uint64_t>(y) * p2) ^ 
               (static_cast<uint64_t>(z) * p3);
    }
};
