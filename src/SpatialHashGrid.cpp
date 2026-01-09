#include "SpatialHashGrid.h"
#include <algorithm>
#include <cmath>

// Prime numbers for hash function to ensure good distribution
static const int64_t HASH_PRIME_X = 73856093;
static const int64_t HASH_PRIME_Y = 19349663;
static const int64_t HASH_PRIME_Z = 83492791;

SpatialHashGrid::SpatialHashGrid()
    : m_CellSize(1.0f)
    , m_InvCellSize(1.0f)
    , m_ParticleCount(0)
{
}

SpatialHashGrid::~SpatialHashGrid() {
    Clear();
}

void SpatialHashGrid::Build(const std::vector<Vec3>& positions, float cellSize) {
    Clear();
    
    if (positions.empty() || cellSize <= 0.0f) {
        return;
    }
    
    m_CellSize = cellSize;
    m_InvCellSize = 1.0f / cellSize;
    m_ParticleCount = positions.size();
    m_Positions = positions;
    
    // Reserve space to reduce allocations
    m_Cells.reserve(m_ParticleCount / 4);  // Estimate ~4 particles per cell
    
    // Insert each particle into its grid cell
    for (size_t i = 0; i < positions.size(); ++i) {
        int x, y, z;
        WorldToGrid(positions[i], x, y, z);
        
        int64_t hash = ComputeCellHash(x, y, z);
        m_Cells[hash].push_back(static_cast<int>(i));
    }
}

void SpatialHashGrid::Clear() {
    m_Cells.clear();
    m_Positions.clear();
    m_ParticleCount = 0;
}

std::vector<int> SpatialHashGrid::QueryAABB(const Vec3& min, const Vec3& max) const {
    std::vector<int> results;
    
    if (IsEmpty()) {
        return results;
    }
    
    // Convert AABB to grid coordinates
    int minX, minY, minZ;
    int maxX, maxY, maxZ;
    WorldToGrid(min, minX, minY, minZ);
    WorldToGrid(max, maxX, maxY, maxZ);
    
    // Reserve space for results
    results.reserve((maxX - minX + 1) * (maxY - minY + 1) * (maxZ - minZ + 1) * 4);
    
    // Iterate through all cells in the AABB
    for (int x = minX; x <= maxX; ++x) {
        for (int y = minY; y <= maxY; ++y) {
            for (int z = minZ; z <= maxZ; ++z) {
                const std::vector<int>* cell = GetCell(x, y, z);
                if (cell) {
                    // Add all particles from this cell
                    for (int particleIdx : *cell) {
                        const Vec3& pos = m_Positions[particleIdx];
                        
                        // Verify particle is actually within AABB
                        if (pos.x >= min.x && pos.x <= max.x &&
                            pos.y >= min.y && pos.y <= max.y &&
                            pos.z >= min.z && pos.z <= max.z) {
                            results.push_back(particleIdx);
                        }
                    }
                }
            }
        }
    }
    
    return results;
}

std::vector<int> SpatialHashGrid::QuerySphere(const Vec3& center, float radius) const {
    std::vector<int> results;
    
    if (IsEmpty() || radius <= 0.0f) {
        return results;
    }
    
    // Create AABB around sphere
    Vec3 min = center - Vec3(radius, radius, radius);
    Vec3 max = center + Vec3(radius, radius, radius);
    
    // Convert to grid coordinates
    int minX, minY, minZ;
    int maxX, maxY, maxZ;
    WorldToGrid(min, minX, minY, minZ);
    WorldToGrid(max, maxX, maxY, maxZ);
    
    float radiusSq = radius * radius;
    
    // Iterate through cells in AABB
    for (int x = minX; x <= maxX; ++x) {
        for (int y = minY; y <= maxY; ++y) {
            for (int z = minZ; z <= maxZ; ++z) {
                const std::vector<int>* cell = GetCell(x, y, z);
                if (cell) {
                    // Check each particle in cell
                    for (int particleIdx : *cell) {
                        const Vec3& pos = m_Positions[particleIdx];
                        Vec3 diff = pos - center;
                        float distSq = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
                        
                        if (distSq <= radiusSq) {
                            results.push_back(particleIdx);
                        }
                    }
                }
            }
        }
    }
    
    return results;
}

std::vector<int> SpatialHashGrid::QueryLineSegment(const Vec3& start, const Vec3& end, float width) const {
    std::vector<int> results;
    
    if (IsEmpty() || width <= 0.0f) {
        return results;
    }
    
    // Create AABB around line segment
    Vec3 min(
        std::min(start.x, end.x) - width,
        std::min(start.y, end.y) - width,
        std::min(start.z, end.z) - width
    );
    Vec3 max(
        std::max(start.x, end.x) + width,
        std::max(start.y, end.y) + width,
        std::max(start.z, end.z) + width
    );
    
    // Convert to grid coordinates
    int minX, minY, minZ;
    int maxX, maxY, maxZ;
    WorldToGrid(min, minX, minY, minZ);
    WorldToGrid(max, maxX, maxY, maxZ);
    
    // Precompute line segment data
    Vec3 lineDir = end - start;
    float lineLength = lineDir.Length();
    
    if (lineLength < 0.0001f) {
        // Degenerate line segment, treat as sphere
        return QuerySphere(start, width);
    }
    
    lineDir = lineDir / lineLength;  // Normalize
    float widthSq = width * width;
    
    // Iterate through cells in AABB
    for (int x = minX; x <= maxX; ++x) {
        for (int y = minY; y <= maxY; ++y) {
            for (int z = minZ; z <= maxZ; ++z) {
                const std::vector<int>* cell = GetCell(x, y, z);
                if (cell) {
                    // Check each particle in cell
                    for (int particleIdx : *cell) {
                        const Vec3& pos = m_Positions[particleIdx];
                        
                        // Project particle onto line segment
                        Vec3 toParticle = pos - start;
                        float projection = toParticle.Dot(lineDir);
                        
                        // Clamp projection to line segment
                        projection = std::max(0.0f, std::min(lineLength, projection));
                        
                        // Find closest point on line segment
                        Vec3 closestPoint = start + lineDir * projection;
                        
                        // Check distance
                        Vec3 diff = pos - closestPoint;
                        float distSq = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
                        
                        if (distSq <= widthSq) {
                            results.push_back(particleIdx);
                        }
                    }
                }
            }
        }
    }
    
    return results;
}

int64_t SpatialHashGrid::ComputeCellHash(int x, int y, int z) const {
    // Use prime number multiplication for good hash distribution
    // XOR combines the hash values
    return (static_cast<int64_t>(x) * HASH_PRIME_X) ^
           (static_cast<int64_t>(y) * HASH_PRIME_Y) ^
           (static_cast<int64_t>(z) * HASH_PRIME_Z);
}

void SpatialHashGrid::WorldToGrid(const Vec3& position, int& outX, int& outY, int& outZ) const {
    // Convert world position to grid cell coordinates
    // Use floor to ensure consistent cell assignment
    outX = static_cast<int>(std::floor(position.x * m_InvCellSize));
    outY = static_cast<int>(std::floor(position.y * m_InvCellSize));
    outZ = static_cast<int>(std::floor(position.z * m_InvCellSize));
}

const std::vector<int>* SpatialHashGrid::GetCell(int x, int y, int z) const {
    int64_t hash = ComputeCellHash(x, y, z);
    auto it = m_Cells.find(hash);
    
    if (it != m_Cells.end()) {
        return &it->second;
    }
    
    return nullptr;
}
