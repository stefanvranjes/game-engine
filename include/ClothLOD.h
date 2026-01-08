#pragma once

#include "Math/Vec3.h"
#include <vector>

/**
 * @brief LOD level configuration for cloth simulation
 */
struct ClothLODLevel {
    int lodIndex;                    // 0 = highest quality
    float minDistance;               // Minimum distance for this LOD
    
    // Mesh data (simplified mesh for this LOD)
    std::vector<Vec3> particlePositions;
    std::vector<int> triangleIndices;
    int particleCount;
    int triangleCount;
    
    // Simulation quality settings
    int solverIterations;            // Constraint solver iterations
    int substeps;                    // Physics substeps per frame
    int updateFrequency;             // 1 = every frame, 2 = every 2 frames, etc.
    
    // Particle mapping (for state transfer)
    std::vector<int> particleMapping; // Maps original particles to LOD particles
    
    // State
    bool isFrozen;                   // If true, disable simulation entirely
    bool hasMeshData;                // If true, mesh data is populated
    
    ClothLODLevel()
        : lodIndex(0)
        , minDistance(0.0f)
        , particleCount(0)
        , triangleCount(0)
        , solverIterations(10)
        , substeps(1)
        , updateFrequency(1)
        , isFrozen(false)
        , hasMeshData(false)
    {}
};

/**
 * @brief Configuration for cloth LOD levels
 */
class ClothLODConfig {
public:
    ClothLODConfig();
    
    /**
     * @brief Add LOD level (must be added in order from LOD 0 to N)
     */
    void AddLODLevel(const ClothLODLevel& level);
    
    /**
     * @brief Get appropriate LOD level for given distance
     * @param distance Distance from camera
     * @param currentLOD Current LOD (for hysteresis)
     * @return LOD level index
     */
    int GetLODForDistance(float distance, int currentLOD = -1) const;
    
    /**
     * @brief Get LOD level data by index
     */
    const ClothLODLevel* GetLODLevel(int index) const;
    
    /**
     * @brief Get number of LOD levels
     */
    int GetLODCount() const { return static_cast<int>(m_Levels.size()); }
    
    /**
     * @brief Set hysteresis distance (prevents LOD flickering)
     * @param value Distance buffer (e.g., 2.0m)
     */
    void SetHysteresis(float value) { m_Hysteresis = value; }
    
    /**
     * @brief Get hysteresis distance
     */
    float GetHysteresis() const { return m_Hysteresis; }
    
    /**
     * @brief Clear all LOD levels
     */
    void Clear() { m_Levels.clear(); }
    
    /**
     * @brief Generate LOD meshes from base mesh using mesh simplification
     * @param basePositions Base mesh vertex positions
     * @param baseIndices Base mesh triangle indices
     */
    void GenerateLODMeshes(
        const std::vector<Vec3>& basePositions,
        const std::vector<int>& baseIndices
    );
    
    /**
     * @brief Create default LOD configuration
     * @param baseParticleCount Particle count for LOD 0
     * @param baseTriangleCount Triangle count for LOD 0
     * @return Default configuration with 4 LOD levels
     */
    static ClothLODConfig CreateDefault(int baseParticleCount, int baseTriangleCount);
    
private:
    std::vector<ClothLODLevel> m_Levels;
    float m_Hysteresis;  // Distance buffer to prevent LOD flickering
};
