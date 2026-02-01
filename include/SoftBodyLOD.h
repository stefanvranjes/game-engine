#pragma once

#include "Math/Vec3.h"
#include <vector>
#include <cassert>
#include <nlohmann/json.hpp>

/**
 * @brief LOD level configuration for soft body simulation
 */
struct SoftBodyLODLevel {
    int lodIndex;                    // 0 = highest quality
    float minDistance;               // Minimum distance for this LOD
    
    // Mesh data (simplified tetrahedral mesh for this LOD)
    std::vector<Vec3> vertexPositions;
    std::vector<int> tetrahedronIndices;  // 4 indices per tetrahedron
    int vertexCount;
    int tetrahedronCount;
    
    // Simulation quality settings
    int solverIterations;            // Constraint solver iterations
    int substeps;                    // Physics substeps per frame
    int updateFrequency;             // 1 = every frame, 2 = every 2 frames, etc.
    
    // Vertex mapping (for state transfer)
    std::vector<int> vertexMapping;  // Maps original vertices to LOD vertices
    
    // State
    bool isFrozen;                   // If true, disable simulation entirely
    bool hasMeshData;                // If true, mesh data is populated
    
    SoftBodyLODLevel()
        : lodIndex(0)
        , minDistance(0.0f)
        , vertexCount(0)
        , tetrahedronCount(0)
        , solverIterations(10)
        , substeps(1)
        , updateFrequency(1)
        , isFrozen(false)
        , hasMeshData(false)
    {}
};

/**
 * @brief Configuration for soft body LOD levels
 */
class SoftBodyLODConfig {
public:
    SoftBodyLODConfig();
    
    /**
     * @brief Add LOD level (must be added in order from LOD 0 to N)
     */
    void AddLODLevel(const SoftBodyLODLevel& level);
    
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
    const SoftBodyLODLevel* GetLODLevel(int index) const;
    
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
     * @param baseTetrahedra Base mesh tetrahedral indices (4 per tetrahedron)
     */
    void GenerateLODMeshes(
        const std::vector<Vec3>& basePositions,
        const std::vector<int>& baseTetrahedra
    );
    
    /**
     * @brief Create default LOD configuration
     * @param baseVertexCount Vertex count for LOD 0
     * @param baseTetrahedronCount Tetrahedron count for LOD 0
     * @return Default configuration with 4 LOD levels
     */
    static SoftBodyLODConfig CreateDefault(int baseVertexCount, int baseTetrahedronCount);
    
    /**
     * @brief Serialize LOD configuration to JSON
     * @return JSON object
     */
    nlohmann::json Serialize() const;
    
    /**
     * @brief Deserialize LOD configuration from JSON
     * @param json JSON object
     * @return Deserialized configuration
     */
    static SoftBodyLODConfig Deserialize(const nlohmann::json& json);
    
private:
    std::vector<SoftBodyLODLevel> m_Levels;
    float m_Hysteresis;  // Distance buffer to prevent LOD flickering
};
