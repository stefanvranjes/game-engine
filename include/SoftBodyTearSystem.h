#pragma once

#include "Math/Vec3.h"
#include <vector>
#include <functional>

class TearResistanceMap;

/**
 * @brief System for detecting stress and managing tears in soft body tetrahedral meshes
 */
class SoftBodyTearSystem {
public:
    /**
     * @brief Information about a detected tear
     */
    struct TearInfo {
        int tetrahedronIndex;      // Index of torn tetrahedron
        int edgeVertices[2];       // Vertices forming the torn edge
        float stress;              // Stress value that caused tear
        Vec3 tearPosition;         // World position of tear
        Vec3 tearNormal;           // Normal direction of tear
        float timestamp;           // When tear occurred
    };

    /**
     * @brief Stress data for a single tetrahedron
     */
    struct StressData {
        float edgeStress[6];       // Stress on each of 6 edges
        float volumeStress;        // Volume change stress
        bool isTorn;               // Whether this tet has torn
        int tornEdgeIndex;         // Which edge tore (-1 if none)
    };
    
    /**
     * @brief Healing tear data
     */
    struct HealingTear {
        int tetrahedronIndex;
        float healingProgress;     // 0.0 = fully torn, 1.0 = fully healed
        float timeSinceTear;       // Time since tear occurred
        float originalResistance;  // Resistance before tear
    };

    SoftBodyTearSystem();
    ~SoftBodyTearSystem();

    /**
     * @brief Detect overstressed edges in tetrahedral mesh
     * 
     * @param currentPositions Current vertex positions
     * @param restPositions Rest (original) vertex positions
     * @param tetrahedronIndices Tetrahedral indices (4 per tet)
     * @param tetrahedronCount Number of tetrahedra
     * @param tearThreshold Stress threshold for tearing (e.g., 1.5 = 150% stretch)
     * @param resistanceMap Optional resistance map for variable thresholds
     * @param outTears Output vector of detected tears
     */
    void DetectStress(
        const Vec3* currentPositions,
        const Vec3* restPositions,
        const int* tetrahedronIndices,
        int tetrahedronCount,
        float tearThreshold,
        const TearResistanceMap* resistanceMap,
        std::vector<TearInfo>& outTears
    );

    /**
     * @brief Calculate stress on a single edge
     * 
     * @param v0Current Current position of vertex 0
     * @param v1Current Current position of vertex 1
     * @param v0Rest Rest position of vertex 0
     * @param v1Rest Rest position of vertex 1
     * @return Stress ratio (currentLength / restLength)
     */
    static float CalculateEdgeStress(
        const Vec3& v0Current, const Vec3& v1Current,
        const Vec3& v0Rest, const Vec3& v1Rest
    );

    /**
     * @brief Calculate volume stress of a tetrahedron
     * 
     * @param v0, v1, v2, v3 Current vertex positions
     * @param v0Rest, v1Rest, v2Rest, v3Rest Rest vertex positions
     * @return Volume stress ratio (currentVolume / restVolume)
     */
    static float CalculateVolumeStress(
        const Vec3& v0, const Vec3& v1, const Vec3& v2, const Vec3& v3,
        const Vec3& v0Rest, const Vec3& v1Rest, const Vec3& v2Rest, const Vec3& v3Rest
    );

    /**
     * @brief Get stress data for all tetrahedra
     */
    const std::vector<StressData>& GetStressData() const { return m_StressData; }

    /**
     * @brief Clear all stress data
     */
    void ClearStressData();

    /**
     * @brief Set time for tear timestamps
     */
    void SetCurrentTime(float time) { m_CurrentTime = time; }
    
    /**
     * @brief Enable/disable healing
     */
    void SetHealingEnabled(bool enabled) { m_HealingEnabled = enabled; }
    
    /**
     * @brief Set healing rate (progress per second)
     */
    void SetHealingRate(float rate) { m_HealingRate = rate; }
    
    /**
     * @brief Set delay before healing starts
     */
    void SetHealingDelay(float delay) { m_HealingDelay = delay; }
    
    /**
     * @brief Update healing progress
     */
    void UpdateHealing(float deltaTime, TearResistanceMap& resistanceMap);
    
    /**
     * @brief Register tear for healing
     */
    void RegisterTearForHealing(int tetIndex, float originalResistance);
    
    /**
     * @brief Get healing tears
     */
    const std::vector<HealingTear>& GetHealingTears() const { return m_HealingTears; }

private:
    std::vector<StressData> m_StressData;
    float m_CurrentTime;

    // Edge indices for a tetrahedron (6 edges)
    static const int EDGE_INDICES[6][2];

    // Calculate tetrahedron volume
    static float CalculateTetrahedronVolume(
        const Vec3& v0, const Vec3& v1, const Vec3& v2, const Vec3& v3
    );
};
