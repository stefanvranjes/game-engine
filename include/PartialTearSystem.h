#pragma once

#include "Math/Vec3.h"
#include "SoftBodyTearSystem.h"
#include <vector>
#include <unordered_map>

/**
 * @brief Manages partial tears (cracks) that weaken material without full separation
 * 
 * Tracks cracks that form when stress exceeds crack threshold but is below
 * tear threshold. Cracks progressively damage the material and can eventually
 * convert to full tears.
 */
class PartialTearSystem {
public:
    /**
     * @brief Information about a crack in the soft body
     */
    struct Crack {
        int tetrahedronIndex;       // Which tetrahedron contains the crack
        int edgeIndex;              // Which edge is cracked (0-5)
        float damage;               // Damage level: 0.0 = no damage, 1.0 = fully cracked
        float stiffnessMultiplier;  // Current stiffness: 1.0 = full strength, 0.0 = no strength
        Vec3 crackPosition;         // World position of crack
        Vec3 crackNormal;           // Normal direction of crack
        float creationTime;         // When crack was created
        float lastStressTime;       // Last time stress was applied
    };

    PartialTearSystem();
    ~PartialTearSystem();

    /**
     * @brief Detect new cracks based on stress levels
     * 
     * @param currentPositions Current vertex positions
     * @param restPositions Rest vertex positions
     * @param tetrahedronIndices Tetrahedral mesh indices
     * @param tetrahedronCount Number of tetrahedra
     * @param stressData Stress data for all tetrahedra
     * @param crackThreshold Stress threshold to initiate crack
     * @param tearThreshold Stress threshold for full tear
     * @param currentTime Current simulation time
     */
    void DetectCracks(
        const Vec3* currentPositions,
        const Vec3* restPositions,
        const int* tetrahedronIndices,
        int tetrahedronCount,
        const SoftBodyTearSystem::StressData* stressData,
        float crackThreshold,
        float tearThreshold,
        float currentTime
    );

    /**
     * @brief Progress existing cracks based on continued stress
     * 
     * @param deltaTime Time since last update
     * @param stressData Current stress data
     * @param progressionRate Damage increase per second under stress
     * @param currentTime Current simulation time
     */
    void ProgressCracks(
        float deltaTime,
        const SoftBodyTearSystem::StressData* stressData,
        float progressionRate,
        float currentTime
    );

    /**
     * @brief Get tetrahedra with fully damaged cracks (ready to tear)
     * 
     * @return Indices of tetrahedra that should be converted to full tears
     */
    std::vector<int> GetFullyDamagedTets() const;

    /**
     * @brief Remove cracks from tetrahedra (after they've been torn)
     * 
     * @param tetrahedronIndices Indices of tetrahedra to remove cracks from
     */
    void RemoveCracks(const std::vector<int>& tetrahedronIndices);

    /**
     * @brief Get all active cracks
     */
    const std::vector<Crack>& GetCracks() const { return m_Cracks; }

    /**
     * @brief Get crack for specific tetrahedron
     * 
     * @param tetIndex Tetrahedron index
     * @return Pointer to crack, or nullptr if no crack exists
     */
    const Crack* GetCrack(int tetIndex) const;

    /**
     * @brief Clear all cracks
     */
    void ClearCracks();

    /**
     * @brief Get stiffness multiplier for a tetrahedron
     * 
     * @param tetIndex Tetrahedron index
     * @return Stiffness multiplier (1.0 = full strength, 0.0 = no strength)
     */
    float GetStiffnessMultiplier(int tetIndex) const;

    /**
     * @brief Enable/disable crack healing
     */
    void SetHealingEnabled(bool enabled) { m_HealingEnabled = enabled; }

    /**
     * @brief Set crack healing rate
     * 
     * @param rate Damage reduction per second when not under stress
     */
    void SetHealingRate(float rate) { m_HealingRate = rate; }

    /**
     * @brief Get number of active cracks
     */
    int GetCrackCount() const { return static_cast<int>(m_Cracks.size()); }

private:
    std::vector<Crack> m_Cracks;
    std::unordered_map<int, int> m_TetToCrackIndex;  // Maps tet index to crack index
    
    bool m_HealingEnabled;
    float m_HealingRate;
    float m_HealingDelay;  // Time before healing starts

    /**
     * @brief Calculate stiffness multiplier from damage
     */
    static float CalculateStiffnessMultiplier(float damage);

    /**
     * @brief Check if crack already exists for tetrahedron
     */
    bool HasCrack(int tetIndex) const;
};
