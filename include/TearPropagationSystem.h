#pragma once

#include "Math/Vec3.h"
#include "SoftBodyTearSystem.h"
#include "ProceduralTearGenerator.h"
#include <vector>
#include <memory>

/**
 * @brief Manages progressive tear propagation over time
 * 
 * Instead of instant tearing, this system propagates tears gradually
 * based on a configurable propagation speed, creating more realistic
 * fracture behavior.
 */
class TearPropagationSystem {
public:
    /**
     * @brief Information about an actively propagating tear
     */
    struct PropagatingTear {
        int currentTetrahedron;           // Current tet being torn
        Vec3 currentPosition;             // Current tear position
        Vec3 direction;                   // Propagation direction
        float energy;                     // Remaining energy
        float propagationProgress;        // Progress through current tet (0-1)
        std::vector<int> path;            // Path of tets already torn
        ProceduralTearGenerator::TearPath fullPath;  // Complete generated path
        int pathIndex;                    // Current index in fullPath
    };

    TearPropagationSystem();
    ~TearPropagationSystem();

    /**
     * @brief Update all propagating tears
     * 
     * @param deltaTime Time since last update in seconds
     */
    void Update(float deltaTime);

    /**
     * @brief Start a new propagating tear
     * 
     * @param initialTear Initial tear information
     * @param generatedPath Pre-generated tear path
     */
    void StartPropagation(
        const SoftBodyTearSystem::TearInfo& initialTear,
        const ProceduralTearGenerator::TearPath& generatedPath
    );

    /**
     * @brief Set propagation speed
     * 
     * @param tetsPerSecond Number of tetrahedra to tear per second
     */
    void SetPropagationSpeed(float tetsPerSecond);

    /**
     * @brief Get propagation speed
     */
    float GetPropagationSpeed() const { return m_PropagationSpeed; }

    /**
     * @brief Get all active tears
     */
    const std::vector<PropagatingTear>& GetActiveTears() const { return m_ActiveTears; }

    /**
     * @brief Check if any tears are currently propagating
     */
    bool HasActiveTears() const { return !m_ActiveTears.empty(); }

    /**
     * @brief Clear all active tears
     */
    void ClearActiveTears();

    /**
     * @brief Set callback for when a tetrahedron is torn
     * 
     * @param callback Function called with (tetIndex, position, normal)
     */
    void SetTearCallback(std::function<void(int, const Vec3&, const Vec3&)> callback);

private:
    std::vector<PropagatingTear> m_ActiveTears;
    float m_PropagationSpeed;  // Tets per second
    std::function<void(int, const Vec3&, const Vec3&)> m_TearCallback;
};
