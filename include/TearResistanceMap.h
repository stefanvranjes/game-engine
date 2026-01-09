#pragma once

#include "Math/Vec3.h"
#include <vector>

/**
 * @brief Maps tear resistance values to tetrahedra
 */
class TearResistanceMap {
public:
    TearResistanceMap();
    ~TearResistanceMap();

    /**
     * @brief Initialize with tetrahedron count
     * @param tetrahedronCount Number of tetrahedra
     * @param defaultResistance Default resistance multiplier (1.0 = normal)
     */
    void Initialize(int tetrahedronCount, float defaultResistance = 1.0f);

    /**
     * @brief Set resistance for specific tetrahedron
     * @param tetIndex Tetrahedron index
     * @param resistance Resistance multiplier (>1.0 = stronger, <1.0 = weaker)
     */
    void SetTetrahedronResistance(int tetIndex, float resistance);

    /**
     * @brief Set resistance for all tetrahedra in sphere
     * @param tetCenters Tetrahedron center positions
     * @param tetCount Number of tetrahedra
     * @param center Sphere center
     * @param radius Sphere radius
     * @param resistance Resistance multiplier
     */
    void SetSphereResistance(
        const Vec3* tetCenters,
        int tetCount,
        const Vec3& center,
        float radius,
        float resistance
    );

    /**
     * @brief Set resistance gradient between two points
     * @param tetCenters Tetrahedron center positions
     * @param tetCount Number of tetrahedra
     * @param start Gradient start point
     * @param end Gradient end point
     * @param startResistance Resistance at start
     * @param endResistance Resistance at end
     */
    void SetGradient(
        const Vec3* tetCenters,
        int tetCount,
        const Vec3& start,
        const Vec3& end,
        float startResistance,
        float endResistance
    );

    /**
     * @brief Get resistance multiplier for tetrahedron
     * @param tetIndex Tetrahedron index
     * @return Resistance multiplier
     */
    float GetResistance(int tetIndex) const;

    /**
     * @brief Reset all resistances to default
     */
    void Reset();

    /**
     * @brief Check if initialized
     */
    bool IsInitialized() const { return !m_Resistances.empty(); }

private:
    std::vector<float> m_Resistances;
    float m_DefaultResistance;
};
