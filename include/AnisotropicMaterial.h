#pragma once

#include "Math/Vec3.h"
#include <vector>

/**
 * @brief Fiber data for anisotropic materials
 * 
 * Represents directional material properties (e.g., muscle fibers, wood grain, fabric weave)
 */
struct FiberData {
    Vec3 direction;              // Normalized fiber direction
    float longitudinalStiffness; // Stiffness along fiber (default: 1.0)
    float transverseStiffness;   // Stiffness perpendicular to fiber (default: 0.3)
    
    FiberData()
        : direction(1, 0, 0)
        , longitudinalStiffness(1.0f)
        , transverseStiffness(0.3f)
    {}
    
    FiberData(const Vec3& dir, float longStiff = 1.0f, float transStiff = 0.3f)
        : direction(dir)
        , longitudinalStiffness(longStiff)
        , transverseStiffness(transStiff)
    {
        // Ensure direction is normalized
        float len = direction.Length();
        if (len > 1e-6f) {
            direction = direction * (1.0f / len);
        } else {
            direction = Vec3(1, 0, 0);
        }
    }
};

/**
 * @brief Anisotropic material model for soft bodies
 * 
 * Provides directional stiffness based on fiber orientation.
 * Uses transversely isotropic model (one preferred direction per element).
 */
class AnisotropicMaterial {
public:
    AnisotropicMaterial();
    
    /**
     * @brief Initialize fiber data for all tetrahedra
     * @param numTetrahedra Number of tetrahedra in the mesh
     */
    void Initialize(int numTetrahedra);
    
    /**
     * @brief Set fiber direction for a specific tetrahedron
     * @param tetIndex Tetrahedron index
     * @param direction Fiber direction (will be normalized)
     * @param longitudinalStiffness Stiffness along fiber
     * @param transverseStiffness Stiffness perpendicular to fiber
     */
    void SetFiberDirection(
        int tetIndex,
        const Vec3& direction,
        float longitudinalStiffness = 1.0f,
        float transverseStiffness = 0.3f
    );
    
    /**
     * @brief Get stiffness multiplier based on strain direction
     * @param tetIndex Tetrahedron index
     * @param strainDir Direction of strain (normalized)
     * @return Stiffness multiplier (1.0 = isotropic baseline)
     */
    float GetStiffnessMultiplier(int tetIndex, const Vec3& strainDir) const;
    
    /**
     * @brief Get fiber data for a tetrahedron
     */
    const FiberData& GetFiberData(int tetIndex) const;
    
    /**
     * @brief Set all fibers to the same direction (e.g., for muscles)
     */
    void SetUniformFiberDirection(
        const Vec3& direction,
        float longitudinalStiffness = 1.0f,
        float transverseStiffness = 0.3f
    );
    
    /**
     * @brief Check if material is initialized
     */
    bool IsInitialized() const { return !m_FiberData.empty(); }
    
    /**
     * @brief Get number of tetrahedra
     */
    int GetTetrahedronCount() const { return static_cast<int>(m_FiberData.size()); }

private:
    std::vector<FiberData> m_FiberData;
};
