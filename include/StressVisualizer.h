#pragma once

#include "Math/Vec3.h"
#include <vector>

class PhysXSoftBody;

/**
 * @brief Calculates and visualizes stress distribution on soft bodies
 * 
 * Provides real-time stress calculation and heatmap visualization
 * to show areas of high deformation and potential tearing.
 */
class StressVisualizer {
public:
    StressVisualizer();
    ~StressVisualizer();
    
    /**
     * @brief Calculate stress values for all vertices
     * @param softBody Soft body to analyze
     */
    void CalculateStress(PhysXSoftBody* softBody);
    
    /**
     * @brief Get stress value for a specific vertex
     * @param vertexIndex Vertex index
     * @return Stress value (0.0 = no stress, higher = more stress)
     */
    float GetVertexStress(int vertexIndex) const;
    
    /**
     * @brief Get all stress values
     */
    const std::vector<float>& GetStressValues() const { return m_StressValues; }
    
    /**
     * @brief Get min/max stress for normalization
     */
    float GetMinStress() const { return m_MinStress; }
    float GetMaxStress() const { return m_MaxStress; }
    
    /**
     * @brief Set stress calculation mode
     */
    enum class StressMode {
        Displacement,  // Based on vertex displacement from rest
        VolumeChange,  // Based on tetrahedron volume change
        Combined       // Combination of both
    };
    
    void SetStressMode(StressMode mode) { m_StressMode = mode; }
    StressMode GetStressMode() const { return m_StressMode; }
    
    /**
     * @brief Enable/disable stress visualization
     */
    void SetEnabled(bool enabled) { m_Enabled = enabled; }
    bool IsEnabled() const { return m_Enabled; }
    
private:
    std::vector<float> m_StressValues;
    float m_MinStress;
    float m_MaxStress;
    StressMode m_StressMode;
    bool m_Enabled;
    
    // Calculation methods
    float CalculateDisplacementStress(PhysXSoftBody* softBody, int vertexIndex);
    float CalculateVolumeStress(PhysXSoftBody* softBody, int vertexIndex);
    float GetAverageEdgeLength(PhysXSoftBody* softBody, int vertexIndex);
};
