#pragma once

#include "Math/Vec3.h"
#include <vector>

class PhysXSoftBody;

/**
 * @brief Provides preview of tear effects before applying
 * 
 * Calculates and visualizes the impact of a tear operation
 * including affected elements and estimated results.
 */
class TearPreview {
public:
    struct PreviewInfo {
        std::vector<int> affectedTetrahedra;
        std::vector<int> affectedVertices;
        float estimatedStress;
        int estimatedNewVertices;
        bool isValid;
        
        PreviewInfo() 
            : estimatedStress(0.0f)
            , estimatedNewVertices(0)
            , isValid(false)
        {}
    };
    
    TearPreview();
    
    /**
     * @brief Set tear path for preview
     */
    void SetTearPath(const std::vector<int>& vertexPath);
    
    /**
     * @brief Calculate preview information
     */
    void Calculate(PhysXSoftBody* softBody);
    
    /**
     * @brief Get preview information
     */
    const PreviewInfo& GetPreviewInfo() const { return m_Info; }
    
    /**
     * @brief Enable/disable preview
     */
    void SetEnabled(bool enabled) { m_Enabled = enabled; }
    bool IsEnabled() const { return m_Enabled; }
    
    /**
     * @brief Render preview visualization
     * @param softBody Soft body to render preview on
     */
    void Render(PhysXSoftBody* softBody);
    
private:
    std::vector<int> m_TearPath;
    PreviewInfo m_Info;
    bool m_Enabled;
    
    void CalculateAffectedElements(PhysXSoftBody* softBody);
    void EstimateStress(PhysXSoftBody* softBody);
    void EstimateNewVertices(PhysXSoftBody* softBody);
};
