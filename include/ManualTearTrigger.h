#pragma once

#include "Math/Vec3.h"
#include <vector>

class PhysXSoftBody;

/**
 * @brief Manual tear trigger for user-controlled tearing
 * 
 * Allows users to manually trigger tears at specific locations
 * with configurable intensity and affected regions.
 */
class ManualTearTrigger {
public:
    enum class TriggerType {
        Point,      // Single point tear
        Path,       // Path along vertices
        Region      // Circular region
    };
    
    ManualTearTrigger();
    
    /**
     * @brief Set trigger type
     */
    void SetType(TriggerType type) { m_Type = type; }
    TriggerType GetType() const { return m_Type; }
    
    /**
     * @brief Set affected vertices
     */
    void SetAffectedVertices(const std::vector<int>& vertices) { m_AffectedVertices = vertices; }
    const std::vector<int>& GetAffectedVertices() const { return m_AffectedVertices; }
    
    /**
     * @brief Set tear intensity (0.0 = no tear, 1.0 = complete tear)
     */
    void SetIntensity(float intensity) { m_Intensity = intensity; }
    float GetIntensity() const { return m_Intensity; }
    
    /**
     * @brief Apply tear to soft body
     * @param softBody Target soft body
     */
    void Apply(PhysXSoftBody* softBody);
    
    /**
     * @brief Get affected tetrahedra for preview
     */
    std::vector<int> GetAffectedTetrahedra(PhysXSoftBody* softBody) const;
    
private:
    TriggerType m_Type;
    std::vector<int> m_AffectedVertices;
    float m_Intensity;
    
    void ApplyPointTear(PhysXSoftBody* softBody);
    void ApplyPathTear(PhysXSoftBody* softBody);
    void ApplyRegionTear(PhysXSoftBody* softBody);
};
