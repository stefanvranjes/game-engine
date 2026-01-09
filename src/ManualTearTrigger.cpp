#include "ManualTearTrigger.h"
#include "PhysXSoftBody.h"
#include "TearResistanceMap.h"
#include <algorithm>
#include <iostream>

ManualTearTrigger::ManualTearTrigger()
    : m_Type(TriggerType::Point)
    , m_Intensity(1.0f)
{
}

void ManualTearTrigger::Apply(PhysXSoftBody* softBody) {
    if (!softBody || m_AffectedVertices.empty()) {
        return;
    }
    
    switch (m_Type) {
        case TriggerType::Point:
            ApplyPointTear(softBody);
            break;
        case TriggerType::Path:
            ApplyPathTear(softBody);
            break;
        case TriggerType::Region:
            ApplyRegionTear(softBody);
            break;
    }
    
    std::cout << "Applied manual tear (type=" << static_cast<int>(m_Type) 
              << ", vertices=" << m_AffectedVertices.size() 
              << ", intensity=" << m_Intensity << ")" << std::endl;
}

std::vector<int> ManualTearTrigger::GetAffectedTetrahedra(PhysXSoftBody* softBody) const {
    std::vector<int> affectedTets;
    
    // TODO: Implement when tetrahedral connectivity is available
    // For now, return empty vector
    
    return affectedTets;
}

void ManualTearTrigger::ApplyPointTear(PhysXSoftBody* softBody) {
    if (m_AffectedVertices.empty()) return;
    
    // Get the primary vertex
    int vertexIndex = m_AffectedVertices[0];
    
    // Reduce resistance around this vertex
    // TODO: Implement when resistance map has per-vertex or per-tetrahedron setters
    
    // For now, reduce global tear threshold to trigger tear
    float currentThreshold = softBody->GetTearThreshold();
    float newThreshold = currentThreshold * (1.0f - m_Intensity);
    softBody->SetTearThreshold(newThreshold);
    
    std::cout << "Point tear at vertex " << vertexIndex << std::endl;
}

void ManualTearTrigger::ApplyPathTear(PhysXSoftBody* softBody) {
    if (m_AffectedVertices.size() < 2) {
        std::cerr << "Path tear requires at least 2 vertices" << std::endl;
        return;
    }
    
    // Find tetrahedra along the path
    // TODO: Implement path-finding between vertices
    
    // Reduce resistance along path
    float currentThreshold = softBody->GetTearThreshold();
    float newThreshold = currentThreshold * (1.0f - m_Intensity);
    softBody->SetTearThreshold(newThreshold);
    
    std::cout << "Path tear along " << m_AffectedVertices.size() << " vertices" << std::endl;
}

void ManualTearTrigger::ApplyRegionTear(PhysXSoftBody* softBody) {
    if (m_AffectedVertices.empty()) return;
    
    // Reduce resistance for all vertices in region
    float currentThreshold = softBody->GetTearThreshold();
    float newThreshold = currentThreshold * (1.0f - m_Intensity);
    softBody->SetTearThreshold(newThreshold);
    
    std::cout << "Region tear affecting " << m_AffectedVertices.size() << " vertices" << std::endl;
}
