#include "TearPreview.h"
#include "PhysXSoftBody.h"
#include <algorithm>
#include <iostream>

TearPreview::TearPreview()
    : m_Enabled(false)
{
}

void TearPreview::SetTearPath(const std::vector<int>& vertexPath) {
    m_TearPath = vertexPath;
    m_Info = PreviewInfo(); // Reset info
}

void TearPreview::Calculate(PhysXSoftBody* softBody) {
    if (!softBody || m_TearPath.empty()) {
        m_Info.isValid = false;
        return;
    }
    
    CalculateAffectedElements(softBody);
    EstimateStress(softBody);
    EstimateNewVertices(softBody);
    
    m_Info.isValid = true;
    
    std::cout << "Tear preview calculated:" << std::endl;
    std::cout << "  Affected vertices: " << m_Info.affectedVertices.size() << std::endl;
    std::cout << "  Affected tetrahedra: " << m_Info.affectedTetrahedra.size() << std::endl;
    std::cout << "  Estimated new vertices: " << m_Info.estimatedNewVertices << std::endl;
    std::cout << "  Estimated stress: " << m_Info.estimatedStress << std::endl;
}

void TearPreview::Render(PhysXSoftBody* softBody) {
    if (!m_Enabled || !m_Info.isValid || !softBody) {
        return;
    }
    
    // TODO: Implement visual rendering
    // - Draw tear path as yellow line
    // - Highlight affected tetrahedra in red
    // - Show affected vertices
}

void TearPreview::CalculateAffectedElements(PhysXSoftBody* softBody) {
    m_Info.affectedVertices = m_TearPath;
    
    // TODO: Find affected tetrahedra when connectivity is available
    // For now, estimate based on vertex count
    m_Info.affectedTetrahedra.clear();
}

void TearPreview::EstimateStress(PhysXSoftBody* softBody) {
    // Estimate stress based on tear path length and current deformation
    // TODO: Implement proper stress calculation
    
    // Simple estimate: assume stress proportional to path length
    m_Info.estimatedStress = static_cast<float>(m_TearPath.size()) * 0.5f;
}

void TearPreview::EstimateNewVertices(PhysXSoftBody* softBody) {
    // Estimate number of new vertices created by tear
    // Each vertex along tear path may be duplicated
    
    // Conservative estimate: one new vertex per path vertex
    m_Info.estimatedNewVertices = static_cast<int>(m_TearPath.size());
}
