#include "StressVisualizer.h"
#include "PhysXSoftBody.h"
#include <algorithm>
#include <cmath>
#include <limits>

StressVisualizer::StressVisualizer()
    : m_MinStress(0.0f)
    , m_MaxStress(1.0f)
    , m_StressMode(StressMode::Displacement)
    , m_Enabled(false)
{
}

StressVisualizer::~StressVisualizer() {
}

void StressVisualizer::CalculateStress(PhysXSoftBody* softBody) {
    if (!softBody || !m_Enabled) {
        return;
    }
    
    int vertexCount = softBody->GetVertexCount();
    m_StressValues.resize(vertexCount);
    
    m_MinStress = std::numeric_limits<float>::max();
    m_MaxStress = std::numeric_limits<float>::lowest();
    
    // Calculate stress for each vertex
    for (int i = 0; i < vertexCount; ++i) {
        float stress = 0.0f;
        
        switch (m_StressMode) {
            case StressMode::Displacement:
                stress = CalculateDisplacementStress(softBody, i);
                break;
                
            case StressMode::VolumeChange:
                stress = CalculateVolumeStress(softBody, i);
                break;
                
            case StressMode::Combined:
                stress = (CalculateDisplacementStress(softBody, i) + 
                         CalculateVolumeStress(softBody, i)) * 0.5f;
                break;
        }
        
        m_StressValues[i] = stress;
        m_MinStress = std::min(m_MinStress, stress);
        m_MaxStress = std::max(m_MaxStress, stress);
    }
    
    // Ensure we have a valid range
    if (m_MaxStress <= m_MinStress) {
        m_MaxStress = m_MinStress + 1.0f;
    }
}

float StressVisualizer::GetVertexStress(int vertexIndex) const {
    if (vertexIndex < 0 || vertexIndex >= static_cast<int>(m_StressValues.size())) {
        return 0.0f;
    }
    return m_StressValues[vertexIndex];
}

float StressVisualizer::CalculateDisplacementStress(PhysXSoftBody* softBody, int vertexIndex) {
    // Get current position
    std::vector<Vec3> currentPositions(softBody->GetVertexCount());
    softBody->GetVertexPositions(currentPositions.data());
    
    Vec3 current = currentPositions[vertexIndex];
    
    // Get rest position (initial position)
    // TODO: Need getter for rest positions
    // For now, use a simplified approach
    Vec3 rest = current; // Placeholder
    
    // Calculate displacement
    Vec3 displacement = current - rest;
    float displacementMagnitude = displacement.Length();
    
    // Normalize by average edge length
    float avgEdgeLength = GetAverageEdgeLength(softBody, vertexIndex);
    if (avgEdgeLength > 0.0f) {
        return displacementMagnitude / avgEdgeLength;
    }
    
    return displacementMagnitude;
}

float StressVisualizer::CalculateVolumeStress(PhysXSoftBody* softBody, int vertexIndex) {
    // This would require tetrahedron connectivity information
    // TODO: Implement when tetrahedral mesh data is accessible
    
    // For now, return displacement-based stress
    return CalculateDisplacementStress(softBody, vertexIndex);
}

float StressVisualizer::GetAverageEdgeLength(PhysXSoftBody* softBody, int vertexIndex) {
    // This would require edge connectivity information
    // TODO: Implement when mesh topology is accessible
    
    // Return a default value for now
    return 1.0f;
}
