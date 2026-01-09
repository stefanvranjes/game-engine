#include "SoftBodyTearSystem.h"
#include "TearResistanceMap.h"
#include <cmath>
#include <algorithm>
#include <iostream>

// Edge indices for a tetrahedron
// Each tetrahedron has 4 vertices (0,1,2,3) and 6 edges
const int SoftBodyTearSystem::EDGE_INDICES[6][2] = {
    {0, 1}, {0, 2}, {0, 3},  // Edges from vertex 0
    {1, 2}, {1, 3},          // Edges from vertex 1
    {2, 3}                   // Edge from vertex 2
};

SoftBodyTearSystem::SoftBodyTearSystem()
    : m_CurrentTime(0.0f)
    , m_HealingEnabled(false)
    , m_HealingRate(0.2f)
    , m_HealingDelay(2.0f)
    , m_PlasticityEnabled(false)
    , m_PlasticThreshold(1.3f)
    , m_PlasticityRate(0.05f)
{
}

SoftBodyTearSystem::~SoftBodyTearSystem() {
}

void SoftBodyTearSystem::DetectStress(
    const Vec3* currentPositions,
    const Vec3* restPositions,
    const int* tetrahedronIndices,
    int tetrahedronCount,
    float tearThreshold,
    const TearResistanceMap* resistanceMap,
    std::vector<TearInfo>& outTears)
{
    // Resize stress data if needed
    if (m_StressData.size() != static_cast<size_t>(tetrahedronCount)) {
        m_StressData.resize(tetrahedronCount);
    }

    outTears.clear();

    // Check each tetrahedron
    for (int tetIdx = 0; tetIdx < tetrahedronCount; ++tetIdx) {
        StressData& stressData = m_StressData[tetIdx];
        
        // Skip if already torn
        if (stressData.isTorn) {
            continue;
        }

        // Get tetrahedron vertices
        int v0 = tetrahedronIndices[tetIdx * 4 + 0];
        int v1 = tetrahedronIndices[tetIdx * 4 + 1];
        int v2 = tetrahedronIndices[tetIdx * 4 + 2];
        int v3 = tetrahedronIndices[tetIdx * 4 + 3];

        // Calculate volume stress
        stressData.volumeStress = CalculateVolumeStress(
            currentPositions[v0], currentPositions[v1], currentPositions[v2], currentPositions[v3],
            restPositions[v0], restPositions[v1], restPositions[v2], restPositions[v3]
        );

        // Calculate stress on each edge
        int vertexIndices[4] = {v0, v1, v2, v3};
        float maxStress = 0.0f;
        int maxStressEdge = -1;

        for (int edgeIdx = 0; edgeIdx < 6; ++edgeIdx) {
            int i0 = vertexIndices[EDGE_INDICES[edgeIdx][0]];
            int i1 = vertexIndices[EDGE_INDICES[edgeIdx][1]];

            float stress = CalculateEdgeStress(
                currentPositions[i0], currentPositions[i1],
                restPositions[i0], restPositions[i1]
            );

            stressData.edgeStress[edgeIdx] = stress;

            if (stress > maxStress) {
                maxStress = stress;
                maxStressEdge = edgeIdx;
            }
        }

        // Check if any edge exceeds tear threshold
        // Apply resistance multiplier if available
        float resistance = resistanceMap ? resistanceMap->GetResistance(tetIdx) : 1.0f;
        float effectiveThreshold = tearThreshold * resistance;
        
        if (maxStress > effectiveThreshold && maxStressEdge >= 0) {
            // Create tear info
            TearInfo tear;
            tear.tetrahedronIndex = tetIdx;
            tear.edgeVertices[0] = vertexIndices[EDGE_INDICES[maxStressEdge][0]];
            tear.edgeVertices[1] = vertexIndices[EDGE_INDICES[maxStressEdge][1]];
            tear.stress = maxStress;
            tear.timestamp = m_CurrentTime;

            // Calculate tear position (midpoint of edge)
            tear.tearPosition = (currentPositions[tear.edgeVertices[0]] + 
                                currentPositions[tear.edgeVertices[1]]) * 0.5f;

            // Calculate tear normal (perpendicular to edge)
            Vec3 edgeDir = currentPositions[tear.edgeVertices[1]] - 
                          currentPositions[tear.edgeVertices[0]];
            edgeDir = edgeDir.Normalized();
            
            // Use cross product with up vector to get perpendicular
            Vec3 up(0, 1, 0);
            tear.tearNormal = edgeDir.Cross(up).Normalized();

            outTears.push_back(tear);

            // Mark as torn
            stressData.isTorn = true;
            stressData.tornEdgeIndex = maxStressEdge;

            std::cout << "Tear detected in tetrahedron " << tetIdx 
                     << " on edge (" << tear.edgeVertices[0] << ", " << tear.edgeVertices[1] << ")"
                     << " with stress " << maxStress << std::endl;
        }
    }
}

float SoftBodyTearSystem::CalculateEdgeStress(
    const Vec3& v0Current, const Vec3& v1Current,
    const Vec3& v0Rest, const Vec3& v1Rest)
{
    // Calculate current and rest lengths
    float currentLength = (v1Current - v0Current).Length();
    float restLength = (v1Rest - v0Rest).Length();

    // Avoid division by zero
    if (restLength < 0.0001f) {
        return 1.0f;
    }

    // Stress ratio
    return currentLength / restLength;
}

float SoftBodyTearSystem::CalculateVolumeStress(
    const Vec3& v0, const Vec3& v1, const Vec3& v2, const Vec3& v3,
    const Vec3& v0Rest, const Vec3& v1Rest, const Vec3& v2Rest, const Vec3& v3Rest)
{
    float currentVolume = CalculateTetrahedronVolume(v0, v1, v2, v3);
    float restVolume = CalculateTetrahedronVolume(v0Rest, v1Rest, v2Rest, v3Rest);

    // Avoid division by zero
    if (std::abs(restVolume) < 0.0001f) {
        return 1.0f;
    }

    return currentVolume / restVolume;
}

float SoftBodyTearSystem::CalculateTetrahedronVolume(
    const Vec3& v0, const Vec3& v1, const Vec3& v2, const Vec3& v3)
{
    // Volume = |det(v1-v0, v2-v0, v3-v0)| / 6
    Vec3 a = v1 - v0;
    Vec3 b = v2 - v0;
    Vec3 c = v3 - v0;

    // Scalar triple product: a · (b × c)
    float det = a.Dot(b.Cross(c));

    return std::abs(det) / 6.0f;
}

void SoftBodyTearSystem::ClearStressData() {
    for (auto& data : m_StressData) {
        data.isTorn = false;
        data.tornEdgeIndex = -1;
        data.volumeStress = 1.0f;
        for (int i = 0; i < 6; ++i) {
            data.edgeStress[i] = 1.0f;
        }
    }
}

void SoftBodyTearSystem::UpdateHealing(float deltaTime, TearResistanceMap& resistanceMap) {
    if (!m_HealingEnabled || m_HealingTears.empty()) {
        return;
    }
    
    for (auto it = m_HealingTears.begin(); it != m_HealingTears.end();) {
        it->timeSinceTear += deltaTime;
        
        // Check if healing delay has passed
        if (it->timeSinceTear < m_HealingDelay) {
            ++it;
            continue;
        }
        
        // Increase healing progress
        float healingAmount = m_HealingRate * deltaTime;
        it->healingProgress += healingAmount;
        
        if (it->healingProgress >= 1.0f) {
            // Fully healed
            it->healingProgress = 1.0f;
            resistanceMap.SetTetrahedronResistance(
                it->tetrahedronIndex,
                it->originalResistance
            );
            
            // Mark as no longer torn
            if (it->tetrahedronIndex < static_cast<int>(m_StressData.size())) {
                m_StressData[it->tetrahedronIndex].isTorn = false;
            }
            
            std::cout << "Tetrahedron " << it->tetrahedronIndex << " fully healed" << std::endl;
            it = m_HealingTears.erase(it);
        } else {
            // Partial healing - interpolate resistance
            float currentResistance = it->healingProgress * it->originalResistance;
            resistanceMap.SetTetrahedronResistance(
                it->tetrahedronIndex,
                currentResistance
            );
            ++it;
        }
    }
}

void SoftBodyTearSystem::RegisterTearForHealing(int tetIndex, float originalResistance) {
    if (!m_HealingEnabled) {
        return;
    }
    
    // Check if already healing
    for (const auto& tear : m_HealingTears) {
        if (tear.tetrahedronIndex == tetIndex) {
            return;  // Already registered
        }
    }
    
    HealingTear healingTear;
    healingTear.tetrahedronIndex = tetIndex;
    healingTear.healingProgress = 0.0f;
    healingTear.timeSinceTear = 0.0f;
    healingTear.originalResistance = originalResistance;
    
    m_HealingTears.push_back(healingTear);
    
    std::cout << "Registered tetrahedron " << tetIndex << " for healing" << std::endl;
}

void SoftBodyTearSystem::UpdatePlasticity(
    const Vec3* currentPositions,
    Vec3* restPositions,
    const int* tetrahedronIndices,
    int tetrahedronCount,
    float tearThreshold)
{
    if (!m_PlasticityEnabled || !currentPositions || !restPositions || !tetrahedronIndices) {
        return;
    }
    
    // Edge indices for tetrahedron
    const int EDGE_INDICES[6][2] = {
        {0, 1}, {0, 2}, {0, 3},
        {1, 2}, {1, 3}, {2, 3}
    };
    
    // Track which vertices have been modified
    std::vector<Vec3> positionDeltas(tetrahedronCount * 4, Vec3(0, 0, 0));
    std::vector<int> deltaCount(tetrahedronCount * 4, 0);
    
    // Process each tetrahedron
    for (int tetIdx = 0; tetIdx < tetrahedronCount; ++tetIdx) {
        const int* tet = &tetrahedronIndices[tetIdx * 4];
        
        // Check each edge
        for (int edgeIdx = 0; edgeIdx < 6; ++edgeIdx) {
            int vi = tet[EDGE_INDICES[edgeIdx][0]];
            int vj = tet[EDGE_INDICES[edgeIdx][1]];
            
            // Calculate current and rest edge lengths
            Vec3 currentEdge = currentPositions[vj] - currentPositions[vi];
            Vec3 restEdge = restPositions[vj] - restPositions[vi];
            
            float currentLength = currentEdge.Length();
            float restLength = restEdge.Length();
            
            if (restLength < 0.0001f) continue;
            
            float stress = currentLength / restLength;
            
            // Check if in plastic range
            if (stress > m_PlasticThreshold && stress < tearThreshold) {
                // Calculate plastic deformation
                // Move rest edge toward current edge
                Vec3 targetEdge = currentEdge;
                Vec3 edgeDelta = (targetEdge - restEdge) * m_PlasticityRate;
                
                // Distribute change to both vertices
                positionDeltas[vi] -= edgeDelta * 0.5f;
                positionDeltas[vj] += edgeDelta * 0.5f;
                deltaCount[vi]++;
                deltaCount[vj]++;
            }
        }
    }
    
    // Apply averaged deltas to rest positions
    int modifiedCount = 0;
    for (int i = 0; i < tetrahedronCount * 4; ++i) {
        if (deltaCount[i] > 0) {
            Vec3 avgDelta = positionDeltas[i] * (1.0f / deltaCount[i]);
            restPositions[i] = restPositions[i] + avgDelta;
            modifiedCount++;
        }
    }
    
    if (modifiedCount > 0) {
        std::cout << "Plastic deformation: " << modifiedCount << " vertices modified" << std::endl;
    }
}
