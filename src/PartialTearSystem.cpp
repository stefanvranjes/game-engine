#include "PartialTearSystem.h"
#include <algorithm>
#include <cmath>
#include <iostream>

PartialTearSystem::PartialTearSystem()
    : m_HealingEnabled(false)
    , m_HealingRate(0.1f)
    , m_HealingDelay(2.0f)
{
}

PartialTearSystem::~PartialTearSystem() {
}

void PartialTearSystem::DetectCracks(
    const Vec3* currentPositions,
    const Vec3* restPositions,
    const int* tetrahedronIndices,
    int tetrahedronCount,
    const SoftBodyTearSystem::StressData* stressData,
    float crackThreshold,
    float tearThreshold,
    float currentTime)
{
    if (!currentPositions || !restPositions || !tetrahedronIndices || !stressData) {
        return;
    }

    // Edge indices for a tetrahedron (6 edges)
    const int edgeIndices[6][2] = {
        {0, 1}, {0, 2}, {0, 3},
        {1, 2}, {1, 3}, {2, 3}
    };

    // Check each tetrahedron for potential cracks
    for (int tetIdx = 0; tetIdx < tetrahedronCount; ++tetIdx) {
        // Skip if already has a crack or is torn
        if (HasCrack(tetIdx) || stressData[tetIdx].isTorn) {
            continue;
        }

        const int* tet = &tetrahedronIndices[tetIdx * 4];
        
        // Check each edge for stress in crack range
        for (int edgeIdx = 0; edgeIdx < 6; ++edgeIdx) {
            float edgeStress = stressData[tetIdx].edgeStress[edgeIdx];
            
            // Crack if stress is between crack threshold and tear threshold
            if (edgeStress >= crackThreshold && edgeStress < tearThreshold) {
                // Create new crack
                Crack crack;
                crack.tetrahedronIndex = tetIdx;
                crack.edgeIndex = edgeIdx;
                crack.damage = 0.0f;  // Start with no damage
                crack.stiffnessMultiplier = 1.0f;
                crack.creationTime = currentTime;
                crack.lastStressTime = currentTime;
                
                // Calculate crack position (midpoint of edge)
                int v0 = tet[edgeIndices[edgeIdx][0]];
                int v1 = tet[edgeIndices[edgeIdx][1]];
                crack.crackPosition = (currentPositions[v0] + currentPositions[v1]) * 0.5f;
                
                // Calculate crack normal (perpendicular to edge)
                Vec3 edgeDir = (currentPositions[v1] - currentPositions[v0]).Normalized();
                Vec3 up(0, 1, 0);
                crack.crackNormal = edgeDir.Cross(up).Normalized();
                
                // Add crack
                int crackIndex = static_cast<int>(m_Cracks.size());
                m_Cracks.push_back(crack);
                m_TetToCrackIndex[tetIdx] = crackIndex;
                
                std::cout << "Crack initiated in tet " << tetIdx << " on edge " << edgeIdx 
                         << " (stress: " << edgeStress << ")" << std::endl;
                
                break;  // Only one crack per tet
            }
        }
    }
}

void PartialTearSystem::ProgressCracks(
    float deltaTime,
    const SoftBodyTearSystem::StressData* stressData,
    float progressionRate,
    float currentTime)
{
    if (!stressData || m_Cracks.empty()) {
        return;
    }

    for (auto& crack : m_Cracks) {
        if (crack.tetrahedronIndex < 0) {
            continue;  // Invalid crack
        }

        const auto& stress = stressData[crack.tetrahedronIndex];
        float edgeStress = stress.edgeStress[crack.edgeIndex];
        
        // Check if edge is still under stress
        bool underStress = edgeStress > 1.0f;  // Any stress above rest length
        
        if (underStress) {
            // Progress damage
            float damageIncrease = progressionRate * deltaTime;
            crack.damage = std::min(1.0f, crack.damage + damageIncrease);
            crack.lastStressTime = currentTime;
            
            // Update stiffness multiplier
            crack.stiffnessMultiplier = CalculateStiffnessMultiplier(crack.damage);
            
            if (crack.damage >= 1.0f) {
                std::cout << "Crack in tet " << crack.tetrahedronIndex 
                         << " fully damaged (ready to tear)" << std::endl;
            }
        } else if (m_HealingEnabled) {
            // Check if healing delay has passed
            float timeSinceStress = currentTime - crack.lastStressTime;
            if (timeSinceStress >= m_HealingDelay) {
                // Heal crack
                float healingAmount = m_HealingRate * deltaTime;
                crack.damage = std::max(0.0f, crack.damage - healingAmount);
                crack.stiffnessMultiplier = CalculateStiffnessMultiplier(crack.damage);
                
                if (crack.damage <= 0.0f) {
                    std::cout << "Crack in tet " << crack.tetrahedronIndex << " healed" << std::endl;
                }
            }
        }
    }
    
    // Remove fully healed cracks
    if (m_HealingEnabled) {
        auto it = m_Cracks.begin();
        while (it != m_Cracks.end()) {
            if (it->damage <= 0.0f) {
                m_TetToCrackIndex.erase(it->tetrahedronIndex);
                it = m_Cracks.erase(it);
            } else {
                ++it;
            }
        }
    }
}

std::vector<int> PartialTearSystem::GetFullyDamagedTets() const {
    std::vector<int> fullyDamagedTets;
    
    for (const auto& crack : m_Cracks) {
        if (crack.damage >= 1.0f) {
            fullyDamagedTets.push_back(crack.tetrahedronIndex);
        }
    }
    
    return fullyDamagedTets;
}

void PartialTearSystem::RemoveCracks(const std::vector<int>& tetrahedronIndices) {
    for (int tetIdx : tetrahedronIndices) {
        auto it = m_TetToCrackIndex.find(tetIdx);
        if (it != m_TetToCrackIndex.end()) {
            int crackIndex = it->second;
            
            // Mark crack as invalid (will be cleaned up later)
            if (crackIndex >= 0 && crackIndex < static_cast<int>(m_Cracks.size())) {
                m_Cracks[crackIndex].tetrahedronIndex = -1;
            }
            
            m_TetToCrackIndex.erase(it);
        }
    }
    
    // Remove invalid cracks
    auto it = m_Cracks.begin();
    while (it != m_Cracks.end()) {
        if (it->tetrahedronIndex < 0) {
            it = m_Cracks.erase(it);
        } else {
            ++it;
        }
    }
    
    // Rebuild index map
    m_TetToCrackIndex.clear();
    for (size_t i = 0; i < m_Cracks.size(); ++i) {
        m_TetToCrackIndex[m_Cracks[i].tetrahedronIndex] = static_cast<int>(i);
    }
}

const PartialTearSystem::Crack* PartialTearSystem::GetCrack(int tetIndex) const {
    auto it = m_TetToCrackIndex.find(tetIndex);
    if (it != m_TetToCrackIndex.end()) {
        int crackIndex = it->second;
        if (crackIndex >= 0 && crackIndex < static_cast<int>(m_Cracks.size())) {
            return &m_Cracks[crackIndex];
        }
    }
    return nullptr;
}

void PartialTearSystem::ClearCracks() {
    m_Cracks.clear();
    m_TetToCrackIndex.clear();
}

float PartialTearSystem::GetStiffnessMultiplier(int tetIndex) const {
    const Crack* crack = GetCrack(tetIndex);
    if (crack) {
        return crack->stiffnessMultiplier;
    }
    return 1.0f;  // Full strength if no crack
}

float PartialTearSystem::CalculateStiffnessMultiplier(float damage) {
    // Stiffness reduces as damage increases
    // At 0% damage: 100% stiffness
    // At 100% damage: 20% stiffness (not completely zero to maintain stability)
    return 1.0f - (damage * 0.8f);
}

bool PartialTearSystem::HasCrack(int tetIndex) const {
    return m_TetToCrackIndex.find(tetIndex) != m_TetToCrackIndex.end();
}
