#include "ClothLOD.h"
#include "ClothMeshSimplifier.h"
#include <algorithm>
#include <cmath>
#include <iostream>

ClothLODConfig::ClothLODConfig()
    : m_Hysteresis(2.0f)  // 2 meter hysteresis by default
{
}

void ClothLODConfig::AddLODLevel(const ClothLODLevel& level) {
    m_Levels.push_back(level);
    
    // Sort by distance (should already be sorted, but ensure it)
    std::sort(m_Levels.begin(), m_Levels.end(),
        [](const ClothLODLevel& a, const ClothLODLevel& b) {
            return a.minDistance < b.minDistance;
        });
}

int ClothLODConfig::GetLODForDistance(float distance, int currentLOD) const {
    if (m_Levels.empty()) {
        return 0;
    }
    
    // Find appropriate LOD level
    int selectedLOD = 0;
    for (int i = static_cast<int>(m_Levels.size()) - 1; i >= 0; --i) {
        if (distance >= m_Levels[i].minDistance) {
            selectedLOD = i;
            break;
        }
    }
    
    // Apply hysteresis if we have a current LOD
    if (currentLOD >= 0 && currentLOD < static_cast<int>(m_Levels.size())) {
        // If transitioning to higher LOD (lower quality), add hysteresis
        if (selectedLOD > currentLOD) {
            float threshold = m_Levels[selectedLOD].minDistance + m_Hysteresis;
            if (distance < threshold) {
                selectedLOD = currentLOD;  // Stay at current LOD
            }
        }
        // If transitioning to lower LOD (higher quality), subtract hysteresis
        else if (selectedLOD < currentLOD) {
            float threshold = m_Levels[currentLOD].minDistance - m_Hysteresis;
            if (distance > threshold) {
                selectedLOD = currentLOD;  // Stay at current LOD
            }
        }
    }
    
    return selectedLOD;
}

const ClothLODLevel* ClothLODConfig::GetLODLevel(int index) const {
    if (index < 0 || index >= static_cast<int>(m_Levels.size())) {
        return nullptr;
    }
    return &m_Levels[index];
}

void ClothLODConfig::GenerateLODMeshes(
    const std::vector<Vec3>& basePositions,
    const std::vector<int>& baseIndices)
{
    if (basePositions.empty() || baseIndices.empty()) {
        std::cerr << "ClothLODConfig: Cannot generate LOD meshes from empty base mesh" << std::endl;
        return;
    }
    
    std::cout << "ClothLODConfig: Generating LOD meshes from base mesh with " 
              << basePositions.size() << " vertices..." << std::endl;
    
    for (auto& level : m_Levels) {
        if (level.lodIndex == 0) {
            // LOD 0 uses original mesh
            level.particlePositions = basePositions;
            level.triangleIndices = baseIndices;
            level.particleCount = static_cast<int>(basePositions.size());
            level.triangleCount = static_cast<int>(baseIndices.size()) / 3;
            level.hasMeshData = true;
            
            // Identity mapping for LOD 0
            level.particleMapping.resize(level.particleCount);
            for (int i = 0; i < level.particleCount; ++i) {
                level.particleMapping[i] = i;
            }
            
            std::cout << "  LOD 0: " << level.particleCount << " vertices (original)" << std::endl;
        } 
        else if (level.isFrozen) {
            // Frozen LOD doesn't need mesh data
            level.particleCount = 0;
            level.triangleCount = 0;
            level.hasMeshData = false;
            
            std::cout << "  LOD " << level.lodIndex << ": Frozen (no mesh)" << std::endl;
        }
        else {
            // Simplify mesh for this LOD
            float ratio = static_cast<float>(level.particleCount) / static_cast<float>(basePositions.size());
            
            // Ensure we don't try to simplify to too few vertices
            int targetCount = std::max(4, level.particleCount);
            
            auto result = ClothMeshSimplifier::Simplify(
                basePositions, 
                baseIndices, 
                targetCount
            );
            
            if (result.success) {
                level.particlePositions = result.positions;
                level.triangleIndices = result.indices;
                level.particleCount = result.simplifiedVertexCount;
                level.triangleCount = result.simplifiedTriangleCount;
                level.particleMapping = result.vertexMapping;
                level.hasMeshData = true;
                
                std::cout << "  LOD " << level.lodIndex << ": " 
                          << level.particleCount << " vertices ("
                          << (ratio * 100.0f) << "% of original)" << std::endl;
            } else {
                std::cerr << "  LOD " << level.lodIndex << ": Simplification failed" << std::endl;
                level.hasMeshData = false;
            }
        }
    }
    
    std::cout << "ClothLODConfig: LOD mesh generation complete" << std::endl;
}

ClothLODConfig ClothLODConfig::CreateDefault(int baseParticleCount, int baseTriangleCount) {
    ClothLODConfig config;
    config.SetHysteresis(2.0f);
    
    // LOD 0: Full quality (0-10m)
    ClothLODLevel lod0;
    lod0.lodIndex = 0;
    lod0.minDistance = 0.0f;
    lod0.particleCount = baseParticleCount;
    lod0.triangleCount = baseTriangleCount;
    lod0.solverIterations = 10;
    lod0.substeps = 1;
    lod0.updateFrequency = 1;
    lod0.isFrozen = false;
    config.AddLODLevel(lod0);
    
    // LOD 1: Medium quality (10-25m)
    ClothLODLevel lod1;
    lod1.lodIndex = 1;
    lod1.minDistance = 10.0f;
    lod1.particleCount = baseParticleCount / 2;
    lod1.triangleCount = baseTriangleCount / 2;
    lod1.solverIterations = 5;
    lod1.substeps = 1;
    lod1.updateFrequency = 1;
    lod1.isFrozen = false;
    config.AddLODLevel(lod1);
    
    // LOD 2: Low quality (25-50m)
    ClothLODLevel lod2;
    lod2.lodIndex = 2;
    lod2.minDistance = 25.0f;
    lod2.particleCount = baseParticleCount / 4;
    lod2.triangleCount = baseTriangleCount / 4;
    lod2.solverIterations = 3;
    lod2.substeps = 1;
    lod2.updateFrequency = 2;  // Update every 2 frames
    lod2.isFrozen = false;
    config.AddLODLevel(lod2);
    
    // LOD 3: Frozen (50m+)
    ClothLODLevel lod3;
    lod3.lodIndex = 3;
    lod3.minDistance = 50.0f;
    lod3.particleCount = 0;
    lod3.triangleCount = 0;
    lod3.solverIterations = 0;
    lod3.substeps = 0;
    lod3.updateFrequency = 0;
    lod3.isFrozen = true;
    config.AddLODLevel(lod3);
    
    return config;
}
