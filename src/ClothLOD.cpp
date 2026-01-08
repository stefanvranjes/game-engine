#include "ClothLOD.h"
#include <algorithm>
#include <cmath>

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
