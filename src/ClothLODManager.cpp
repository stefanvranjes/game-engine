#include "ClothLODManager.h"

#ifdef USE_PHYSX

#include "PhysXCloth.h"
#include <algorithm>
#include <cmath>
#include <iostream>

ClothLODManager::ClothLODManager()
    : m_CameraPosition(0, 0, 0)
    , m_Enabled(true)
    , m_FrameNumber(0)
{
    // Create default configuration
    m_DefaultConfig = ClothLODConfig::CreateDefault(100, 100);
    
    std::cout << "ClothLODManager initialized" << std::endl;
}

ClothLODManager::~ClothLODManager() {
    m_Cloths.clear();
}

ClothLODManager& ClothLODManager::GetInstance() {
    static ClothLODManager instance;
    return instance;
}

void ClothLODManager::SetCameraPosition(const Vec3& position) {
    m_CameraPosition = position;
}

void ClothLODManager::UpdateLODs() {
    if (!m_Enabled) {
        return;
    }
    
    m_FrameNumber++;
    
    for (auto& entry : m_Cloths) {
        if (!entry.cloth) {
            continue;
        }
        
        // Calculate distance to camera
        float distance = GetDistance(entry.position);
        
        // Get appropriate LOD for this distance
        int targetLOD = m_DefaultConfig.GetLODForDistance(distance, entry.currentLOD);
        
        // Apply LOD if changed
        if (targetLOD != entry.currentLOD) {
            ApplyLOD(entry, targetLOD);
        }
        
        // Update frame counter for update frequency control
        entry.frameCounter = m_FrameNumber;
    }
}

void ClothLODManager::RegisterCloth(std::shared_ptr<PhysXCloth> cloth, const Vec3& position) {
    if (!cloth) {
        return;
    }
    
    // Check if already registered
    for (const auto& entry : m_Cloths) {
        if (entry.cloth == cloth) {
            return;  // Already registered
        }
    }
    
    // Add new entry
    ClothLODEntry entry;
    entry.cloth = cloth;
    entry.position = position;
    entry.currentLOD = 0;  // Start at highest quality
    entry.frameCounter = m_FrameNumber;
    
    m_Cloths.push_back(entry);
    
    std::cout << "ClothLODManager: Registered cloth at position (" 
              << position.x << ", " << position.y << ", " << position.z << ")" 
              << std::endl;
}

void ClothLODManager::UnregisterCloth(std::shared_ptr<PhysXCloth> cloth) {
    if (!cloth) {
        return;
    }
    
    m_Cloths.erase(
        std::remove_if(m_Cloths.begin(), m_Cloths.end(),
            [&cloth](const ClothLODEntry& entry) {
                return entry.cloth == cloth;
            }),
        m_Cloths.end()
    );
}

void ClothLODManager::UpdateClothPosition(std::shared_ptr<PhysXCloth> cloth, const Vec3& position) {
    if (!cloth) {
        return;
    }
    
    for (auto& entry : m_Cloths) {
        if (entry.cloth == cloth) {
            entry.position = position;
            return;
        }
    }
}

void ClothLODManager::SetDefaultLODConfig(const ClothLODConfig& config) {
    m_DefaultConfig = config;
}

int ClothLODManager::GetClothCountByLOD(int lodLevel) const {
    int count = 0;
    for (const auto& entry : m_Cloths) {
        if (entry.currentLOD == lodLevel) {
            count++;
        }
    }
    return count;
}

ClothLODManager::Statistics ClothLODManager::GetStatistics() const {
    Statistics stats;
    stats.totalCloths = static_cast<int>(m_Cloths.size());
    stats.lod0Count = GetClothCountByLOD(0);
    stats.lod1Count = GetClothCountByLOD(1);
    stats.lod2Count = GetClothCountByLOD(2);
    stats.lod3Count = GetClothCountByLOD(3);
    
    stats.frozenCount = 0;
    for (const auto& entry : m_Cloths) {
        if (entry.cloth && entry.cloth->IsFrozen()) {
            stats.frozenCount++;
        }
    }
    
    return stats;
}

float ClothLODManager::GetDistance(const Vec3& position) const {
    Vec3 delta = position - m_CameraPosition;
    return std::sqrt(delta.x * delta.x + delta.y * delta.y + delta.z * delta.z);
}

void ClothLODManager::ApplyLOD(ClothLODEntry& entry, int newLOD) {
    if (!entry.cloth) {
        return;
    }
    
    const ClothLODLevel* lodLevel = m_DefaultConfig.GetLODLevel(newLOD);
    if (!lodLevel) {
        return;
    }
    
    std::cout << "ClothLODManager: Transitioning cloth from LOD " 
              << entry.currentLOD << " to LOD " << newLOD << std::endl;
    
    // Apply LOD to cloth
    entry.cloth->SetLOD(newLOD);
    entry.currentLOD = newLOD;
    
    // Handle frozen state
    if (lodLevel->isFrozen) {
        entry.cloth->Freeze();
    } else {
        entry.cloth->Unfreeze();
    }
}

#endif // USE_PHYSX
