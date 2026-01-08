#pragma once

#include "ClothLOD.h"
#include "Math/Vec3.h"
#include <memory>
#include <vector>
#include <unordered_map>

#ifdef USE_PHYSX

class PhysXCloth;

/**
 * @brief Global manager for cloth LOD system
 * 
 * Manages LOD transitions for all cloth objects based on distance from camera.
 */
class ClothLODManager {
public:
    /**
     * @brief Get singleton instance
     */
    static ClothLODManager& GetInstance();
    
    /**
     * @brief Set camera position for distance calculations
     */
    void SetCameraPosition(const Vec3& position);
    
    /**
     * @brief Get current camera position
     */
    const Vec3& GetCameraPosition() const { return m_CameraPosition; }
    
    /**
     * @brief Update all cloth LODs based on distance
     */
    void UpdateLODs();
    
    /**
     * @brief Register cloth for LOD management
     * @param cloth Cloth instance
     * @param position World position of cloth
     */
    void RegisterCloth(std::shared_ptr<PhysXCloth> cloth, const Vec3& position);
    
    /**
     * @brief Unregister cloth from LOD management
     */
    void UnregisterCloth(std::shared_ptr<PhysXCloth> cloth);
    
    /**
     * @brief Update cloth position (for moving cloth objects)
     */
    void UpdateClothPosition(std::shared_ptr<PhysXCloth> cloth, const Vec3& position);
    
    /**
     * @brief Set default LOD configuration for new cloths
     */
    void SetDefaultLODConfig(const ClothLODConfig& config);
    
    /**
     * @brief Get default LOD configuration
     */
    const ClothLODConfig& GetDefaultLODConfig() const { return m_DefaultConfig; }
    
    /**
     * @brief Enable/disable LOD system
     */
    void SetEnabled(bool enabled) { m_Enabled = enabled; }
    bool IsEnabled() const { return m_Enabled; }
    
    /**
     * @brief Get total number of registered cloths
     */
    int GetClothCount() const { return static_cast<int>(m_Cloths.size()); }
    
    /**
     * @brief Get number of cloths at specific LOD level
     */
    int GetClothCountByLOD(int lodLevel) const;
    
    /**
     * @brief Get statistics
     */
    struct Statistics {
        int totalCloths;
        int lod0Count;
        int lod1Count;
        int lod2Count;
        int lod3Count;
        int frozenCount;
    };
    Statistics GetStatistics() const;
    
private:
    ClothLODManager();
    ~ClothLODManager();
    
    // Prevent copying
    ClothLODManager(const ClothLODManager&) = delete;
    ClothLODManager& operator=(const ClothLODManager&) = delete;
    
    /**
     * @brief Entry for each registered cloth
     */
    struct ClothLODEntry {
        std::shared_ptr<PhysXCloth> cloth;
        Vec3 position;
        int currentLOD;
        int frameCounter;  // For update frequency control
        
        ClothLODEntry()
            : currentLOD(0)
            , frameCounter(0)
        {}
    };
    
    std::vector<ClothLODEntry> m_Cloths;
    Vec3 m_CameraPosition;
    ClothLODConfig m_DefaultConfig;
    bool m_Enabled;
    int m_FrameNumber;
    
    /**
     * @brief Calculate distance from cloth to camera
     */
    float GetDistance(const Vec3& position) const;
    
    /**
     * @brief Apply LOD transition to cloth
     */
    void ApplyLOD(ClothLODEntry& entry, int newLOD);
};

#endif // USE_PHYSX
