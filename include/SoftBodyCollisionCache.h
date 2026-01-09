#pragma once

#include "Math/Vec3.h"
#include "PhysXSoftBody.h"
#include <vector>
#include <unordered_map>
#include <cstdint>

/**
 * @brief Cached collision shape data for a soft body
 */
struct CachedCollisionData {
    std::vector<Vec3> spherePositions;
    std::vector<float> sphereRadii;
    std::vector<PhysXSoftBody::CollisionCapsule> capsules;
    uint64_t frameGenerated;
    Vec3 centroid;
    bool isValid;
    
    CachedCollisionData() 
        : frameGenerated(0)
        , centroid(0, 0, 0)
        , isValid(false)
    {}
};

/**
 * @brief Singleton cache manager for soft body collision shapes
 * 
 * Caches collision shapes per soft body to avoid redundant PCA computations
 * when multiple cloth objects reference the same soft body.
 */
class SoftBodyCollisionCache {
public:
    static SoftBodyCollisionCache& Instance();
    
    /**
     * @brief Get collision spheres for a soft body (cached or fresh)
     * @param body Soft body to get shapes for
     * @param positions Output sphere positions
     * @param radii Output sphere radii
     * @param maxSpheres Maximum spheres to generate
     * @param currentFrame Current frame number for cache validation
     * @return Number of spheres
     */
    int GetSpheres(
        const PhysXSoftBody* body,
        std::vector<Vec3>& positions,
        std::vector<float>& radii,
        int maxSpheres,
        uint64_t currentFrame
    );
    
    /**
     * @brief Get collision capsules for a soft body (cached or fresh)
     * @param body Soft body to get shapes for
     * @param capsules Output capsules
     * @param maxCapsules Maximum capsules to generate
     * @param currentFrame Current frame number for cache validation
     * @return Number of capsules
     */
    int GetCapsules(
        const PhysXSoftBody* body,
        std::vector<PhysXSoftBody::CollisionCapsule>& capsules,
        int maxCapsules,
        uint64_t currentFrame
    );
    
    /**
     * @brief Invalidate cache for a specific soft body
     */
    void InvalidateCache(const PhysXSoftBody* body);
    
    /**
     * @brief Clear old cache entries
     * @param currentFrame Current frame number
     * @param maxAge Maximum age in frames before clearing
     */
    void ClearOldCaches(uint64_t currentFrame, uint64_t maxAge = 60);
    
    /**
     * @brief Clear all caches
     */
    void ClearAll();

private:
    SoftBodyCollisionCache() = default;
    ~SoftBodyCollisionCache() = default;
    SoftBodyCollisionCache(const SoftBodyCollisionCache&) = delete;
    SoftBodyCollisionCache& operator=(const SoftBodyCollisionCache&) = delete;
    
    std::unordered_map<const PhysXSoftBody*, CachedCollisionData> m_Cache;
    
    bool IsCacheValid(const PhysXSoftBody* body, const CachedCollisionData& data, uint64_t currentFrame) const;
};
