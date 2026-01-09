#include "SoftBodyCollisionCache.h"
#include <algorithm>

SoftBodyCollisionCache& SoftBodyCollisionCache::Instance() {
    static SoftBodyCollisionCache instance;
    return instance;
}

int SoftBodyCollisionCache::GetSpheres(
    const PhysXSoftBody* body,
    std::vector<Vec3>& positions,
    std::vector<float>& radii,
    int maxSpheres,
    uint64_t currentFrame
) {
    if (!body) {
        positions.clear();
        radii.clear();
        return 0;
    }
    
    auto it = m_Cache.find(body);
    
    // Check if we have valid cached data
    if (it != m_Cache.end() && IsCacheValid(body, it->second, currentFrame)) {
        // Use cached data
        positions = it->second.spherePositions;
        radii = it->second.sphereRadii;
        return static_cast<int>(positions.size());
    }
    
    // Generate fresh data
    int count = const_cast<PhysXSoftBody*>(body)->GetCollisionSpheres(positions, radii, maxSpheres);
    
    // Cache it
    CachedCollisionData& cache = m_Cache[body];
    cache.spherePositions = positions;
    cache.sphereRadii = radii;
    cache.frameGenerated = currentFrame;
    cache.centroid = body->GetCenterOfMass();
    cache.isValid = true;
    
    return count;
}

int SoftBodyCollisionCache::GetCapsules(
    const PhysXSoftBody* body,
    std::vector<PhysXSoftBody::CollisionCapsule>& capsules,
    int maxCapsules,
    uint64_t currentFrame
) {
    if (!body) {
        capsules.clear();
        return 0;
    }
    
    auto it = m_Cache.find(body);
    
    // Check if we have valid cached data
    if (it != m_Cache.end() && IsCacheValid(body, it->second, currentFrame)) {
        // Use cached data
        capsules = it->second.capsules;
        return static_cast<int>(capsules.size());
    }
    
    // Generate fresh data
    int count = const_cast<PhysXSoftBody*>(body)->GetCollisionCapsules(capsules, maxCapsules);
    
    // Cache it (update existing cache entry)
    CachedCollisionData& cache = m_Cache[body];
    cache.capsules = capsules;
    cache.frameGenerated = currentFrame;
    cache.centroid = body->GetCenterOfMass();
    cache.isValid = true;
    
    return count;
}

void SoftBodyCollisionCache::InvalidateCache(const PhysXSoftBody* body) {
    auto it = m_Cache.find(body);
    if (it != m_Cache.end()) {
        it->second.isValid = false;
    }
}

void SoftBodyCollisionCache::ClearOldCaches(uint64_t currentFrame, uint64_t maxAge) {
    for (auto it = m_Cache.begin(); it != m_Cache.end(); ) {
        if (currentFrame - it->second.frameGenerated > maxAge) {
            it = m_Cache.erase(it);
        } else {
            ++it;
        }
    }
}

void SoftBodyCollisionCache::ClearAll() {
    m_Cache.clear();
}

bool SoftBodyCollisionCache::IsCacheValid(
    const PhysXSoftBody* body,
    const CachedCollisionData& data,
    uint64_t currentFrame
) const {
    if (!data.isValid) {
        return false;
    }
    
    // Cache is valid for same frame
    if (data.frameGenerated == currentFrame) {
        return true;
    }
    
    // Check if centroid has moved significantly (invalidation threshold)
    Vec3 currentCentroid = body->GetCenterOfMass();
    Vec3 diff = currentCentroid - data.centroid;
    float distSq = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
    
    // If moved more than 0.1 units, invalidate
    const float threshold = 0.1f * 0.1f;
    if (distSq > threshold) {
        return false;
    }
    
    // Cache is still valid
    return true;
}
