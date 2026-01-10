#pragma once

#include "Job.h"
#include "SpatialGrid.h"
#include "WorkStealingThreadPool.h"
#include <vector>
#include <mutex>

class PhysXSoftBody;

/**
 * @brief Collision result
 */
struct CollisionResult {
    PhysXSoftBody* objectA;
    PhysXSoftBody* objectB;
    Vec3 contactPoint;
    Vec3 contactNormal;
    float penetrationDepth;
};

/**
 * @brief Async collision detector using spatial grid and thread pool
 */
class AsyncCollisionDetector {
public:
    /**
     * @brief Constructor
     * @param threadPool Thread pool for parallel execution
     * @param cellSize Spatial grid cell size
     */
    explicit AsyncCollisionDetector(WorkStealingThreadPool& threadPool, float cellSize = 10.0f);
    
    /**
     * @brief Register object for collision detection
     */
    void RegisterObject(PhysXSoftBody* object);
    
    /**
     * @brief Unregister object
     */
    void UnregisterObject(PhysXSoftBody* object);
    
    /**
     * @brief Detect collisions asynchronously
     */
    void DetectCollisionsAsync();
    
    /**
     * @brief Fetch collision results
     */
    std::vector<CollisionResult> FetchResults();
    
    /**
     * @brief Clear all results
     */
    void ClearResults();
    
    /**
     * @brief Get number of registered objects
     */
    size_t GetObjectCount() const;

private:
    WorkStealingThreadPool& m_ThreadPool;
    SpatialGrid<PhysXSoftBody*> m_BroadPhase;
    std::vector<PhysXSoftBody*> m_Objects;
    std::vector<CollisionResult> m_Results;
    mutable std::mutex m_ObjectsMutex;
    mutable std::mutex m_ResultsMutex;
    
    /**
     * @brief Update spatial grid with current object positions
     */
    void UpdateBroadPhase();
    
    /**
     * @brief Perform narrow-phase collision detection
     */
    void NarrowPhaseCollision(PhysXSoftBody* a, PhysXSoftBody* b);
};
