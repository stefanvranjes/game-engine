#pragma once

#include "PhysXScenePartition.h"
#include "WorkStealingThreadPool.h"
#include <vector>
#include <memory>
#include <mutex>

// Forward declarations
namespace physx {
    class PxPhysics;
}

class PhysXSoftBody;

/**
 * @brief Manages multiple PhysX scene partitions for parallel simulation
 */
class ScenePartitionManager {
public:
    /**
     * @brief Constructor
     * @param threadPool Thread pool for parallel execution
     * @param physics PhysX physics instance
     */
    ScenePartitionManager(WorkStealingThreadPool& threadPool, physx::PxPhysics* physics);
    
    /**
     * @brief Destructor
     */
    ~ScenePartitionManager();
    
    /**
     * @brief Set number of scene partitions
     * @param count Number of partitions (scenes)
     */
    void SetPartitionCount(size_t count);
    
    /**
     * @brief Get number of partitions
     */
    size_t GetPartitionCount() const { return m_Partitions.size(); }
    
    /**
     * @brief Register soft body (assigns to partition)
     */
    void RegisterSoftBody(PhysXSoftBody* softBody);
    
    /**
     * @brief Unregister soft body
     */
    void UnregisterSoftBody(PhysXSoftBody* softBody);
    
    /**
     * @brief Simulate all partitions in parallel
     * @param deltaTime Time step
     */
    void SimulateParallel(float deltaTime);
    
    /**
     * @brief Get least loaded partition index
     */
    int GetLeastLoadedPartition() const;
    
    /**
     * @brief Get total soft body count across all partitions
     */
    size_t GetTotalSoftBodyCount() const;

private:
    std::vector<std::unique_ptr<PhysXScenePartition>> m_Partitions;
    WorkStealingThreadPool& m_ThreadPool;
    physx::PxPhysics* m_Physics;
    mutable std::mutex m_Mutex;
    size_t m_NextPartitionRoundRobin;
};
