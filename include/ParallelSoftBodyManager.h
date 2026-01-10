#pragma once

#include "ThreadPool.h"
#include "Math/Vec3.h"
#include <vector>
#include <mutex>

class PhysXSoftBody;
class SoftBodyLODManager;

/**
 * @brief Manages parallel soft body updates using thread pool
 */
class ParallelSoftBodyManager {
public:
    /**
     * @brief Constructor
     * @param threadPool Reference to thread pool
     */
    explicit ParallelSoftBodyManager(ThreadPool& threadPool);
    
    /**
     * @brief Register soft body for parallel updates
     */
    void RegisterSoftBody(PhysXSoftBody* softBody, SoftBodyLODManager* lodManager);
    
    /**
     * @brief Unregister soft body
     */
    void UnregisterSoftBody(PhysXSoftBody* softBody);
    
    /**
     * @brief Update all soft bodies in parallel
     * @param deltaTime Frame time
     * @param cameraPosition Camera position for LOD
     */
    void UpdateParallel(float deltaTime, const Vec3& cameraPosition);
    
    /**
     * @brief Update LODs in parallel (without physics update)
     */
    void UpdateLODsParallel(const Vec3& cameraPosition, float deltaTime);
    
    /**
     * @brief Enable/disable parallel updates
     */
    void EnableParallelUpdates(bool enable) { m_ParallelEnabled = enable; }
    
    /**
     * @brief Check if parallel updates are enabled
     */
    bool IsParallelEnabled() const { return m_ParallelEnabled; }
    
    /**
     * @brief Set batch size for parallel processing
     * @param batchSize Number of soft bodies per batch
     */
    void SetBatchSize(size_t batchSize) { m_BatchSize = std::max(size_t(1), batchSize); }
    
    /**
     * @brief Get batch size
     */
    size_t GetBatchSize() const { return m_BatchSize; }
    
    /**
     * @brief Get number of registered soft bodies
     */
    size_t GetSoftBodyCount() const;

private:
    struct SoftBodyEntry {
        PhysXSoftBody* softBody;
        SoftBodyLODManager* lodManager;
    };
    
    ThreadPool& m_ThreadPool;
    std::vector<SoftBodyEntry> m_SoftBodies;
    mutable std::mutex m_SoftBodyMutex;
    
    size_t m_BatchSize;
    bool m_ParallelEnabled;
    
    /**
     * @brief Update a batch of soft bodies
     */
    void UpdateBatch(size_t start, size_t end, float deltaTime, const Vec3& cameraPosition);
    
    /**
     * @brief Update LODs for a batch of soft bodies
     */
    void UpdateLODBatch(size_t start, size_t end, const Vec3& cameraPosition, float deltaTime);
};
