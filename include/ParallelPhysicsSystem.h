#pragma once

#include "ScenePartitionManager.h"
#include "AdaptiveQualityIntegration.h"
#include "ParallelSoftBodyManager.h"
#include "GpuBatchManager.h"
#include <memory>

/**
 * @brief Complete parallel physics system integrating all components
 * 
 * Combines scene partitioning, adaptive quality, and parallel LOD updates
 * for optimal soft body simulation performance.
 */
class ParallelPhysicsSystem {
public:
    /**
     * @brief Constructor
     * @param physics PhysX physics instance
     */
    explicit ParallelPhysicsSystem(physx::PxPhysics* physics);
    
    /**
     * @brief Initialize the system
     * @param threadCount Number of threads (default: auto-detect)
     * @param sceneCount Number of scene partitions (default: thread count)
     */
    void Initialize(size_t threadCount = 0, size_t sceneCount = 0);
    
    /**
     * @brief Register soft body for parallel simulation
     */
    void RegisterSoftBody(PhysXSoftBody* softBody, SoftBodyLODManager* lodManager);
    
    /**
     * @brief Unregister soft body
     */
    void UnregisterSoftBody(PhysXSoftBody* softBody);
    
    /**
     * @brief Update all systems
     * @param deltaTime Frame time
     * @param cameraPosition Camera position for LOD
     */
    void Update(float deltaTime, const Vec3& cameraPosition);
    
    /**
     * @brief Enable/disable parallel physics simulation
     */
    void EnableParallelPhysics(bool enable) { m_ParallelPhysicsEnabled = enable; }
    
    /**
     * @brief Enable/disable adaptive quality
     */
    void EnableAdaptiveQuality(bool enable);
    
    /**
     * @brief Enable/disable thread affinity
     */
    void EnableThreadAffinity(bool enable);
    
    /**
     * @brief Get statistics
     */
    struct Statistics {
        size_t threadCount;
        size_t sceneCount;
        size_t totalSoftBodies;
        float averageBodiesPerScene;
        float lastFrameTime;
        float lastPhysicsTime;
        float lastLODTime;
        
        // GPU statistics
        bool gpuEnabled;
        size_t gpuMemoryUsedMB;
        float gpuSimulationTimeMs;
        size_t softBodiesOnGpu;
    };
    
    Statistics GetStatistics() const;

private:
    std::unique_ptr<WorkStealingThreadPool> m_ThreadPool;
    std::unique_ptr<ScenePartitionManager> m_SceneManager;
    std::unique_ptr<AdaptiveQualityIntegration> m_QualitySystem;
    std::unique_ptr<ParallelSoftBodyManager> m_LODManager;
    std::unique_ptr<GpuBatchManager> m_BatchManager;
    
    physx::PxPhysics* m_Physics;
    bool m_ParallelPhysicsEnabled;
    bool m_BatchGpuEnabled;
    
    // Performance tracking
    float m_LastPhysicsTime;
    float m_LastLODTime;
};
