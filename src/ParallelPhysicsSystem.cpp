#include "ParallelPhysicsSystem.h"
#include <iostream>
#include <chrono>

ParallelPhysicsSystem::ParallelPhysicsSystem(physx::PxPhysics* physics)
    : m_Physics(physics)
    , m_ParallelPhysicsEnabled(true)
    , m_LastPhysicsTime(0.0f)
    , m_LastLODTime(0.0f)
{
}

void ParallelPhysicsSystem::Initialize(size_t threadCount, size_t sceneCount) {
    // Auto-detect thread count if not specified
    if (threadCount == 0) {
        threadCount = std::thread::hardware_concurrency();
    }
    
    // Default scene count to thread count
    if (sceneCount == 0) {
        sceneCount = threadCount;
    }
    
    std::cout << "Initializing ParallelPhysicsSystem:" << std::endl;
    std::cout << "  Threads: " << threadCount << std::endl;
    std::cout << "  Scenes: " << sceneCount << std::endl;
    
    // Create thread pool with work stealing
    m_ThreadPool = std::make_unique<WorkStealingThreadPool>(threadCount);
    
    // Create scene partition manager
    m_SceneManager = std::make_unique<ScenePartitionManager>(*m_ThreadPool, m_Physics);
    m_SceneManager->SetPartitionCount(sceneCount);
    
    // Create adaptive quality system
    m_QualitySystem = std::make_unique<AdaptiveQualityIntegration>();
    
    // Create parallel LOD manager
    m_LODManager = std::make_unique<ParallelSoftBodyManager>(*m_ThreadPool);
    
    std::cout << "ParallelPhysicsSystem initialized successfully" << std::endl;
}

void ParallelPhysicsSystem::RegisterSoftBody(PhysXSoftBody* softBody, SoftBodyLODManager* lodManager) {
    if (!softBody) {
        return;
    }
    
    // Register with scene manager for parallel physics
    if (m_SceneManager) {
        m_SceneManager->RegisterSoftBody(softBody);
    }
    
    // Register with quality system
    if (m_QualitySystem && lodManager) {
        m_QualitySystem->RegisterSoftBody(softBody, lodManager);
    }
    
    // Register with LOD manager
    if (m_LODManager && lodManager) {
        m_LODManager->RegisterSoftBody(softBody, lodManager);
    }
}

void ParallelPhysicsSystem::UnregisterSoftBody(PhysXSoftBody* softBody) {
    if (!softBody) {
        return;
    }
    
    if (m_SceneManager) {
        m_SceneManager->UnregisterSoftBody(softBody);
    }
    
    if (m_LODManager) {
        m_LODManager->UnregisterSoftBody(softBody);
    }
}

void ParallelPhysicsSystem::Update(float deltaTime, const Vec3& cameraPosition) {
    // Update adaptive quality
    if (m_QualitySystem) {
        m_QualitySystem->Update(deltaTime, cameraPosition);
    }
    
    // Parallel LOD updates
    auto lodStart = std::chrono::high_resolution_clock::now();
    if (m_LODManager) {
        m_LODManager->UpdateLODsParallel(cameraPosition, deltaTime);
    }
    auto lodEnd = std::chrono::high_resolution_clock::now();
    m_LastLODTime = std::chrono::duration<float, std::milli>(lodEnd - lodStart).count();
    
    // Parallel physics simulation
    auto physicsStart = std::chrono::high_resolution_clock::now();
    if (m_ParallelPhysicsEnabled && m_SceneManager) {
        m_SceneManager->SimulateParallel(deltaTime);
    }
    auto physicsEnd = std::chrono::high_resolution_clock::now();
    m_LastPhysicsTime = std::chrono::duration<float, std::milli>(physicsEnd - physicsStart).count();
}

void ParallelPhysicsSystem::EnableAdaptiveQuality(bool enable) {
    if (m_QualitySystem) {
        m_QualitySystem->EnableAdaptiveQuality(enable);
    }
}

void ParallelPhysicsSystem::EnableThreadAffinity(bool enable) {
    if (m_ThreadPool) {
        m_ThreadPool->EnableThreadAffinity(enable);
    }
}

ParallelPhysicsSystem::Statistics ParallelPhysicsSystem::GetStatistics() const {
    Statistics stats = {};
    
    if (m_ThreadPool) {
        stats.threadCount = m_ThreadPool->GetThreadCount();
    }
    
    if (m_SceneManager) {
        stats.sceneCount = m_SceneManager->GetPartitionCount();
        stats.totalSoftBodies = m_SceneManager->GetTotalSoftBodyCount();
        
        if (stats.sceneCount > 0) {
            stats.averageBodiesPerScene = static_cast<float>(stats.totalSoftBodies) / stats.sceneCount;
        }
    }
    
    stats.lastPhysicsTime = m_LastPhysicsTime;
    stats.lastLODTime = m_LastLODTime;
    stats.lastFrameTime = m_LastPhysicsTime + m_LastLODTime;
    
    return stats;
}
