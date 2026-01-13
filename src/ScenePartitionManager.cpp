#include "ScenePartitionManager.h"
#ifdef USE_PHYSX
#include "PhysXSoftBody.h"
#include <PxPhysicsAPI.h>
#endif
#include <iostream>

ScenePartitionManager::ScenePartitionManager(WorkStealingThreadPool& threadPool, 
                                             physx::PxPhysics* physics)
    : m_ThreadPool(threadPool)
    , m_Physics(physics)
    , m_NextPartitionRoundRobin(0)
{
    if (!m_Physics) {
        std::cerr << "ScenePartitionManager: Invalid physics instance" << std::endl;
    }
}

ScenePartitionManager::~ScenePartitionManager() {
#ifdef USE_PHYSX
    m_Partitions.clear();
#endif
    std::cout << "ScenePartitionManager destroyed" << std::endl;
}

void ScenePartitionManager::SetPartitionCount(size_t count) {
    if (count == 0) {
        std::cerr << "ScenePartitionManager: Partition count must be > 0" << std::endl;
        return;
    }
    
    std::lock_guard<std::mutex> lock(m_Mutex);
    
#ifdef USE_PHYSX
    // Clear existing partitions
    m_Partitions.clear();
    
    // Create new partitions
    for (size_t i = 0; i < count; ++i) {
        m_Partitions.push_back(
            std::make_unique<PhysXScenePartition>(m_Physics, static_cast<int>(i))
        );
    }
#endif
    
    std::cout << "Created " << count << " scene partitions" << std::endl;
}

void ScenePartitionManager::RegisterSoftBody(PhysXSoftBody* softBody) {
    if (!softBody) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(m_Mutex);
    
#ifdef USE_PHYSX
    if (m_Partitions.empty()) {
        std::cerr << "ScenePartitionManager: No partitions created. Call SetPartitionCount first." << std::endl;
        return;
    }
    
    // Get least loaded partition
    int partitionIndex = GetLeastLoadedPartition();
    
    // Add to partition
    m_Partitions[partitionIndex]->AddSoftBody(softBody);
#endif
}

void ScenePartitionManager::UnregisterSoftBody(PhysXSoftBody* softBody) {
    if (!softBody) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(m_Mutex);
    
#ifdef USE_PHYSX
    // Find and remove from partition
    int partition = softBody->GetScenePartition();
    if (partition >= 0 && partition < static_cast<int>(m_Partitions.size())) {
        m_Partitions[partition]->RemoveSoftBody(softBody);
    }
#endif
}

void ScenePartitionManager::SimulateParallel(float deltaTime) {
#ifdef USE_PHYSX
    if (m_Partitions.empty()) {
        return;
    }
    
    std::vector<std::future<void>> futures;
    
    // Submit each partition to thread pool
    for (auto& partition : m_Partitions) {
        futures.push_back(m_ThreadPool.Submit([&partition, deltaTime]() {
            // Simulate this partition as raw pointer to avoid unique_ptr copy issues in lambda if needed, 
            // but unique_ptr is not copyable. Use reference or get().
            // Lambda capture of ref to unique_ptr is OK if lifetime exceeds lambda.
            partition->Simulate(deltaTime);
            partition->FetchResults(true);
        }));
    }
    
    // Wait for all partitions to complete
    for (auto& future : futures) {
        future.get();
    }
#endif
}

int ScenePartitionManager::GetLeastLoadedPartition() const {
    // Note: Caller must hold mutex
    int bestPartition = 0;
#ifdef USE_PHYSX
    size_t minLoad = SIZE_MAX;
    
    for (size_t i = 0; i < m_Partitions.size(); ++i) {
        size_t load = m_Partitions[i]->GetSoftBodyCount();
        if (load < minLoad) {
            minLoad = load;
            bestPartition = static_cast<int>(i);
        }
    }
#endif
    
    return bestPartition;
}

size_t ScenePartitionManager::GetTotalSoftBodyCount() const {
    std::lock_guard<std::mutex> lock(m_Mutex);
    
    size_t total = 0;
#ifdef USE_PHYSX
    for (const auto& partition : m_Partitions) {
        total += partition->GetSoftBodyCount();
    }
#endif
    
    return total;
}
