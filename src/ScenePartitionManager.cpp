#include "ScenePartitionManager.h"
#include "PhysXSoftBody.h"
#ifdef USE_PHYSX
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
    m_Partitions.clear();
    std::cout << "ScenePartitionManager destroyed" << std::endl;
}

void ScenePartitionManager::SetPartitionCount(size_t count) {
    if (count == 0) {
        std::cerr << "ScenePartitionManager: Partition count must be > 0" << std::endl;
        return;
    }
    
    std::lock_guard<std::mutex> lock(m_Mutex);
    
    // Clear existing partitions
    m_Partitions.clear();
    
    // Create new partitions
    for (size_t i = 0; i < count; ++i) {
        m_Partitions.push_back(
            std::make_unique<PhysXScenePartition>(m_Physics, static_cast<int>(i))
        );
    }
    
    std::cout << "Created " << count << " scene partitions" << std::endl;
}

void ScenePartitionManager::RegisterSoftBody(PhysXSoftBody* softBody) {
    if (!softBody) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(m_Mutex);
    
    if (m_Partitions.empty()) {
        std::cerr << "ScenePartitionManager: No partitions created. Call SetPartitionCount first." << std::endl;
        return;
    }
    
    // Get least loaded partition
    int partitionIndex = GetLeastLoadedPartition();
    
    // Add to partition
    m_Partitions[partitionIndex]->AddSoftBody(softBody);
}

void ScenePartitionManager::UnregisterSoftBody(PhysXSoftBody* softBody) {
    if (!softBody) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(m_Mutex);
    
    // Find and remove from partition
    int partition = softBody->GetScenePartition();
    if (partition >= 0 && partition < static_cast<int>(m_Partitions.size())) {
        m_Partitions[partition]->RemoveSoftBody(softBody);
    }
}

void ScenePartitionManager::SimulateParallel(float deltaTime) {
    if (m_Partitions.empty()) {
        return;
    }
    
    std::vector<std::future<void>> futures;
    
    // Submit each partition to thread pool
    for (auto& partition : m_Partitions) {
        futures.push_back(m_ThreadPool.Submit([&partition, deltaTime]() {
            // Simulate this partition
            partition->Simulate(deltaTime);
            partition->FetchResults(true);
        }));
    }
    
    // Wait for all partitions to complete
    for (auto& future : futures) {
        future.get();
    }
}

int ScenePartitionManager::GetLeastLoadedPartition() const {
    // Note: Caller must hold mutex
    
    int bestPartition = 0;
    size_t minLoad = SIZE_MAX;
    
    for (size_t i = 0; i < m_Partitions.size(); ++i) {
        size_t load = m_Partitions[i]->GetSoftBodyCount();
        if (load < minLoad) {
            minLoad = load;
            bestPartition = static_cast<int>(i);
        }
    }
    
    return bestPartition;
}

size_t ScenePartitionManager::GetTotalSoftBodyCount() const {
    std::lock_guard<std::mutex> lock(m_Mutex);
    
    size_t total = 0;
    for (const auto& partition : m_Partitions) {
        total += partition->GetSoftBodyCount();
    }
    
    return total;
}
