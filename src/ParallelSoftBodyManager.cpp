#include "ParallelSoftBodyManager.h"
#include "PhysXSoftBody.h"
#include "SoftBodyLODManager.h"
#include <algorithm>
#include <iostream>

ParallelSoftBodyManager::ParallelSoftBodyManager(WorkStealingThreadPool& threadPool)
    : m_ThreadPool(threadPool)
    , m_BatchSize(4)
    , m_ParallelEnabled(true)
{
}

void ParallelSoftBodyManager::RegisterSoftBody(PhysXSoftBody* softBody, SoftBodyLODManager* lodManager) {
    if (!softBody) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(m_SoftBodyMutex);
    
    // Check if already registered
    for (const auto& entry : m_SoftBodies) {
        if (entry.softBody == softBody) {
            return;
        }
    }
    
    m_SoftBodies.push_back({softBody, lodManager});
    std::cout << "Registered soft body for parallel updates (total: " << m_SoftBodies.size() << ")" << std::endl;
}

void ParallelSoftBodyManager::UnregisterSoftBody(PhysXSoftBody* softBody) {
    std::lock_guard<std::mutex> lock(m_SoftBodyMutex);
    
    auto it = std::remove_if(m_SoftBodies.begin(), m_SoftBodies.end(),
        [softBody](const SoftBodyEntry& entry) {
            return entry.softBody == softBody;
        });
    
    if (it != m_SoftBodies.end()) {
        m_SoftBodies.erase(it, m_SoftBodies.end());
        std::cout << "Unregistered soft body from parallel updates (total: " << m_SoftBodies.size() << ")" << std::endl;
    }
}

void ParallelSoftBodyManager::UpdateParallel(float deltaTime, const Vec3& cameraPosition) {
    if (!m_ParallelEnabled || m_SoftBodies.empty()) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(m_SoftBodyMutex);
    
    size_t threadCount = m_ThreadPool.GetThreadCount();
    size_t softBodyCount = m_SoftBodies.size();
    
    // Calculate batch size
    size_t batchSize = std::max(m_BatchSize, (softBodyCount + threadCount - 1) / threadCount);
    
    // Submit batches to thread pool
    std::vector<std::future<void>> futures;
    for (size_t i = 0; i < softBodyCount; i += batchSize) {
        size_t end = std::min(i + batchSize, softBodyCount);
        
        futures.push_back(m_ThreadPool.Submit([this, i, end, deltaTime, cameraPosition]() {
            UpdateBatch(i, end, deltaTime, cameraPosition);
        }));
    }
    
    // Wait for all batches to complete
    for (auto& future : futures) {
        future.get();
    }
}

void ParallelSoftBodyManager::UpdateLODsParallel(const Vec3& cameraPosition, float deltaTime) {
    if (!m_ParallelEnabled || m_SoftBodies.empty()) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(m_SoftBodyMutex);
    
    size_t threadCount = m_ThreadPool.GetThreadCount();
    size_t softBodyCount = m_SoftBodies.size();
    
    // Calculate batch size
    size_t batchSize = std::max(m_BatchSize, (softBodyCount + threadCount - 1) / threadCount);
    
    // Submit batches to thread pool
    std::vector<std::future<void>> futures;
    for (size_t i = 0; i < softBodyCount; i += batchSize) {
        size_t end = std::min(i + batchSize, softBodyCount);
        
        futures.push_back(m_ThreadPool.Submit([this, i, end, cameraPosition, deltaTime]() {
            UpdateLODBatch(i, end, cameraPosition, deltaTime);
        }));
    }
    
    // Wait for all batches to complete
    for (auto& future : futures) {
        future.get();
    }
}

size_t ParallelSoftBodyManager::GetSoftBodyCount() const {
    std::lock_guard<std::mutex> lock(m_SoftBodyMutex);
    return m_SoftBodies.size();
}

void ParallelSoftBodyManager::UpdateBatch(size_t start, size_t end, float deltaTime, const Vec3& cameraPosition) {
    for (size_t i = start; i < end; ++i) {
        auto& entry = m_SoftBodies[i];
        
        if (!entry.softBody) {
            continue;
        }
        
        // Update LOD if manager is available
        if (entry.lodManager) {
            entry.lodManager->UpdateLOD(entry.softBody, cameraPosition, deltaTime);
            
            // Only update physics if LOD says we should
            if (!entry.lodManager->ShouldUpdateThisFrame()) {
                continue;
            }
        }
        
        // Update soft body physics
        // Note: PhysXSoftBody::Update would need to be thread-safe
        // For now, we just update LOD in parallel
    }
}

void ParallelSoftBodyManager::UpdateLODBatch(size_t start, size_t end, const Vec3& cameraPosition, float deltaTime) {
    for (size_t i = start; i < end; ++i) {
        auto& entry = m_SoftBodies[i];
        
        if (entry.softBody && entry.lodManager) {
            entry.lodManager->UpdateLOD(entry.softBody, cameraPosition, deltaTime);
        }
    }
}
