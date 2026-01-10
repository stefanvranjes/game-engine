// GPU-Direct RDMA Integration for DistributedBatchManager

#include "DistributedBatchManager.h"
#include "RdmaManager.h"
#include "PhysXSoftBody.h"
#include <iostream>

#ifdef HAS_RDMA

// Add to Impl structure
struct DistributedBatchManager::Impl {
    // ... existing members ...
    
    // RDMA support
    std::unique_ptr<RdmaManager> rdmaManager;
    bool rdmaEnabled = false;
    std::unordered_map<PhysXSoftBody*, void*> gpuBuffers;  // Soft body -> GPU buffer mapping
};

bool DistributedBatchManager::EnableRdma(bool enable) {
    if (enable && !m_Impl->rdmaManager) {
        m_Impl->rdmaManager = std::make_unique<RdmaManager>();
        
        if (!m_Impl->rdmaManager->Initialize()) {
            std::cerr << "Failed to initialize RDMA" << std::endl;
            m_Impl->rdmaManager.reset();
            return false;
        }
        
        std::cout << "RDMA enabled for distributed batch manager" << std::endl;
        m_Impl->rdmaEnabled = true;
        return true;
    } else if (!enable) {
        if (m_Impl->rdmaManager) {
            m_Impl->rdmaManager->Shutdown();
            m_Impl->rdmaManager.reset();
        }
        m_Impl->rdmaEnabled = false;
        std::cout << "RDMA disabled" << std::endl;
        return true;
    }
    
    return m_Impl->rdmaEnabled;
}

bool DistributedBatchManager::IsRdmaEnabled() const {
    return m_Impl->rdmaEnabled && m_Impl->rdmaManager && 
           m_Impl->rdmaManager->IsRdmaAvailable();
}

bool DistributedBatchManager::RegisterBufferForRdma(PhysXSoftBody* softBody, 
                                                     void* gpuPtr, size_t size) {
    if (!m_Impl->rdmaEnabled || !m_Impl->rdmaManager) {
        return false;
    }
    
    uint32_t lkey, rkey;
    if (!m_Impl->rdmaManager->RegisterGpuMemory(gpuPtr, size, lkey, rkey)) {
        std::cerr << "Failed to register GPU buffer for RDMA" << std::endl;
        return false;
    }
    
    m_Impl->gpuBuffers[softBody] = gpuPtr;
    
    std::cout << "Registered soft body GPU buffer for RDMA: " 
              << gpuPtr << " (" << (size / 1024) << " KB)" << std::endl;
    
    return true;
}

void DistributedBatchManager::MigrateSoftBodyRdma(PhysXSoftBody* softBody, int targetNode) {
    if (!m_Impl->rdmaEnabled) {
        std::cerr << "RDMA not enabled, falling back to standard migration" << std::endl;
        MigrateSoftBody(softBody, targetNode);
        return;
    }
    
    // Get GPU buffer for soft body
    auto it = m_Impl->gpuBuffers.find(softBody);
    if (it == m_Impl->gpuBuffers.end()) {
        std::cerr << "Soft body GPU buffer not registered for RDMA" << std::endl;
        MigrateSoftBody(softBody, targetNode);
        return;
    }
    
    void* localGpuPtr = it->second;
    
    // Calculate buffer size
    size_t vertexCount = 1000;  // TODO: Get actual vertex count from soft body
    size_t bufferSize = vertexCount * sizeof(float) * 6;  // Position + velocity
    
    // Get remote GPU buffer address (exchanged during connection)
    // TODO: Implement remote buffer address exchange
    void* remoteGpuPtr = nullptr;  // Placeholder
    
    if (!remoteGpuPtr) {
        std::cerr << "Remote GPU buffer address unknown" << std::endl;
        MigrateSoftBody(softBody, targetNode);
        return;
    }
    
    std::cout << "Migrating soft body via RDMA to node " << targetNode << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Direct GPU-to-GPU transfer via RDMA
    bool success = m_Impl->rdmaManager->RdmaWrite(
        targetNode,
        localGpuPtr,
        remoteGpuPtr,
        bufferSize,
        true  // Signaled
    );
    
    if (success) {
        // Wait for completion
        success = m_Impl->rdmaManager->WaitForCompletion(targetNode, 1000);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end - start).count();
    
    if (success) {
        std::cout << "RDMA migration completed in " << duration << " μs" << std::endl;
        
        // Update batch assignment
        // TODO: Update internal state
        
        std::lock_guard<std::mutex> lock(m_Impl->statsMutex);
        m_Impl->stats.totalMigrations++;
    } else {
        std::cerr << "RDMA migration failed, falling back to standard" << std::endl;
        MigrateSoftBody(softBody, targetNode);
    }
}

void DistributedBatchManager::SyncStateRdma(PhysXSoftBody* softBody, int targetNode) {
    if (!m_Impl->rdmaEnabled) {
        // Fall back to standard state sync
        SyncSoftBodyState(softBody);
        return;
    }
    
    auto it = m_Impl->gpuBuffers.find(softBody);
    if (it == m_Impl->gpuBuffers.end()) {
        SyncSoftBodyState(softBody);
        return;
    }
    
    void* localGpuPtr = it->second;
    size_t bufferSize = 1000 * sizeof(float) * 6;  // TODO: Get actual size
    
    // TODO: Get remote GPU buffer
    void* remoteGpuPtr = nullptr;
    
    if (!remoteGpuPtr) {
        SyncSoftBodyState(softBody);
        return;
    }
    
    // RDMA write for state sync
    bool success = m_Impl->rdmaManager->RdmaWrite(
        targetNode,
        localGpuPtr,
        remoteGpuPtr,
        bufferSize,
        false  // Unsignaled for performance
    );
    
    if (!success) {
        // Fall back to standard sync
        SyncSoftBodyState(softBody);
    }
}

// Get RDMA statistics
void DistributedBatchManager::PrintRdmaStatistics() const {
    if (!m_Impl->rdmaEnabled || !m_Impl->rdmaManager) {
        std::cout << "RDMA not enabled" << std::endl;
        return;
    }
    
    auto stats = m_Impl->rdmaManager->GetStatistics();
    
    std::cout << "\n=== RDMA Statistics ===" << std::endl;
    std::cout << "Total writes: " << stats.totalWrites << std::endl;
    std::cout << "Total reads: " << stats.totalReads << std::endl;
    std::cout << "Total bytes transferred: " 
              << (stats.totalBytesTransferred / 1024 / 1024) << " MB" << std::endl;
    std::cout << "Failed operations: " << stats.failedOperations << std::endl;
    std::cout << "Avg latency: " << stats.avgLatencyUs << " μs" << std::endl;
    std::cout << "Avg bandwidth: " << stats.avgBandwidthMBps << " MB/s" << std::endl;
}

#else

// Stub implementations when RDMA not available

bool DistributedBatchManager::EnableRdma(bool enable) {
    if (enable) {
        std::cerr << "RDMA support not compiled in" << std::endl;
    }
    return false;
}

bool DistributedBatchManager::IsRdmaEnabled() const {
    return false;
}

bool DistributedBatchManager::RegisterBufferForRdma(PhysXSoftBody* softBody,
                                                     void* gpuPtr, size_t size) {
    return false;
}

void DistributedBatchManager::MigrateSoftBodyRdma(PhysXSoftBody* softBody, int targetNode) {
    // Fall back to standard migration
    MigrateSoftBody(softBody, targetNode);
}

void DistributedBatchManager::SyncStateRdma(PhysXSoftBody* softBody, int targetNode) {
    // Fall back to standard sync
    SyncSoftBodyState(softBody);
}

void DistributedBatchManager::PrintRdmaStatistics() const {
    std::cout << "RDMA not available" << std::endl;
}

#endif
