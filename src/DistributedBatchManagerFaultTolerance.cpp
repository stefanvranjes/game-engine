#include "DistributedBatchManager.h"
#include "DistributedBatchManagerImpl.h"

void DistributedBatchManager::SendHeartbeats() {
    while (m_Impl->running) {
        // Send heartbeat to master
        NetworkManager::Message heartbeat;
        heartbeat.type = NetworkManager::MessageType::HEARTBEAT;
        heartbeat.sourceNode = m_Impl->networkManager->GetLocalNodeId();
        heartbeat.targetNode = 0;  // Master is always node 0
        
        // TODO: Add current load info to heartbeat data
        
        m_Impl->networkManager->SendMessageToNode(0, heartbeat, false);  // Unreliable
        
        std::this_thread::sleep_for(
            std::chrono::milliseconds(m_Impl->heartbeatIntervalMs));
    }
}

void DistributedBatchManager::MonitorWorkerHealth() {
    while (m_Impl->running) {
        auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        
        std::vector<int> failedNodes;
        
        // Check each worker's heartbeat
        for (auto& [nodeId, worker] : m_Impl->workers) {
            if (!worker.isHealthy) {
                continue;  // Already marked as failed
            }
            
            uint64_t timeSinceHeartbeat = now - worker.lastHeartbeat;
            
            if (timeSinceHeartbeat > m_Impl->heartbeatTimeoutMs) {
                std::cerr << "Worker node " << nodeId 
                          << " heartbeat timeout (" << timeSinceHeartbeat 
                          << "ms), marking as failed" << std::endl;
                
                worker.isHealthy = false;
                failedNodes.push_back(nodeId);
                
                std::lock_guard<std::mutex> lock(m_Impl->statsMutex);
                m_Impl->stats.failedNodes++;
                m_Impl->stats.activeWorkers--;
            }
        }
        
        // Handle failed nodes
        for (int nodeId : failedNodes) {
            HandleNodeFailure(nodeId);
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
}

void DistributedBatchManager::HandleNodeFailure(int nodeId) {
    std::cout << "Handling failure of node " << nodeId << std::endl;
    
    // Reassign batches from failed node
    ReassignBatches(nodeId);
    
    // Disconnect from failed node
    m_Impl->networkManager->DisconnectNode(nodeId);
    
    // Log failure
    std::cerr << "Node " << nodeId << " failed and removed from cluster" << std::endl;
}

void DistributedBatchManager::ReassignBatches(int failedNodeId) {
    std::vector<uint32_t> batchesToReassign;
    
    // Find all batches assigned to failed node
    for (const auto& [batchId, assignment] : m_Impl->batchAssignments) {
        if (assignment.assignedNode == failedNodeId) {
            batchesToReassign.push_back(batchId);
        }
    }
    
    if (batchesToReassign.empty()) {
        return;
    }
    
    std::cout << "Reassigning " << batchesToReassign.size() 
              << " batches from failed node " << failedNodeId << std::endl;
    
    // Reassign each batch
    for (uint32_t batchId : batchesToReassign) {
        auto& assignment = m_Impl->batchAssignments[batchId];
        
        // Mark as unassigned
        assignment.assignedNode = -1;
        assignment.isProcessing = false;
        
        // Select new node
        int newNode = SelectNodeForBatch(assignment.softBodies, assignment.priority);
        
        if (newNode >= 0) {
            std::cout << "Reassigning batch " << batchId 
                      << " to node " << newNode << std::endl;
            
            AssignBatchToNode(batchId, newNode);
            
            std::lock_guard<std::mutex> lock(m_Impl->statsMutex);
            m_Impl->stats.batchesFailed++;
        } else {
            std::cerr << "Failed to reassign batch " << batchId 
                      << " - no healthy nodes available" << std::endl;
        }
    }
}

void DistributedBatchManager::RegisterWorker(int nodeId, const NetworkManager::NodeInfo& info) {
    WorkerNode worker;
    worker.nodeId = nodeId;
    worker.address = info.address;
    worker.port = info.port;
    worker.currentLoad = 0.0f;
    worker.gpuCount = info.gpuCount;
    worker.totalMemoryMB = info.totalMemoryMB;
    worker.availableMemoryMB = info.totalMemoryMB;
    worker.assignedBatches = 0;
    worker.isHealthy = true;
    worker.lastHeartbeat = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    
    m_Impl->workers[nodeId] = worker;
    
    std::lock_guard<std::mutex> lock(m_Impl->statsMutex);
    m_Impl->stats.totalWorkers++;
    m_Impl->stats.activeWorkers++;
    
    std::cout << "Registered worker node " << nodeId 
              << " (" << info.address << ":" << info.port 
              << ", GPUs: " << info.gpuCount << ")" << std::endl;
}


void DistributedBatchManager::SetHeartbeatConfig(uint32_t intervalMs, uint32_t timeoutMs) {
    m_Impl->heartbeatIntervalMs = intervalMs;
    m_Impl->heartbeatTimeoutMs = timeoutMs;
    
    std::cout << "Heartbeat config updated: interval=" << intervalMs 
              << "ms, timeout=" << timeoutMs << "ms" << std::endl;
}

bool DistributedBatchManager::IsNodeHealthy(int nodeId) const {
    const auto it = m_Impl->workers.find(nodeId);
    if (it != m_Impl->workers.end()) {
        return it->second.isHealthy;
    }
    return false;
}

