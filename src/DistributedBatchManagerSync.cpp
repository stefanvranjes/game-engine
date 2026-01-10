// State Synchronization Implementation for DistributedBatchManager

#include "DistributedBatchManager.h"
#include "PhysXSoftBody.h"
#include <iostream>

// Add to Impl structure
struct DistributedBatchManager::Impl {
    // ... existing members ...
    
    // State synchronization
    uint32_t syncIntervalMs = 100;           // Sync every 100ms (10 Hz)
    bool useDeltaSync = true;                // Use delta encoding
    std::thread syncThread;
    std::unordered_map<PhysXSoftBody*, StateSerializer::SoftBodyState> lastSyncedStates;
    std::mutex syncMutex;
    uint32_t currentFrame = 0;
};

void DistributedBatchManager::StartStateSynchronization() {
    if (m_Impl->role != NodeRole::WORKER) {
        return;  // Only workers sync state back to master
    }
    
    m_Impl->syncThread = std::thread([this]() {
        SynchronizeState();
    });
    
    std::cout << "State synchronization started (interval: " 
              << m_Impl->syncIntervalMs << "ms)" << std::endl;
}

void DistributedBatchManager::SynchronizeState() {
    while (m_Impl->running) {
        // Sync all assigned soft bodies
        for (const auto& [batchId, softBodies] : m_Impl->assignedBatches) {
            for (PhysXSoftBody* softBody : softBodies) {
                SyncSoftBodyState(softBody);
            }
        }
        
        m_Impl->currentFrame++;
        
        std::this_thread::sleep_for(
            std::chrono::milliseconds(m_Impl->syncIntervalMs));
    }
}

void DistributedBatchManager::SyncSoftBodyState(PhysXSoftBody* softBody) {
    if (!softBody) return;
    
    NetworkManager::Message syncMsg;
    syncMsg.type = NetworkManager::MessageType::STATE_SYNC_DELTA;
    syncMsg.sourceNode = m_Impl->networkManager->GetLocalNodeId();
    syncMsg.targetNode = m_Impl->currentMasterId;
    
    std::lock_guard<std::mutex> lock(m_Impl->syncMutex);
    
    // Check if we should use delta or full sync
    bool useDelta = m_Impl->useDeltaSync && 
                    m_Impl->lastSyncedStates.count(softBody) > 0;
    
    if (useDelta) {
        // Delta sync - send only changes
        auto& lastState = m_Impl->lastSyncedStates[softBody];
        
        StateSerializer::SerializationOptions options;
        options.enableCompression = true;
        options.enableDeltaEncoding = true;
        
        std::vector<uint8_t> deltaData = m_Impl->stateSerializer->SerializeDelta(
            softBody, lastState, options);
        
        if (!deltaData.empty()) {
            syncMsg.data = deltaData;
            m_Impl->networkManager->SendMessage(
                m_Impl->currentMasterId, syncMsg, false);  // Unreliable for deltas
            
            // Update last synced state
            lastState = m_Impl->stateSerializer->CaptureState(softBody);
        }
    } else {
        // Full sync - send complete state
        syncMsg.type = NetworkManager::MessageType::STATE_SYNC_FULL;
        
        StateSerializer::SerializationOptions options;
        options.enableCompression = true;
        
        std::vector<uint8_t> fullData = m_Impl->stateSerializer->SerializeSoftBody(
            softBody, options);
        
        syncMsg.data = fullData;
        m_Impl->networkManager->SendMessage(
            m_Impl->currentMasterId, syncMsg, true);  // Reliable for full state
        
        // Store as last synced state
        m_Impl->lastSyncedStates[softBody] = 
            m_Impl->stateSerializer->CaptureState(softBody);
    }
}

void DistributedBatchManager::HandleStateSyncFull(int nodeId, const NetworkManager::Message& msg) {
    // Master receives full state update from worker
    
    if (m_Impl->role != NodeRole::MASTER) {
        return;
    }
    
    // Deserialize state
    StateSerializer::SerializationOptions options;
    options.enableCompression = true;
    
    // TODO: Find corresponding soft body
    PhysXSoftBody* softBody = nullptr;  // Placeholder
    
    if (softBody) {
        m_Impl->stateSerializer->DeserializeSoftBody(softBody, msg.data, options);
        
        std::cout << "Received full state sync from node " << nodeId << std::endl;
    }
}

void DistributedBatchManager::HandleStateSyncDelta(int nodeId, const NetworkManager::Message& msg) {
    // Master receives delta state update from worker
    
    if (m_Impl->role != NodeRole::MASTER) {
        return;
    }
    
    // TODO: Find corresponding soft body
    PhysXSoftBody* softBody = nullptr;  // Placeholder
    
    if (softBody) {
        StateSerializer::SerializationOptions options;
        options.enableCompression = true;
        options.enableDeltaEncoding = true;
        
        m_Impl->stateSerializer->ApplyDelta(softBody, msg.data, options);
        
        // Update statistics
        std::lock_guard<std::mutex> lock(m_Impl->statsMutex);
        m_Impl->stats.avgNetworkLatencyMs = 
            (m_Impl->stats.avgNetworkLatencyMs * 0.9f) + 
            (msg.timestamp * 0.1f);  // Exponential moving average
    }
}

void DistributedBatchManager::BroadcastStateUpdate(PhysXSoftBody* softBody) {
    // Master broadcasts state update to all workers
    
    if (m_Impl->role != NodeRole::MASTER) {
        return;
    }
    
    StateSerializer::SerializationOptions options;
    options.enableCompression = true;
    
    std::vector<uint8_t> stateData = m_Impl->stateSerializer->SerializeSoftBody(
        softBody, options);
    
    // Broadcast to all workers
    for (const auto& [nodeId, worker] : m_Impl->workers) {
        if (!worker.isHealthy) continue;
        
        NetworkManager::Message msg;
        msg.type = NetworkManager::MessageType::STATE_SYNC_FULL;
        msg.sourceNode = m_Impl->networkManager->GetLocalNodeId();
        msg.targetNode = nodeId;
        msg.data = stateData;
        
        m_Impl->networkManager->SendMessage(nodeId, msg, false);  // Unreliable broadcast
    }
}

void DistributedBatchManager::SetSyncInterval(uint32_t intervalMs) {
    m_Impl->syncIntervalMs = intervalMs;
    
    std::cout << "Sync interval set to " << intervalMs << "ms ("
              << (1000.0f / intervalMs) << " Hz)" << std::endl;
}

void DistributedBatchManager::EnableDeltaSync(bool enable) {
    m_Impl->useDeltaSync = enable;
    
    std::cout << "Delta sync " << (enable ? "enabled" : "disabled") << std::endl;
}

// Conflict resolution
void DistributedBatchManager::ResolveStateConflict(PhysXSoftBody* softBody,
                                                    const StateSerializer::SoftBodyState& workerState,
                                                    const StateSerializer::SoftBodyState& masterState) {
    // Simple conflict resolution: master wins
    // In production, could use:
    // - Timestamp-based (latest wins)
    // - Version vector (detect concurrent modifications)
    // - Application-specific logic
    
    std::cout << "State conflict detected, using master state" << std::endl;
    
    m_Impl->stateSerializer->RestoreState(softBody, masterState);
}

// Bandwidth optimization for sync
size_t DistributedBatchManager::EstimateSyncBandwidth() const {
    size_t totalBandwidth = 0;
    
    // Calculate bandwidth for all soft bodies
    for (const auto& [batchId, assignment] : m_Impl->batchAssignments) {
        for (PhysXSoftBody* softBody : assignment.softBodies) {
            // Estimate based on vertex count
            // Full state: ~12 bytes per vertex (position + velocity)
            // Delta state: ~3 bytes per vertex (compressed)
            
            size_t vertexCount = 1000;  // TODO: Get actual vertex count
            size_t stateSize = m_Impl->useDeltaSync ? 
                (vertexCount * 3) : (vertexCount * 12);
            
            totalBandwidth += stateSize;
        }
    }
    
    // Multiply by sync frequency
    float syncHz = 1000.0f / m_Impl->syncIntervalMs;
    totalBandwidth = static_cast<size_t>(totalBandwidth * syncHz);
    
    return totalBandwidth;  // Bytes per second
}

void DistributedBatchManager::OptimizeSyncBandwidth() {
    size_t estimatedBandwidth = EstimateSyncBandwidth();
    size_t maxBandwidth = 10 * 1024 * 1024;  // 10 MB/s limit
    
    if (estimatedBandwidth > maxBandwidth) {
        // Reduce sync frequency
        float ratio = static_cast<float>(maxBandwidth) / estimatedBandwidth;
        uint32_t newInterval = static_cast<uint32_t>(m_Impl->syncIntervalMs / ratio);
        
        std::cout << "Bandwidth optimization: reducing sync frequency from "
                  << (1000.0f / m_Impl->syncIntervalMs) << " Hz to "
                  << (1000.0f / newInterval) << " Hz" << std::endl;
        
        SetSyncInterval(newInterval);
    }
}

// Priority-based sync
void DistributedBatchManager::SyncHighPriorityFirst() {
    // Sync high priority soft bodies more frequently
    
    std::vector<std::pair<PhysXSoftBody*, uint32_t>> prioritizedBodies;
    
    for (const auto& [batchId, assignment] : m_Impl->batchAssignments) {
        for (PhysXSoftBody* softBody : assignment.softBodies) {
            prioritizedBodies.push_back({softBody, assignment.priority});
        }
    }
    
    // Sort by priority (highest first)
    std::sort(prioritizedBodies.begin(), prioritizedBodies.end(),
        [](const auto& a, const auto& b) {
            return a.second > b.second;
        });
    
    // Sync in priority order
    for (const auto& [softBody, priority] : prioritizedBodies) {
        SyncSoftBodyState(softBody);
    }
}

// Update message handlers
void DistributedBatchManager::HandleMasterMessage(int nodeId, const NetworkManager::Message& msg) {
    switch (msg.type) {
        case NetworkManager::MessageType::STATE_SYNC_FULL:
            HandleStateSyncFull(nodeId, msg);
            break;
            
        case NetworkManager::MessageType::STATE_SYNC_DELTA:
            HandleStateSyncDelta(nodeId, msg);
            break;
            
        // ... existing cases ...
        
        default:
            break;
    }
}

void DistributedBatchManager::HandleWorkerMessage(int nodeId, const NetworkManager::Message& msg) {
    switch (msg.type) {
        case NetworkManager::MessageType::STATE_SYNC_FULL:
            // Worker receives state update from master
            HandleStateSyncFull(nodeId, msg);
            break;
            
        // ... existing cases ...
        
        default:
            break;
    }
}
