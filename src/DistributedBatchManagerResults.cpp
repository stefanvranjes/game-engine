#include "DistributedBatchManager.h"
#include "DistributedBatchManagerImpl.h"
#include "PhysXSoftBody.h"
#include <iostream>

// Worker-side: Process assigned batch and send results
void DistributedBatchManager::ProcessAssignedBatch(uint32_t batchId, float deltaTime) {
    if (m_Impl->role != NodeRole::WORKER) {
        std::cerr << "ProcessAssignedBatch called on non-worker node" << std::endl;
        return;
    }
    
    auto it = m_Impl->assignedBatches.find(batchId);
    if (it == m_Impl->assignedBatches.end()) {
        std::cerr << "Batch " << batchId << " not found in assigned batches" << std::endl;
        return;
    }
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    std::vector<PhysXSoftBody*>& softBodies = it->second;
    
    std::cout << "Processing batch " << batchId << " with " 
              << softBodies.size() << " soft bodies" << std::endl;
    
    try {
        // Process batch using local GPU batch manager
        if (m_Impl->localBatchManager) {
            // Add soft bodies to local batch manager if not already added
            for (PhysXSoftBody* softBody : softBodies) {
                // TODO: Check if already added
                m_Impl->localBatchManager->AddSoftBody(softBody, 5);  // Medium priority
            }
            
            // Process batches
            m_Impl->localBatchManager->ProcessBatches(deltaTime);
        }
        
        // Serialize results
        std::vector<uint8_t> resultData = SerializeBatchResults(softBodies);
        
        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            endTime - startTime).count();
        
        // Send results back to master
        SendBatchResults(batchId, resultData, duration, true, "");
        
    } catch (const std::exception& e) {
        std::cerr << "Error processing batch " << batchId << ": " 
                  << e.what() << std::endl;
        
        // Send error result
        SendBatchResults(batchId, {}, 0, false, e.what());
    }
}

std::vector<uint8_t> DistributedBatchManager::SerializeBatchResults(
    const std::vector<PhysXSoftBody*>& softBodies) {
    
    std::vector<uint8_t> allData;
    
    // Serialize each soft body's state
    for (PhysXSoftBody* softBody : softBodies) {
        StateSerializer::SerializationOptions options;
        options.enableCompression = true;
        options.enableDeltaEncoding = false;  // Full state for results
        
        std::vector<uint8_t> softBodyData = 
            m_Impl->stateSerializer->SerializeSoftBody(softBody, options);
        
        // Prepend size
        uint32_t size = static_cast<uint32_t>(softBodyData.size());
        allData.insert(allData.end(), 
                      reinterpret_cast<uint8_t*>(&size),
                      reinterpret_cast<uint8_t*>(&size) + sizeof(uint32_t));
        
        // Append data
        allData.insert(allData.end(), softBodyData.begin(), softBodyData.end());
    }
    
    return allData;
}

void DistributedBatchManager::SendBatchResults(uint32_t batchId, 
                                                const std::vector<uint8_t>& resultData,
                                                uint64_t processingTimeMs,
                                                bool success,
                                                const std::string& errorMessage) {
    
    NetworkManager::Message msg;
    msg.type = NetworkManager::MessageType::BATCH_RESULT;
    msg.sourceNode = m_Impl->networkManager->GetLocalNodeId();
    msg.targetNode = m_Impl->currentMasterId;
    
    // Pack result data
    std::vector<uint8_t> packedData;
    
    // Batch ID
    packedData.insert(packedData.end(),
                     reinterpret_cast<const uint8_t*>(&batchId),
                     reinterpret_cast<const uint8_t*>(&batchId) + sizeof(uint32_t));
    
    // Processing time
    packedData.insert(packedData.end(),
                     reinterpret_cast<const uint8_t*>(&processingTimeMs),
                     reinterpret_cast<const uint8_t*>(&processingTimeMs) + sizeof(uint64_t));
    
    // Success flag
    uint8_t successByte = success ? 1 : 0;
    packedData.push_back(successByte);
    
    // Error message length and data
    uint32_t errorLen = static_cast<uint32_t>(errorMessage.size());
    packedData.insert(packedData.end(),
                     reinterpret_cast<const uint8_t*>(&errorLen),
                     reinterpret_cast<const uint8_t*>(&errorLen) + sizeof(uint32_t));
    
    if (errorLen > 0) {
        packedData.insert(packedData.end(), 
                         errorMessage.begin(), 
                         errorMessage.end());
    }
    
    // Result data
    packedData.insert(packedData.end(), resultData.begin(), resultData.end());
    
    msg.data = packedData;
    
    // Send with reliability
    m_Impl->networkManager->SendMessageToNode(m_Impl->currentMasterId, msg, true);
    
    std::cout << "Sent batch " << batchId << " results to master ("
              << (resultData.size() / 1024) << " KB, "
              << processingTimeMs << "ms)" << std::endl;
}

// Master-side: Handle batch results from workers
void DistributedBatchManager::HandleBatchResult(int nodeId, 
                                                 uint32_t batchId, 
                                                 const std::vector<uint8_t>& resultData) {
    
    if (m_Impl->role != NodeRole::MASTER) {
        return;
    }
    
    std::cout << "Received batch " << batchId << " results from node " 
              << nodeId << std::endl;
    
    // Find batch assignment
    auto it = m_Impl->batchAssignments.find(batchId);
    if (it == m_Impl->batchAssignments.end()) {
        std::cerr << "Received results for unknown batch " << batchId << std::endl;
        return;
    }
    
    BatchAssignment& assignment = it->second;
    
    // Verify it came from the assigned node
    if (assignment.assignedNode != nodeId) {
        std::cerr << "Received results from wrong node (expected " 
                  << assignment.assignedNode << ", got " << nodeId << ")" << std::endl;
        return;
    }
    
    // Deserialize and apply results
    DeserializeBatchResults(assignment.softBodies, resultData);
    
    // Mark batch as completed
    assignment.isProcessing = false;
    
    // Update worker load
    auto workerIt = m_Impl->workers.find(nodeId);
    if (workerIt != m_Impl->workers.end()) {
        workerIt->second.assignedBatches--;
        UpdateWorkerLoad(nodeId, workerIt->second.currentLoad - 0.1f);
    }
    
    // Update statistics
    {
        std::lock_guard<std::mutex> lock(m_Impl->statsMutex);
        m_Impl->stats.batchesCompleted++;
    }
    
    std::cout << "Batch " << batchId << " completed successfully" << std::endl;
}

void DistributedBatchManager::DeserializeBatchResults(
    const std::vector<PhysXSoftBody*>& softBodies,
    const std::vector<uint8_t>& resultData) {
    
    size_t offset = 0;
    size_t softBodyIndex = 0;
    
    while (offset < resultData.size() && softBodyIndex < softBodies.size()) {
        // Read size
        if (offset + sizeof(uint32_t) > resultData.size()) {
            std::cerr << "Invalid result data: incomplete size field" << std::endl;
            break;
        }
        
        uint32_t size;
        std::memcpy(&size, &resultData[offset], sizeof(uint32_t));
        offset += sizeof(uint32_t);
        
        // Read soft body data
        if (offset + size > resultData.size()) {
            std::cerr << "Invalid result data: incomplete soft body data" << std::endl;
            break;
        }
        
        std::vector<uint8_t> softBodyData(
            resultData.begin() + offset,
            resultData.begin() + offset + size);
        offset += size;
        
        // Deserialize into soft body
        StateSerializer::SerializationOptions options;
        options.enableCompression = true;
        
        PhysXSoftBody* softBody = softBodies[softBodyIndex];
        m_Impl->stateSerializer->DeserializeSoftBody(softBody, softBodyData, options);
        
        softBodyIndex++;
    }
    
    if (softBodyIndex != softBodies.size()) {
        std::cerr << "Warning: deserialized " << softBodyIndex 
                  << " soft bodies, expected " << softBodies.size() << std::endl;
    }
}

// Parse batch result message
void DistributedBatchManager::ParseBatchResultMessage(const NetworkManager::Message& msg) {
    const std::vector<uint8_t>& data = msg.data;
    size_t offset = 0;
    
    // Parse batch ID
    if (offset + sizeof(uint32_t) > data.size()) {
        std::cerr << "Invalid batch result: missing batch ID" << std::endl;
        return;
    }
    
    uint32_t batchId;
    std::memcpy(&batchId, &data[offset], sizeof(uint32_t));
    offset += sizeof(uint32_t);
    
    // Parse processing time
    if (offset + sizeof(uint64_t) > data.size()) {
        std::cerr << "Invalid batch result: missing processing time" << std::endl;
        return;
    }
    
    uint64_t processingTimeMs;
    std::memcpy(&processingTimeMs, &data[offset], sizeof(uint64_t));
    offset += sizeof(uint64_t);
    
    // Parse success flag
    if (offset + 1 > data.size()) {
        std::cerr << "Invalid batch result: missing success flag" << std::endl;
        return;
    }
    
    bool success = (data[offset] != 0);
    offset += 1;
    
    // Parse error message
    if (offset + sizeof(uint32_t) > data.size()) {
        std::cerr << "Invalid batch result: missing error length" << std::endl;
        return;
    }
    
    uint32_t errorLen;
    std::memcpy(&errorLen, &data[offset], sizeof(uint32_t));
    offset += sizeof(uint32_t);
    
    std::string errorMessage;
    if (errorLen > 0) {
        if (offset + errorLen > data.size()) {
            std::cerr << "Invalid batch result: incomplete error message" << std::endl;
            return;
        }
        
        errorMessage = std::string(
            reinterpret_cast<const char*>(&data[offset]),
            errorLen);
        offset += errorLen;
    }
    
    // Remaining data is the result
    std::vector<uint8_t> resultData(data.begin() + offset, data.end());
    
    // Update statistics
    {
        std::lock_guard<std::mutex> lock(m_Impl->statsMutex);
        
        // Update average processing time
        m_Impl->stats.avgProcessingTimeMs = 
            (m_Impl->stats.avgProcessingTimeMs * 0.9f) + 
            (processingTimeMs * 0.1f);
    }
    
    if (success) {
        HandleBatchResult(msg.sourceNode, batchId, resultData);
    } else {
        std::cerr << "Batch " << batchId << " failed on node " 
                  << msg.sourceNode << ": " << errorMessage << std::endl;
        
        // Mark batch as failed and potentially reassign
        auto it = m_Impl->batchAssignments.find(batchId);
        if (it != m_Impl->batchAssignments.end()) {
            it->second.isProcessing = false;
            it->second.assignedNode = -1;
            
            std::lock_guard<std::mutex> lock(m_Impl->statsMutex);
            m_Impl->stats.batchesFailed++;
        }
    }
}

// Update HandleMasterMessage to process batch results
void DistributedBatchManager::HandleMasterMessage(int nodeId, const NetworkManager::Message& msg) {
    switch (msg.type) {
        case NetworkManager::MessageType::BATCH_RESULT:
            ParseBatchResultMessage(msg);
            break;
            
        // ... existing cases ...
        
        default:
            break;
    }
}

// Result timeout monitoring
void DistributedBatchManager::MonitorResultTimeouts() {
    while (m_Impl->running) {
        auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        
        std::vector<uint32_t> timedOutBatches;
        
        // Check for timed out batches
        for (const auto& [batchId, assignment] : m_Impl->batchAssignments) {
            if (assignment.isProcessing) {
                uint64_t timeSinceAssignment = now - assignment.assignedTime;
                
                if (timeSinceAssignment > m_Impl->resultTimeoutMs) {
                    std::cerr << "Batch " << batchId << " timed out (assigned to node "
                              << assignment.assignedNode << ")" << std::endl;
                    
                    timedOutBatches.push_back(batchId);
                }
            }
        }
        
        // Reassign timed out batches
        for (uint32_t batchId : timedOutBatches) {
            auto& assignment = m_Impl->batchAssignments[batchId];
            
            std::cout << "Reassigning timed out batch " << batchId << std::endl;
            
            assignment.isProcessing = false;
            assignment.assignedNode = -1;
            
            // Select new node and reassign
            int newNode = SelectNodeForBatch(assignment.softBodies, assignment.priority);
            if (newNode >= 0) {
                AssignBatchToNode(batchId, newNode);
            }
            
            std::lock_guard<std::mutex> lock(m_Impl->statsMutex);
            m_Impl->stats.batchesFailed++;
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
}

void DistributedBatchManager::SetResultTimeout(uint32_t timeoutMs) {
    m_Impl->resultTimeoutMs = timeoutMs;
    
    std::cout << "Result timeout set to " << timeoutMs << "ms" << std::endl;
}

// Get batch processing statistics
struct BatchProcessingStats {
    uint32_t totalBatches;
    uint32_t completedBatches;
    uint32_t failedBatches;
    uint32_t pendingBatches;
    float avgProcessingTimeMs;
    float successRate;
};

BatchProcessingStats DistributedBatchManager::GetBatchProcessingStats() const {
    BatchProcessingStats stats;
    
    std::lock_guard<std::mutex> lock(m_Impl->statsMutex);
    
    stats.totalBatches = m_Impl->stats.totalBatches;
    stats.completedBatches = m_Impl->stats.batchesCompleted;
    stats.failedBatches = m_Impl->stats.batchesFailed;
    stats.pendingBatches = m_Impl->stats.batchesAssigned - m_Impl->stats.batchesCompleted;
    stats.avgProcessingTimeMs = m_Impl->stats.avgProcessingTimeMs;
    
    if (stats.totalBatches > 0) {
        stats.successRate = static_cast<float>(stats.completedBatches) / stats.totalBatches;
    } else {
        stats.successRate = 0.0f;
    }
    
    return stats;
}
