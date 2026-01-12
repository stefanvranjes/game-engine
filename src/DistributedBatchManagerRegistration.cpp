#include "DistributedBatchManager.h"
#include "DistributedBatchManagerImpl.h"
#include <iostream>

#ifdef HAS_CUDA_TOOLKIT
#include <cuda_runtime.h>
#endif

// Worker registration
bool DistributedBatchManager::RegisterWithMaster() {
    if (m_Impl->role != NodeRole::WORKER) {
        std::cerr << "RegisterWithMaster called on non-worker node" << std::endl;
        return false;
    }
    
    // Gather node capabilities
    NetworkManager::NodeInfo nodeInfo;
    nodeInfo.nodeId = m_Impl->networkManager->GetLocalNodeId();
    
    // Get GPU count
#ifdef HAS_CUDA_TOOLKIT
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    nodeInfo.gpuCount = static_cast<size_t>(deviceCount);
    
    // Get total GPU memory (sum of all GPUs)
    nodeInfo.totalMemoryMB = 0;
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        nodeInfo.totalMemoryMB += prop.totalGlobalMem / (1024 * 1024);
    }
#else
    nodeInfo.gpuCount = 0;
    nodeInfo.totalMemoryMB = 0;
#endif
    
    // Get system info
    nodeInfo.address = "localhost";  // TODO: Get actual IP
    nodeInfo.port = 0;  // Worker doesn't listen
    
    // Create registration message
    NetworkManager::Message msg;
    msg.type = NetworkManager::MessageType::NODE_REGISTER;
    msg.sourceNode = nodeInfo.nodeId;
    msg.targetNode = 0;  // Master is always node 0
    
    // Pack node info into message data
    std::vector<uint8_t> data;
    
    // GPU count
    data.insert(data.end(),
               reinterpret_cast<const uint8_t*>(&nodeInfo.gpuCount),
               reinterpret_cast<const uint8_t*>(&nodeInfo.gpuCount) + sizeof(size_t));
    
    // Total memory
    data.insert(data.end(),
               reinterpret_cast<const uint8_t*>(&nodeInfo.totalMemoryMB),
               reinterpret_cast<const uint8_t*>(&nodeInfo.totalMemoryMB) + sizeof(uint64_t));
    
    // Address length and data
    uint32_t addrLen = static_cast<uint32_t>(nodeInfo.address.size());
    data.insert(data.end(),
               reinterpret_cast<const uint8_t*>(&addrLen),
               reinterpret_cast<const uint8_t*>(&addrLen) + sizeof(uint32_t));
    data.insert(data.end(), nodeInfo.address.begin(), nodeInfo.address.end());
    
    // Port
    data.insert(data.end(),
               reinterpret_cast<const uint8_t*>(&nodeInfo.port),
               reinterpret_cast<const uint8_t*>(&nodeInfo.port) + sizeof(uint16_t));
    
    msg.data = data;
    
    // Send registration (reliable)
    m_Impl->networkManager->SendMessageToNode(0, msg, true);
    
    std::cout << "Sent registration to master (GPUs: " << nodeInfo.gpuCount
              << ", Memory: " << nodeInfo.totalMemoryMB << " MB)" << std::endl;
    
    return true;
}

// Master-side: Handle worker registration
void DistributedBatchManager::HandleNodeRegistration(int nodeId, const NetworkManager::Message& msg) {
    if (m_Impl->role != NodeRole::MASTER) {
        return;
    }
    
    // Parse node info from message
    const std::vector<uint8_t>& data = msg.data;
    size_t offset = 0;
    
    NetworkManager::NodeInfo nodeInfo;
    nodeInfo.nodeId = nodeId;
    
    // GPU count
    if (offset + sizeof(size_t) > data.size()) {
        std::cerr << "Invalid registration: missing GPU count" << std::endl;
        return;
    }
    std::memcpy(&nodeInfo.gpuCount, &data[offset], sizeof(size_t));
    offset += sizeof(size_t);
    
    // Total memory
    if (offset + sizeof(uint64_t) > data.size()) {
        std::cerr << "Invalid registration: missing memory" << std::endl;
        return;
    }
    std::memcpy(&nodeInfo.totalMemoryMB, &data[offset], sizeof(uint64_t));
    offset += sizeof(uint64_t);
    
    // Address
    if (offset + sizeof(uint32_t) > data.size()) {
        std::cerr << "Invalid registration: missing address length" << std::endl;
        return;
    }
    uint32_t addrLen;
    std::memcpy(&addrLen, &data[offset], sizeof(uint32_t));
    offset += sizeof(uint32_t);
    
    if (offset + addrLen > data.size()) {
        std::cerr << "Invalid registration: incomplete address" << std::endl;
        return;
    }
    nodeInfo.address = std::string(
        reinterpret_cast<const char*>(&data[offset]),
        addrLen);
    offset += addrLen;
    
    // Port
    if (offset + sizeof(uint16_t) > data.size()) {
        std::cerr << "Invalid registration: missing port" << std::endl;
        return;
    }
    std::memcpy(&nodeInfo.port, &data[offset], sizeof(uint16_t));
    offset += sizeof(uint16_t);
    
    // Register the worker
    RegisterWorker(nodeId, nodeInfo);
    
    // Send acknowledgment
    NetworkManager::Message ack;
    ack.type = NetworkManager::MessageType::ACK;
    ack.sourceNode = m_Impl->networkManager->GetLocalNodeId();
    ack.targetNode = nodeId;
    
    m_Impl->networkManager->SendMessageToNode(nodeId, ack, true);
}

// Worker unregistration
void DistributedBatchManager::UnregisterFromMaster() {
    if (m_Impl->role != NodeRole::WORKER) {
        return;
    }
    
    NetworkManager::Message msg;
    msg.type = NetworkManager::MessageType::NODE_UNREGISTER;
    msg.sourceNode = m_Impl->networkManager->GetLocalNodeId();
    msg.targetNode = 0;  // Master
    
    m_Impl->networkManager->SendMessageToNode(0, msg, true);
    
    std::cout << "Sent unregistration to master" << std::endl;
}

void DistributedBatchManager::HandleNodeUnregistration(int nodeId, const NetworkManager::Message& msg) {
    if (m_Impl->role != NodeRole::MASTER) {
        return;
    }
    
    std::cout << "Worker " << nodeId << " unregistered" << std::endl;
    
    // Reassign batches from this node
    ReassignBatches(nodeId);
    
    // Remove from workers list
    auto it = m_Impl->workers.find(nodeId);
    if (it != m_Impl->workers.end()) {
        m_Impl->workers.erase(it);
        
        std::lock_guard<std::mutex> lock(m_Impl->statsMutex);
        m_Impl->stats.totalWorkers--;
        m_Impl->stats.activeWorkers--;
    }
    
    // Disconnect
    m_Impl->networkManager->DisconnectNode(nodeId);
}

// Enhanced heartbeat with load reporting
void DistributedBatchManager::SendHeartbeats() {
    while (m_Impl->running) {
        // Send heartbeat to master
        NetworkManager::Message heartbeat;
        heartbeat.type = NetworkManager::MessageType::HEARTBEAT;
        heartbeat.sourceNode = m_Impl->networkManager->GetLocalNodeId();
        heartbeat.targetNode = 0;  // Master is always node 0
        
        // Pack current load info
        std::vector<uint8_t> data;
        
        // Calculate current load (0.0 - 1.0)
        float currentLoad = 0.0f;
        if (m_Impl->localBatchManager) {
            auto stats = m_Impl->localBatchManager->GetStatistics();
            // Load based on active batches and GPU utilization
            currentLoad = std::min(1.0f, stats.activeBatches / 10.0f);
        }
        
        data.insert(data.end(),
                   reinterpret_cast<const uint8_t*>(&currentLoad),
                   reinterpret_cast<const uint8_t*>(&currentLoad) + sizeof(float));
        
        // Assigned batch count
        uint32_t assignedBatches = static_cast<uint32_t>(m_Impl->assignedBatches.size());
        data.insert(data.end(),
                   reinterpret_cast<const uint8_t*>(&assignedBatches),
                   reinterpret_cast<const uint8_t*>(&assignedBatches) + sizeof(uint32_t));
        
        heartbeat.data = data;
        
        m_Impl->networkManager->SendMessageToNode(0, heartbeat, false);  // Unreliable
        
        std::this_thread::sleep_for(
            std::chrono::milliseconds(m_Impl->heartbeatIntervalMs));
    }
}

// Enhanced heartbeat handler with load extraction
void DistributedBatchManager::HandleHeartbeat(int nodeId, const NetworkManager::Message& msg) {
    auto it = m_Impl->workers.find(nodeId);
    if (it != m_Impl->workers.end()) {
        auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        
        it->second.lastHeartbeat = now;
        
        // If node was previously unhealthy, mark as recovered
        if (!it->second.isHealthy) {
            std::cout << "Worker node " << nodeId << " recovered" << std::endl;
            it->second.isHealthy = true;
            
            std::lock_guard<std::mutex> lock(m_Impl->statsMutex);
            m_Impl->stats.activeWorkers++;
        }
        
        // Extract load info from heartbeat data
        if (msg.data.size() >= sizeof(float)) {
            float currentLoad;
            std::memcpy(&currentLoad, &msg.data[0], sizeof(float));
            
            it->second.currentLoad = currentLoad;
            UpdateWorkerLoad(nodeId, currentLoad);
        }
        
        // Extract assigned batch count
        if (msg.data.size() >= sizeof(float) + sizeof(uint32_t)) {
            uint32_t assignedBatches;
            std::memcpy(&assignedBatches, &msg.data[sizeof(float)], sizeof(uint32_t));
            
            it->second.assignedBatches = assignedBatches;
        }
    }
}

// Update HandleMasterMessage to include registration
void DistributedBatchManager::HandleMasterMessage(int nodeId, const NetworkManager::Message& msg) {
    switch (msg.type) {
        case NetworkManager::MessageType::NODE_REGISTER:
            HandleNodeRegistration(nodeId, msg);
            break;
            
        case NetworkManager::MessageType::NODE_UNREGISTER:
            HandleNodeUnregistration(nodeId, msg);
            break;
            
        case NetworkManager::MessageType::HEARTBEAT:
            HandleHeartbeat(nodeId, msg);
            break;
            
        case NetworkManager::MessageType::BATCH_RESULT:
            ParseBatchResultMessage(msg);
            break;
            
        case NetworkManager::MessageType::VOTE_REQUEST:
            HandleVoteRequest(nodeId, msg);
            break;
            
        case NetworkManager::MessageType::VOTE_RESPONSE:
            HandleVoteResponse(nodeId, msg);
            break;
            
        case NetworkManager::MessageType::STATE_SYNC_FULL:
            HandleStateSyncFull(nodeId, msg);
            break;
            
        case NetworkManager::MessageType::STATE_SYNC_DELTA:
            HandleStateSyncDelta(nodeId, msg);
            break;
            
        case NetworkManager::MessageType::LOAD_REPORT:
            // Handled in heartbeat
            break;
            
        default:
            std::cerr << "Unknown message type received: " 
                      << static_cast<int>(msg.type) << std::endl;
            break;
    }
}

// Shutdown with unregistration
void DistributedBatchManager::Shutdown() {
    std::cout << "Shutting down distributed batch manager..." << std::endl;
    
    // Unregister if worker
    if (m_Impl->role == NodeRole::WORKER) {
        UnregisterFromMaster();
    }
    
    // Stop all threads
    m_Impl->running = false;
    
    // Wait for threads to finish
    if (m_Impl->heartbeatThread.joinable()) {
        m_Impl->heartbeatThread.join();
    }
    if (m_Impl->monitorThread.joinable()) {
        m_Impl->monitorThread.join();
    }
    if (m_Impl->loadBalanceThread.joinable()) {
        m_Impl->loadBalanceThread.join();
    }
    if (m_Impl->syncThread.joinable()) {
        m_Impl->syncThread.join();
    }
    if (m_Impl->electionThread.joinable()) {
        m_Impl->electionThread.join();
    }
    
    // Shutdown network
    if (m_Impl->networkManager) {
        m_Impl->networkManager->Shutdown();
    }
    
    std::cout << "Shutdown complete" << std::endl;
}
