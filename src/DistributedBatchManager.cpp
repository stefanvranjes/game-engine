#include "DistributedBatchManager.h"
#include "DistributedBatchManagerImpl.h"
#include "PhysXSoftBody.h"
#include <iostream>
#include <algorithm>
#include <chrono>
#include <thread>

DistributedBatchManager::DistributedBatchManager() 
    : m_Impl(std::make_unique<Impl>()) {
    m_Impl->networkManager = std::make_unique<NetworkManager>();
    m_Impl->stateSerializer = std::make_unique<StateSerializer>();
}

DistributedBatchManager::~DistributedBatchManager() {
    Shutdown();
}

bool DistributedBatchManager::InitializeAsMaster(uint16_t port) {
    m_Impl->role = NodeRole::MASTER;
    
    if (!m_Impl->networkManager->StartMasterServer(port)) {
        std::cerr << "Failed to start master server on port " << port << std::endl;
        return false;
    }
    
    // Set up message callback
    m_Impl->networkManager->SetMessageCallback(
        [this](int nodeId, const NetworkManager::Message& msg) {
            HandleMasterMessage(nodeId, msg);
        }
    );
    
    // Start fault tolerance monitoring
    m_Impl->running = true;
    
    m_Impl->monitorThread = std::thread([this]() {
        MonitorWorkerHealth();
    });
    
    std::cout << "Initialized as MASTER on port " << port << std::endl;
    std::cout << "Fault tolerance enabled (heartbeat timeout: " 
              << m_Impl->heartbeatTimeoutMs << "ms)" << std::endl;
    
    return true;
}

bool DistributedBatchManager::InitializeAsWorker(const std::string& masterAddress, 
                                                 uint16_t masterPort) {
    m_Impl->role = NodeRole::WORKER;
    
    if (!m_Impl->networkManager->ConnectToMaster(masterAddress, masterPort)) {
        std::cerr << "Failed to connect to master at " << masterAddress 
                  << ":" << masterPort << std::endl;
        return false;
    }
    
    // Initialize local batch manager
    m_Impl->localBatchManager = std::make_unique<GpuBatchManager>();
    
    // Set up message callback
    m_Impl->networkManager->SetMessageCallback(
        [this](int nodeId, const NetworkManager::Message& msg) {
            HandleWorkerMessage(nodeId, msg);
        }
    );
    
    // Register with master
    if (!RegisterWithMaster()) {
        std::cerr << "Failed to register with master" << std::endl;
        return false;
    }
    
    // Start heartbeat thread
    m_Impl->running = true;
    
    m_Impl->heartbeatThread = std::thread([this]() {
        SendHeartbeats();
    });
    
    std::cout << "Initialized as WORKER, connected to master at " 
              << masterAddress << ":" << masterPort << std::endl;
    std::cout << "Heartbeat enabled (interval: " 
              << m_Impl->heartbeatIntervalMs << "ms)" << std::endl;
    
    return true;
}

bool DistributedBatchManager::InitializeAsStandalone() {
    m_Impl->role = NodeRole::STANDALONE;
    m_Impl->localBatchManager = std::make_unique<GpuBatchManager>();
    
    std::cout << "Initialized as STANDALONE (no distribution)" << std::endl;
    return true;
}

void DistributedBatchManager::Shutdown() {
    m_Impl->running = false;
    
    if (m_Impl->loadBalanceThread.joinable()) {
        m_Impl->loadBalanceThread.join();
    }
    
    if (m_Impl->networkManager) {
        m_Impl->networkManager->Shutdown();
    }
    
    std::cout << "DistributedBatchManager shutdown complete" << std::endl;
}

void DistributedBatchManager::AddSoftBody(PhysXSoftBody* softBody, uint32_t priority) {
    if (m_Impl->role == NodeRole::STANDALONE) {
        m_Impl->localBatchManager->AddSoftBody(softBody);
        return;
    }
    
    if (m_Impl->role != NodeRole::MASTER) {
        std::cerr << "Only master can add soft bodies" << std::endl;
        return;
    }
    
    // Create batch assignment
    uint32_t batchId = m_Impl->nextBatchId++;
    
    BatchAssignment assignment;
    assignment.batchId = batchId;
    assignment.softBodies.push_back(softBody);
    assignment.priority = priority;
    assignment.assignedTime = 0;
    assignment.isProcessing = false;
    assignment.assignedNode = -1;
    
    m_Impl->batchAssignments[batchId] = assignment;
    m_Impl->softBodyToBatch[softBody] = batchId;
    
    std::lock_guard<std::mutex> lock(m_Impl->statsMutex);
    m_Impl->stats.totalBatches++;
}

void DistributedBatchManager::RemoveSoftBody(PhysXSoftBody* softBody) {
    if (m_Impl->role == NodeRole::STANDALONE) {
        m_Impl->localBatchManager->RemoveSoftBody(softBody);
        return;
    }
    
    auto it = m_Impl->softBodyToBatch.find(softBody);
    if (it != m_Impl->softBodyToBatch.end()) {
        uint32_t batchId = it->second;
        m_Impl->batchAssignments.erase(batchId);
        m_Impl->softBodyToBatch.erase(it);
    }
}

void DistributedBatchManager::ProcessBatches(float deltaTime) {
    if (m_Impl->role == NodeRole::STANDALONE) {
        m_Impl->localBatchManager->BatchCopyData();
        m_Impl->localBatchManager->BatchApplyData();
        return;
    }
    
    if (m_Impl->role != NodeRole::MASTER) {
        return;  // Workers process assigned batches separately
    }
    
    // Assign unassigned batches
    for (auto& [batchId, assignment] : m_Impl->batchAssignments) {
        if (assignment.assignedNode == -1) {
            int nodeId = SelectNodeForBatch(assignment.softBodies, assignment.priority);
            if (nodeId >= 0) {
                AssignBatchToNode(batchId, nodeId);
            }
        }
    }
}

int DistributedBatchManager::SelectNodeForBatch(
    const std::vector<PhysXSoftBody*>& softBodies, 
    uint32_t priority) {
    
    if (m_Impl->workers.empty()) {
        return -1;
    }
    
    int selectedNode = -1;
    
    switch (m_Impl->loadBalancingStrategy) {
        case LoadBalancingStrategy::ROUND_ROBIN: {
            // Simple round-robin
            static int lastNode = -1;
            for (const auto& [nodeId, worker] : m_Impl->workers) {
                if (worker.isHealthy && nodeId > lastNode) {
                    selectedNode = nodeId;
                    lastNode = nodeId;
                    break;
                }
            }
            if (selectedNode == -1 && !m_Impl->workers.empty()) {
                selectedNode = m_Impl->workers.begin()->first;
                lastNode = selectedNode;
            }
            break;
        }
        
        case LoadBalancingStrategy::LEAST_LOADED: {
            // Select node with lowest PREDICTED load
            float minPredictedLoad = std::numeric_limits<float>::max();
            for (const auto& [nodeId, worker] : m_Impl->workers) {
                if (!worker.isHealthy) continue;
                
                // Predict load after assigning this batch
                float predictedLoad = PredictNodeLoad(nodeId, 1);
                
                if (predictedLoad < minPredictedLoad) {
                    minPredictedLoad = predictedLoad;
                    selectedNode = nodeId;
                }
            }
            break;
        }
        
        case LoadBalancingStrategy::CAPABILITY_BASED: {
            // Consider GPU count, memory, and current load
            float bestScore = -1.0f;
            for (const auto& [nodeId, worker] : m_Impl->workers) {
                if (!worker.isHealthy) continue;
                
                // Calculate capability score
                float gpuScore = worker.gpuCount * 10.0f;
                float memoryScore = (worker.availableMemoryMB / 1024.0f);
                float loadPenalty = PredictNodeLoad(nodeId, 1) * 20.0f;
                
                float score = gpuScore + memoryScore - loadPenalty;
                
                if (score > bestScore) {
                    bestScore = score;
                    selectedNode = nodeId;
                }
            }
            break;
        }
        
        case LoadBalancingStrategy::PRIORITY_AWARE: {
            // Consider priority, load, and load trend
            float bestScore = -1.0f;
            for (const auto& [nodeId, worker] : m_Impl->workers) {
                if (!worker.isHealthy) continue;
                
                // Get load trend
                float loadTrend = m_Impl->loadHistory[nodeId].GetTrend();
                float predictedLoad = PredictNodeLoad(nodeId, 1);
                
                // Higher priority batches prefer nodes with stable/decreasing load
                float priorityBonus = (priority / 10.0f);
                float trendPenalty = loadTrend > 0 ? loadTrend * 10.0f : 0.0f;
                float loadPenalty = predictedLoad * 10.0f;
                
                float score = priorityBonus - trendPenalty - loadPenalty;
                
                if (score > bestScore) {
                    bestScore = score;
                    selectedNode = nodeId;
                }
            }
            break;
        }
    }
    
    return selectedNode;
}

void DistributedBatchManager::AssignBatchToNode(uint32_t batchId, int nodeId) {
    auto it = m_Impl->batchAssignments.find(batchId);
    if (it == m_Impl->batchAssignments.end()) {
        return;
    }
    
    auto& assignment = it->second;
    assignment.assignedNode = nodeId;
    assignment.assignedTime = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    assignment.isProcessing = true;
    
    // Send batch assignment message
    NetworkManager::Message msg;
    msg.type = NetworkManager::MessageType::BATCH_ASSIGN;
    msg.targetNode = nodeId;
    
    // TODO: Serialize soft body data
    // msg.data = SerializeBatch(assignment.softBodies);
    
    m_Impl->networkManager->SendMessageToNode(nodeId, msg);
    
    std::lock_guard<std::mutex> lock(m_Impl->statsMutex);
    m_Impl->stats.batchesAssigned++;
    
    std::cout << "Assigned batch " << batchId << " to node " << nodeId << std::endl;
}

void DistributedBatchManager::SetLoadBalancingStrategy(LoadBalancingStrategy strategy) {
    m_Impl->loadBalancingStrategy = strategy;
    
    const char* strategyNames[] = {
        "ROUND_ROBIN", "LEAST_LOADED", "CAPABILITY_BASED", "PRIORITY_AWARE"
    };
    
    std::cout << "Load balancing strategy set to: " 
              << strategyNames[static_cast<int>(strategy)] << std::endl;
}

void DistributedBatchManager::EnableAutoLoadBalancing(bool enable, uint32_t checkIntervalMs) {
    m_Impl->autoLoadBalancing = enable;
    m_Impl->loadBalanceInterval = checkIntervalMs;
    
    if (enable && m_Impl->role == NodeRole::MASTER) {
        m_Impl->running = true;
        m_Impl->loadBalanceThread = std::thread([this]() {
            while (m_Impl->running) {
                CheckLoadBalance();
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(m_Impl->loadBalanceInterval));
            }
        });
        
        std::cout << "Auto load balancing enabled (interval: " 
                  << checkIntervalMs << "ms)" << std::endl;
    }
}

void DistributedBatchManager::CheckLoadBalance() {
    // Calculate load imbalance
    if (m_Impl->workers.size() < 2) {
        return;
    }
    
    float minLoad = std::numeric_limits<float>::max();
    float maxLoad = 0.0f;
    int minLoadNode = -1;
    int maxLoadNode = -1;
    
    // Update load history and find min/max
    for (auto& [nodeId, worker] : m_Impl->workers) {
        if (worker.isHealthy) {
            m_Impl->loadHistory[nodeId].AddSample(worker.currentLoad);
            
            float avgLoad = m_Impl->loadHistory[nodeId].GetAverage();
            
            if (avgLoad < minLoad) {
                minLoad = avgLoad;
                minLoadNode = nodeId;
            }
            if (avgLoad > maxLoad) {
                maxLoad = avgLoad;
                maxLoadNode = nodeId;
            }
        }
    }
    
    float imbalance = maxLoad - minLoad;
    
    // Check if imbalance exceeds threshold
    if (imbalance > m_Impl->loadImbalanceThreshold) {
        std::cout << "Load imbalance detected: " << (imbalance * 100) 
                  << "% (max: " << maxLoad << ", min: " << minLoad << ")" << std::endl;
        
        // Find batches to migrate from overloaded to underloaded node
        MigrateBatchesForBalance(maxLoadNode, minLoadNode);
    }
}

void DistributedBatchManager::MigrateBatchesForBalance(int sourceNode, int targetNode) {
    if (sourceNode < 0 || targetNode < 0) {
        return;
    }
    
    // Find batches assigned to source node
    std::vector<uint32_t> candidateBatches;
    
    for (const auto& [batchId, assignment] : m_Impl->batchAssignments) {
        if (assignment.assignedNode == sourceNode && !assignment.isProcessing) {
            candidateBatches.push_back(batchId);
        }
    }
    
    if (candidateBatches.empty()) {
        return;
    }
    
    // Sort by priority (migrate lower priority first)
    std::sort(candidateBatches.begin(), candidateBatches.end(),
        [this](uint32_t a, uint32_t b) {
            return m_Impl->batchAssignments[a].priority < 
                   m_Impl->batchAssignments[b].priority;
        });
    
    // Migrate lowest priority batch
    uint32_t batchToMigrate = candidateBatches[0];
    
    std::cout << "Migrating batch " << batchToMigrate 
              << " from node " << sourceNode 
              << " to node " << targetNode << std::endl;
    
    // Perform migration
    PerformBatchMigration(batchToMigrate, sourceNode, targetNode);
    
    std::lock_guard<std::mutex> lock(m_Impl->statsMutex);
    m_Impl->stats.totalMigrations++;
}

void DistributedBatchManager::PerformBatchMigration(uint32_t batchId, 
                                                     int sourceNode, 
                                                     int targetNode) {
    auto it = m_Impl->batchAssignments.find(batchId);
    if (it == m_Impl->batchAssignments.end()) {
        return;
    }
    
    auto& assignment = it->second;
    
    // Track migration
    std::lock_guard<std::mutex> lock(m_Impl->migrationMutex);
    
    Impl::MigrationRecord record;
    record.batchId = batchId;
    record.sourceNode = sourceNode;
    record.targetNode = targetNode;
    record.startTime = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    record.completed = false;
    
    m_Impl->activeMigrations.push_back(record);
    
    // Send migration message to source node (to stop processing)
    NetworkManager::Message stopMsg;
    stopMsg.type = NetworkManager::MessageType::MIGRATION;
    stopMsg.targetNode = sourceNode;
    // TODO: Add batch ID to message data
    
    m_Impl->networkManager->SendMessageToNode(sourceNode, stopMsg);
    
    // Send migration message to target node (to start processing)
    NetworkManager::Message startMsg;
    startMsg.type = NetworkManager::MessageType::MIGRATION;
    startMsg.targetNode = targetNode;
    // TODO: Serialize soft body state and add to message data
    
    m_Impl->networkManager->SendMessageToNode(targetNode, startMsg);
    
    // Update assignment
    assignment.assignedNode = targetNode;
    assignment.assignedTime = record.startTime;
    
    // Update worker stats
    if (m_Impl->workers.count(sourceNode)) {
        m_Impl->workers[sourceNode].assignedBatches--;
    }
    if (m_Impl->workers.count(targetNode)) {
        m_Impl->workers[targetNode].assignedBatches++;
    }
}

void DistributedBatchManager::UpdateWorkerLoad(int nodeId, float load) {
    auto it = m_Impl->workers.find(nodeId);
    if (it != m_Impl->workers.end()) {
        it->second.currentLoad = load;
        it->second.lastHeartbeat = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        
        // Update load history
        m_Impl->loadHistory[nodeId].AddSample(load);
    }
}

float DistributedBatchManager::PredictNodeLoad(int nodeId, size_t additionalBatches) {
    auto it = m_Impl->workers.find(nodeId);
    if (it == m_Impl->workers.end()) {
        return 1.0f;  // Assume full load if node not found
    }
    
    const auto& worker = it->second;
    
    // Get current load trend
    float currentLoad = worker.currentLoad;
    float loadTrend = m_Impl->loadHistory[nodeId].GetTrend();
    
    // Estimate load per batch
    float loadPerBatch = worker.assignedBatches > 0 ? 
        currentLoad / worker.assignedBatches : 0.1f;
    
    // Predict future load
    float predictedLoad = currentLoad + (loadPerBatch * additionalBatches) + loadTrend;
    
    return std::min(1.0f, std::max(0.0f, predictedLoad));
}

DistributedBatchManager::DistributedStats DistributedBatchManager::GetStatistics() const {
    std::lock_guard<std::mutex> lock(m_Impl->statsMutex);
    return m_Impl->stats;
}

std::vector<DistributedBatchManager::WorkerNode> DistributedBatchManager::GetWorkerNodes() const {
    std::vector<WorkerNode> nodes;
    for (const auto& [nodeId, worker] : m_Impl->workers) {
        nodes.push_back(worker);
    }
    return nodes;
}

DistributedBatchManager::NodeRole DistributedBatchManager::GetNodeRole() const {
    return m_Impl->role;
}

// Message handlers (to be implemented)
void DistributedBatchManager::HandleMasterMessage(int nodeId, const NetworkManager::Message& msg) {
    // Handle messages received by master
    switch (msg.type) {
        case NetworkManager::MessageType::NODE_REGISTER:
            // Register new worker
            break;
        case NetworkManager::MessageType::BATCH_RESULT:
            // Handle batch completion
            break;
        case NetworkManager::MessageType::LOAD_REPORT:
            // Update worker load
            break;
        default:
            break;
    }
}

void DistributedBatchManager::HandleWorkerMessage(int nodeId, const NetworkManager::Message& msg) {
    // Handle messages received by worker
    switch (msg.type) {
        case NetworkManager::MessageType::BATCH_ASSIGN:
            // Process assigned batch
            break;
        case NetworkManager::MessageType::MIGRATION:
            // Handle soft body migration
            break;
        default:
            break;
    }
}
