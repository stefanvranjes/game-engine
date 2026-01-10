// Hierarchical Distribution Implementation

#include "DistributedBatchManager.h"
#include <iostream>

// Add to Impl structure
struct DistributedBatchManager::Impl {
    // ... existing members ...
    
    // Hierarchical distribution
    HierarchyConfig hierarchyConfig;
    std::unordered_map<std::string, int> regionalMasters;  // region -> node ID
    std::unordered_map<std::string, std::vector<int>> regionalWorkers;  // region -> workers
    std::unordered_map<int, std::string> nodeRegions;  // node ID -> region
};

// Global Master Initialization
bool DistributedBatchManager::InitializeAsGlobalMaster(uint16_t port) {
    m_Impl->hierarchyConfig.role = NodeRole::GLOBAL_MASTER;
    m_Impl->hierarchyConfig.tier = 0;
    m_Impl->hierarchyConfig.parentNodeId = -1;
    m_Impl->hierarchyConfig.region = "global";
    
    // Initialize as master
    if (!InitializeAsMaster(port)) {
        return false;
    }
    
    std::cout << "Initialized as GLOBAL MASTER (hierarchical mode)" << std::endl;
    std::cout << "Ready to accept regional masters" << std::endl;
    
    return true;
}

// Regional Master Initialization
bool DistributedBatchManager::InitializeAsRegionalMaster(
    const std::string& globalMasterAddress,
    uint16_t globalMasterPort,
    uint16_t localPort,
    const std::string& region) {
    
    m_Impl->hierarchyConfig.role = NodeRole::REGIONAL_MASTER;
    m_Impl->hierarchyConfig.tier = 1;
    m_Impl->hierarchyConfig.region = region;
    
    // Connect to global master as a worker
    if (!m_Impl->networkManager->ConnectToMaster(globalMasterAddress, globalMasterPort)) {
        std::cerr << "Failed to connect to global master" << std::endl;
        return false;
    }
    
    // Also start server for local workers
    if (!m_Impl->networkManager->StartMasterServer(localPort)) {
        std::cerr << "Failed to start regional master server" << std::endl;
        return false;
    }
    
    // Register with global master
    m_Impl->hierarchyConfig.parentNodeId = 0;  // Global master is always 0
    
    // Send registration
    NetworkManager::Message regMsg;
    regMsg.type = NetworkManager::MessageType::NODE_REGISTER;
    regMsg.sourceNode = m_Impl->networkManager->GetLocalNodeId();
    regMsg.targetNode = 0;
    
    // TODO: Include region info in message data
    
    m_Impl->networkManager->SendMessage(0, regMsg);
    
    // Set up message callbacks
    m_Impl->networkManager->SetMessageCallback(
        [this](int nodeId, const NetworkManager::Message& msg) {
            HandleRegionalMasterMessage(nodeId, msg);
        }
    );
    
    // Start threads
    m_Impl->running = true;
    
    m_Impl->heartbeatThread = std::thread([this]() {
        SendHeartbeats();
    });
    
    m_Impl->monitorThread = std::thread([this]() {
        MonitorWorkerHealth();
    });
    
    std::cout << "Initialized as REGIONAL MASTER for region: " << region << std::endl;
    std::cout << "Connected to global master at " << globalMasterAddress 
              << ":" << globalMasterPort << std::endl;
    std::cout << "Listening for workers on port " << localPort << std::endl;
    
    return true;
}

// Worker in Region Initialization
bool DistributedBatchManager::InitializeAsWorkerInRegion(
    const std::string& regionalMasterAddress,
    uint16_t regionalMasterPort) {
    
    m_Impl->hierarchyConfig.role = NodeRole::WORKER;
    m_Impl->hierarchyConfig.tier = 2;
    
    // Initialize as regular worker
    if (!InitializeAsWorker(regionalMasterAddress, regionalMasterPort)) {
        return false;
    }
    
    std::cout << "Initialized as WORKER in hierarchical mode" << std::endl;
    std::cout << "Connected to regional master at " << regionalMasterAddress 
              << ":" << regionalMasterPort << std::endl;
    
    return true;
}

// Get Hierarchy Config
const DistributedBatchManager::HierarchyConfig& 
DistributedBatchManager::GetHierarchyConfig() const {
    return m_Impl->hierarchyConfig;
}

// Register Regional Master
void DistributedBatchManager::RegisterRegionalMaster(int nodeId, const std::string& region) {
    if (m_Impl->hierarchyConfig.role != NodeRole::GLOBAL_MASTER) {
        std::cerr << "Only global master can register regional masters" << std::endl;
        return;
    }
    
    m_Impl->regionalMasters[region] = nodeId;
    m_Impl->nodeRegions[nodeId] = region;
    m_Impl->hierarchyConfig.childNodes.push_back(nodeId);
    
    std::cout << "Registered regional master for region '" << region 
              << "' (node " << nodeId << ")" << std::endl;
}

// Select Region for Batch
std::string DistributedBatchManager::SelectRegionForBatch(
    const std::vector<PhysXSoftBody*>& softBodies) {
    
    if (m_Impl->regionalMasters.empty()) {
        return ""; // No regions available
    }
    
    // Calculate region scores
    std::unordered_map<std::string, float> regionScores;
    
    for (const auto& [region, masterId] : m_Impl->regionalMasters) {
        float score = 0.0f;
        
        // Factor 1: Available capacity (based on worker count and load)
        auto workersIt = m_Impl->regionalWorkers.find(region);
        if (workersIt != m_Impl->regionalWorkers.end()) {
            size_t workerCount = workersIt->second.size();
            
            // More workers = higher score
            score += (workerCount / 10.0f) * 0.5f;
            
            // Calculate average load
            float avgLoad = 0.0f;
            for (int workerId : workersIt->second) {
                auto workerIt = m_Impl->workers.find(workerId);
                if (workerIt != m_Impl->workers.end()) {
                    avgLoad += workerIt->second.currentLoad;
                }
            }
            if (workerCount > 0) {
                avgLoad /= workerCount;
            }
            
            // Lower load = higher score
            score += (1.0f - avgLoad) * 0.5f;
        }
        
        regionScores[region] = score;
    }
    
    // Select region with highest score
    std::string bestRegion;
    float bestScore = -1.0f;
    
    for (const auto& [region, score] : regionScores) {
        if (score > bestScore) {
            bestScore = score;
            bestRegion = region;
        }
    }
    
    return bestRegion;
}

// Assign Batch to Region
void DistributedBatchManager::AssignBatchToRegion(uint32_t batchId, const std::string& region) {
    if (m_Impl->hierarchyConfig.role != NodeRole::GLOBAL_MASTER) {
        std::cerr << "Only global master can assign batches to regions" << std::endl;
        return;
    }
    
    // Find regional master for this region
    auto it = m_Impl->regionalMasters.find(region);
    if (it == m_Impl->regionalMasters.end()) {
        std::cerr << "No regional master for region: " << region << std::endl;
        return;
    }
    
    int regionalMasterId = it->second;
    
    std::cout << "Assigning batch " << batchId << " to region '" << region 
              << "' (regional master " << regionalMasterId << ")" << std::endl;
    
    // Send batch assignment to regional master
    // Regional master will distribute to its workers
    AssignBatchToNode(batchId, regionalMasterId);
}

// Broadcast to All Regions
void DistributedBatchManager::BroadcastToAllRegions(const NetworkManager::Message& msg) {
    if (m_Impl->hierarchyConfig.role != NodeRole::GLOBAL_MASTER) {
        std::cerr << "Only global master can broadcast to all regions" << std::endl;
        return;
    }
    
    std::cout << "Broadcasting message to " << m_Impl->regionalMasters.size() 
              << " regions" << std::endl;
    
    for (const auto& [region, masterId] : m_Impl->regionalMasters) {
        NetworkManager::Message regionalMsg = msg;
        regionalMsg.targetNode = masterId;
        
        m_Impl->networkManager->SendMessage(masterId, regionalMsg);
    }
}

// Regional Master Message Handler
void DistributedBatchManager::HandleRegionalMasterMessage(int nodeId, 
                                                           const NetworkManager::Message& msg) {
    // Regional master receives messages from both global master (parent) and workers (children)
    
    switch (msg.type) {
        case NetworkManager::MessageType::BATCH_ASSIGN:
            // Received from global master - distribute to local workers
            {
                std::cout << "Regional master received batch assignment from global master" 
                          << std::endl;
                
                // Select best local worker
                // TODO: Extract batch info from message
                // For now, use standard assignment logic
                HandleMasterMessage(nodeId, msg);
            }
            break;
            
        case NetworkManager::MessageType::BATCH_RESULT:
            // Received from local worker - forward to global master
            {
                std::cout << "Regional master forwarding batch result to global master" 
                          << std::endl;
                
                NetworkManager::Message forwardMsg = msg;
                forwardMsg.targetNode = m_Impl->hierarchyConfig.parentNodeId;
                
                m_Impl->networkManager->SendMessage(
                    m_Impl->hierarchyConfig.parentNodeId, forwardMsg);
            }
            break;
            
        case NetworkManager::MessageType::NODE_REGISTER:
            // Local worker registering
            {
                std::cout << "Regional master: worker " << nodeId << " registered" 
                          << std::endl;
                
                // Add to regional workers
                m_Impl->regionalWorkers[m_Impl->hierarchyConfig.region].push_back(nodeId);
                
                // Handle as normal worker registration
                HandleNodeRegistration(nodeId, msg);
            }
            break;
            
        default:
            // Handle other messages normally
            HandleMasterMessage(nodeId, msg);
            break;
    }
}

// Update ProcessBatches for hierarchical mode
void DistributedBatchManager::ProcessBatches(float deltaTime) {
    if (m_Impl->hierarchyConfig.role == NodeRole::GLOBAL_MASTER) {
        // Global master: assign batches to regions
        
        for (auto& [batchId, assignment] : m_Impl->batchAssignments) {
            if (assignment.assignedNode < 0 && !assignment.isProcessing) {
                // Select region for this batch
                std::string region = SelectRegionForBatch(assignment.softBodies);
                
                if (!region.empty()) {
                    AssignBatchToRegion(batchId, region);
                }
            }
        }
    } else if (m_Impl->hierarchyConfig.role == NodeRole::REGIONAL_MASTER) {
        // Regional master: assign batches to local workers
        
        for (auto& [batchId, assignment] : m_Impl->batchAssignments) {
            if (assignment.assignedNode < 0 && !assignment.isProcessing) {
                // Select best local worker
                int nodeId = SelectNodeForBatch(assignment.softBodies, assignment.priority);
                
                if (nodeId >= 0) {
                    AssignBatchToNode(batchId, nodeId);
                }
            }
        }
    } else {
        // Worker: process assigned batches locally
        // (handled by existing worker logic)
    }
}
