// Master Failover Implementation for DistributedBatchManager
// Implements Raft-inspired leader election

#include "DistributedBatchManager.h"
#include <random>

// Add to Impl structure
struct DistributedBatchManager::Impl {
    // ... existing members ...
    
    // Master failover
    bool masterFailoverEnabled = false;
    int currentMasterId = 0;
    ElectionState electionState;
    std::thread electionThread;
    std::mt19937 rng;
    
    uint32_t GetRandomElectionTimeout() {
        // Random timeout between 150-300ms (Raft recommendation)
        std::uniform_int_distribution<uint32_t> dist(150, 300);
        return dist(rng);
    }
};

void DistributedBatchManager::EnableMasterFailover(bool enable) {
    m_Impl->masterFailoverEnabled = enable;
    
    if (enable && m_Impl->role == NodeRole::WORKER) {
        // Start election monitoring
        m_Impl->electionThread = std::thread([this]() {
            MonitorMasterHealth();
        });
        
        std::cout << "Master failover enabled" << std::endl;
    }
}

void DistributedBatchManager::MonitorMasterHealth() {
    // Workers monitor master and trigger election if master fails
    
    while (m_Impl->running) {
        auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        
        // Check if we've heard from master recently
        auto masterIt = m_Impl->workers.find(m_Impl->currentMasterId);
        
        bool masterFailed = false;
        
        if (masterIt != m_Impl->workers.end()) {
            uint64_t timeSinceHeartbeat = now - masterIt->second.lastHeartbeat;
            
            if (timeSinceHeartbeat > m_Impl->heartbeatTimeoutMs) {
                std::cerr << "Master node " << m_Impl->currentMasterId 
                          << " failed! Starting election..." << std::endl;
                masterFailed = true;
            }
        } else if (m_Impl->currentMasterId != 0) {
            // Master not in workers list
            masterFailed = true;
        }
        
        if (masterFailed) {
            StartElection();
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
}

void DistributedBatchManager::StartElection() {
    std::cout << "Starting leader election..." << std::endl;
    
    // Transition to CANDIDATE
    m_Impl->role = NodeRole::CANDIDATE;
    m_Impl->electionState.currentTerm++;
    m_Impl->electionState.votedFor = m_Impl->networkManager->GetLocalNodeId();
    m_Impl->electionState.voteCount = 1;  // Vote for self
    m_Impl->electionState.electionInProgress = true;
    
    // Set random election timeout
    auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    m_Impl->electionState.electionTimeout = now + m_Impl->GetRandomElectionTimeout();
    
    // Request votes from all other nodes
    RequestVotes();
    
    // Wait for election to complete
    WaitForElectionResult();
}

void DistributedBatchManager::RequestVotes() {
    int localNodeId = m_Impl->networkManager->GetLocalNodeId();
    
    // Send vote request to all known nodes
    for (const auto& [nodeId, worker] : m_Impl->workers) {
        if (nodeId == localNodeId) continue;
        
        NetworkManager::Message voteRequest;
        voteRequest.type = NetworkManager::MessageType::VOTE_REQUEST;
        voteRequest.sourceNode = localNodeId;
        voteRequest.targetNode = nodeId;
        
        // TODO: Add term and candidate info to message data
        
        m_Impl->networkManager->SendMessage(nodeId, voteRequest);
    }
    
    std::cout << "Sent vote requests for term " 
              << m_Impl->electionState.currentTerm << std::endl;
}

void DistributedBatchManager::WaitForElectionResult() {
    auto startTime = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    
    while (m_Impl->electionState.electionInProgress) {
        auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        
        // Check if we won the election
        size_t totalNodes = m_Impl->workers.size() + 1;  // +1 for self
        size_t majority = (totalNodes / 2) + 1;
        
        if (m_Impl->electionState.voteCount >= majority) {
            // Won election!
            BecomeLeader();
            return;
        }
        
        // Check for timeout
        if (now >= m_Impl->electionState.electionTimeout) {
            std::cout << "Election timeout, starting new election..." << std::endl;
            StartElection();  // Restart election
            return;
        }
        
        // Check if another node became leader
        if (m_Impl->role == NodeRole::WORKER) {
            // Another node won
            std::cout << "Another node became leader" << std::endl;
            m_Impl->electionState.electionInProgress = false;
            return;
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

void DistributedBatchManager::BecomeLeader() {
    std::cout << "WON ELECTION! Becoming master node..." << std::endl;
    
    m_Impl->role = NodeRole::MASTER;
    m_Impl->currentMasterId = m_Impl->networkManager->GetLocalNodeId();
    m_Impl->electionState.electionInProgress = false;
    
    // Announce leadership
    AnnounceLeadership();
    
    // Reconstruct master state
    ReconstructMasterState();
    
    // Start master threads
    StartMasterThreads();
    
    std::cout << "Successfully transitioned to MASTER role" << std::endl;
}

void DistributedBatchManager::AnnounceLeadership() {
    int localNodeId = m_Impl->networkManager->GetLocalNodeId();
    
    // Send leadership announcement to all nodes
    for (const auto& [nodeId, worker] : m_Impl->workers) {
        if (nodeId == localNodeId) continue;
        
        NetworkManager::Message announcement;
        announcement.type = NetworkManager::MessageType::LEADER_ANNOUNCEMENT;
        announcement.sourceNode = localNodeId;
        announcement.targetNode = nodeId;
        
        m_Impl->networkManager->SendMessage(nodeId, announcement);
    }
    
    std::cout << "Announced leadership to all nodes" << std::endl;
}

void DistributedBatchManager::ReconstructMasterState() {
    std::cout << "Reconstructing master state..." << std::endl;
    
    // Request state from all workers
    for (const auto& [nodeId, worker] : m_Impl->workers) {
        NetworkManager::Message stateRequest;
        stateRequest.type = NetworkManager::MessageType::STATE_REQUEST;
        stateRequest.sourceNode = m_Impl->networkManager->GetLocalNodeId();
        stateRequest.targetNode = nodeId;
        
        m_Impl->networkManager->SendMessage(nodeId, stateRequest);
    }
    
    // Wait for responses and rebuild batch assignments
    // TODO: Implement state collection and reconstruction
    
    std::cout << "Master state reconstructed" << std::endl;
}

void DistributedBatchManager::StartMasterThreads() {
    // Start monitoring thread
    if (!m_Impl->monitorThread.joinable()) {
        m_Impl->monitorThread = std::thread([this]() {
            MonitorWorkerHealth();
        });
    }
    
    // Start load balancing thread if enabled
    if (m_Impl->autoLoadBalancing && !m_Impl->loadBalanceThread.joinable()) {
        m_Impl->loadBalanceThread = std::thread([this]() {
            while (m_Impl->running) {
                CheckLoadBalance();
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(m_Impl->loadBalanceInterval));
            }
        });
    }
}

void DistributedBatchManager::HandleVoteRequest(int nodeId, const NetworkManager::Message& msg) {
    // Handle vote request from candidate
    
    // TODO: Extract term and candidate info from message
    uint32_t candidateTerm = 1;  // Placeholder
    
    bool grantVote = false;
    
    // Grant vote if:
    // 1. Haven't voted in this term, OR
    // 2. Already voted for this candidate
    if (m_Impl->electionState.currentTerm < candidateTerm) {
        // New term, reset vote
        m_Impl->electionState.currentTerm = candidateTerm;
        m_Impl->electionState.votedFor = -1;
    }
    
    if (m_Impl->electionState.votedFor == -1 || 
        m_Impl->electionState.votedFor == nodeId) {
        grantVote = true;
        m_Impl->electionState.votedFor = nodeId;
        
        std::cout << "Granted vote to node " << nodeId 
                  << " for term " << candidateTerm << std::endl;
    }
    
    // Send vote response
    NetworkManager::Message voteResponse;
    voteResponse.type = NetworkManager::MessageType::VOTE_RESPONSE;
    voteResponse.sourceNode = m_Impl->networkManager->GetLocalNodeId();
    voteResponse.targetNode = nodeId;
    
    // TODO: Add vote granted flag to message data
    
    m_Impl->networkManager->SendMessage(nodeId, voteResponse);
}

void DistributedBatchManager::HandleVoteResponse(int nodeId, const NetworkManager::Message& msg) {
    // Handle vote response
    
    if (m_Impl->role != NodeRole::CANDIDATE) {
        return;  // Not a candidate anymore
    }
    
    // TODO: Extract vote granted flag from message
    bool voteGranted = true;  // Placeholder
    
    if (voteGranted) {
        m_Impl->electionState.voteCount++;
        
        std::cout << "Received vote from node " << nodeId 
                  << " (total: " << m_Impl->electionState.voteCount << ")" 
                  << std::endl;
    }
}

void DistributedBatchManager::HandleLeaderAnnouncement(int nodeId, const NetworkManager::Message& msg) {
    // Another node became leader
    
    std::cout << "Node " << nodeId << " is now the master" << std::endl;
    
    m_Impl->role = NodeRole::WORKER;
    m_Impl->currentMasterId = nodeId;
    m_Impl->electionState.electionInProgress = false;
    
    // Add new master to workers list if not already there
    if (m_Impl->workers.find(nodeId) == m_Impl->workers.end()) {
        WorkerNode master;
        master.nodeId = nodeId;
        master.isHealthy = true;
        master.lastHeartbeat = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        
        m_Impl->workers[nodeId] = master;
    }
}

bool DistributedBatchManager::IsMaster() const {
    return m_Impl->role == NodeRole::MASTER;
}

int DistributedBatchManager::GetMasterNodeId() const {
    return m_Impl->currentMasterId;
}

// Update message handlers to include election messages
void DistributedBatchManager::HandleMasterMessage(int nodeId, const NetworkManager::Message& msg) {
    switch (msg.type) {
        case NetworkManager::MessageType::VOTE_REQUEST:
            HandleVoteRequest(nodeId, msg);
            break;
            
        case NetworkManager::MessageType::VOTE_RESPONSE:
            HandleVoteResponse(nodeId, msg);
            break;
            
        // ... existing cases ...
        
        default:
            break;
    }
}

void DistributedBatchManager::HandleWorkerMessage(int nodeId, const NetworkManager::Message& msg) {
    switch (msg.type) {
        case NetworkManager::MessageType::VOTE_REQUEST:
            HandleVoteRequest(nodeId, msg);
            break;
            
        case NetworkManager::MessageType::LEADER_ANNOUNCEMENT:
            HandleLeaderAnnouncement(nodeId, msg);
            break;
            
        // ... existing cases ...
        
        default:
            break;
    }
}
