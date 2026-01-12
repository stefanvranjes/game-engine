#pragma once

#include "GpuBatchManager.h"
#include "NetworkManager.h"
#include "StateSerializer.h"
#include "ProtobufSerializer.h"
#include <memory>
#include <vector>
#include <unordered_map>
#include <mutex>

class PhysXSoftBody;

/**
 * @brief Distributed batch manager for multi-node physics simulation
 * 
 * Manages distribution of soft body batches across multiple worker nodes
 * with load balancing, fault tolerance, and state synchronization.
 */
class DistributedBatchManager {
public:
    /**
     * @brief Node role in distributed system
     */
    enum class NodeRole {
        MASTER,      // Coordinates work distribution
        WORKER,      // Processes assigned batches
        CANDIDATE,   // Participating in leader election
        STANDALONE   // Single-node mode (no distribution)
    };
    
    /**
     * @brief Election state
     */
    struct ElectionState {
        uint32_t currentTerm;        // Current election term
        int votedFor;                // Node voted for in current term
        int voteCount;               // Votes received in current term
        uint64_t electionTimeout;    // When to start new election
        bool electionInProgress;     // Is election currently running
        
        ElectionState() : currentTerm(0), votedFor(-1), voteCount(0),
                         electionTimeout(0), electionInProgress(false) {}
    };
    
    /**
     * @brief Load balancing strategy
     */
    enum class LoadBalancingStrategy {
        ROUND_ROBIN,        // Distribute evenly in sequence
        LEAST_LOADED,       // Assign to node with lowest load
        CAPABILITY_BASED,   // Consider GPU count and memory
        PRIORITY_AWARE      // Consider soft body priorities
    };
    
    /**
     * @brief Worker node information
     */
    struct WorkerNode {
        int nodeId;
        std::string address;
        uint16_t port;
        float currentLoad;
        size_t gpuCount;
        uint64_t totalMemoryMB;
        uint64_t availableMemoryMB;
        size_t assignedBatches;
        bool isHealthy;
        uint64_t lastHeartbeat;
    };
    
    /**
     * @brief Batch assignment
     */
    struct BatchAssignment {
        uint32_t batchId;
        int assignedNode;
        std::vector<PhysXSoftBody*> softBodies;
        uint32_t priority;
        uint64_t assignedTime;
        bool isProcessing;
    };
    
    /**
     * @brief Distributed batch statistics
     */
    struct DistributedStats {
        size_t totalBatches;
        size_t batchesAssigned;
        size_t batchesCompleted;
        size_t batchesFailed;
        size_t totalWorkers;
        size_t activeWorkers;
        float avgProcessingTimeMs;
        float avgNetworkLatencyMs;
        size_t totalMigrations;
        size_t failedNodes;
    };
    
    /**
     * @brief Hierarchy configuration
     */
    struct HierarchyConfig {
        NodeRole role;
        int tier;
        int parentNodeId;
        std::string region;
        std::vector<int> childNodes;
        
        HierarchyConfig() : role(NodeRole::STANDALONE), tier(0), parentNodeId(-1) {}
    };
    
    DistributedBatchManager();
    ~DistributedBatchManager();
    
    // Initialization
    
    /**
     * @brief Initialize as master node
     * @param port Port to listen on
     * @return True if successful
     */
    bool InitializeAsMaster(uint16_t port);
    
    /**
     * @brief Initialize as worker node
     * @param masterAddress Master node address
     * @param masterPort Master node port
     * @return True if successful
     */
    bool InitializeAsWorker(const std::string& masterAddress, uint16_t masterPort);
    
    /**
     * @brief Initialize as global master (hierarchical)
     * @param port Port to listen on
     * @return True if successful
     */
    bool InitializeAsGlobalMaster(uint16_t port);
    
    /**
     * @brief Initialize as regional master (hierarchical)
     * @param globalMasterAddress Global master address
     * @param globalMasterPort Global master port
     * @param localPort Port to listen on for workers
     * @param region Region identifier (e.g., "us-west", "europe")
     * @return True if successful
     */
    bool InitializeAsRegionalMaster(const std::string& globalMasterAddress,
                                     uint16_t globalMasterPort,
                                     uint16_t localPort,
                                     const std::string& region);
    
    /**
     * @brief Initialize as worker in specific region (hierarchical)
     * @param regionalMasterAddress Regional master address
     * @param regionalMasterPort Regional master port
     * @return True if successful
     */
    bool InitializeAsWorkerInRegion(const std::string& regionalMasterAddress,
                                     uint16_t regionalMasterPort);
    
    /**
     * @brief Initialize as standalone (no distribution)
     * @return True if successful
     */
    bool InitializeAsStandalone();
    
    /**
     * @brief Shutdown distributed system
     */
    void Shutdown();
    
    // Batch management (Master)
    
    /**
     * @brief Add soft body to distributed system
     * @param softBody Soft body to add
     * @param priority Priority level
     */
    void AddSoftBody(PhysXSoftBody* softBody, uint32_t priority = 0);
    
    /**
     * @brief Remove soft body from distributed system
     * @param softBody Soft body to remove
     */
    void RemoveSoftBody(PhysXSoftBody* softBody);
    
    /**
     * @brief Process all batches (distributes to workers)
     * @param deltaTime Time step
     */
    void ProcessBatches(float deltaTime);
    
    /**
     * @brief Set load balancing strategy
     * @param strategy Load balancing strategy
     */
    void SetLoadBalancingStrategy(LoadBalancingStrategy strategy);
    
    // Worker operations
    
    /**
     * @brief Process assigned batch (worker only)
     * @param batchId Batch ID to process
     * @param deltaTime Time step
     */
    void ProcessAssignedBatch(uint32_t batchId, float deltaTime);
    
    /**
     * @brief Report load to master (worker only)
     */
    void ReportLoad();
    
    // Migration
    
    /**
     * @brief Migrate soft body to different node
     * @param softBody Soft body to migrate
     * @param targetNode Target node ID
     */
    void MigrateSoftBody(PhysXSoftBody* softBody, int targetNode);
    
    /**
     * @brief Enable automatic load balancing
     * @param enable Enable/disable
     * @param checkIntervalMs Check interval in milliseconds
     */
    void EnableAutoLoadBalancing(bool enable, uint32_t checkIntervalMs = 1000);
    
    // Statistics
    
    /**
     * @brief Get distributed statistics
     * @return Statistics
     */
    DistributedStats GetStatistics() const;
    
    /**
     * @brief Get worker nodes
     * @return List of worker nodes
     */
    std::vector<WorkerNode> GetWorkerNodes() const;
    
    /**
     * @brief Get current node role
     * @return Node role
     */
    NodeRole GetNodeRole() const;
    
    // Master Failover
    
    /**
     * @brief Enable master failover with leader election
     * @param enable Enable/disable failover
     */
    void EnableMasterFailover(bool enable);
    
    /**
     * @brief Check if this node is the current master
     * @return True if master
     */
    bool IsMaster() const;
    
    /**
     * @brief Get current master node ID
     * @return Master node ID (-1 if unknown)
     */
    int GetMasterNodeId() const;
    
    /**
     * @brief Set result timeout
     * @param timeoutMs Timeout in milliseconds
     */
    void SetResultTimeout(uint32_t timeoutMs);
    
    // GPU-Direct RDMA
    
    /**
     * @brief Enable/disable RDMA for GPU-Direct transfers
     * @param enable Enable RDMA
     * @return True if RDMA available and enabled
     */
    bool EnableRdma(bool enable);
    
    /**
     * @brief Check if RDMA is enabled
     * @return True if RDMA enabled and available
     */
    bool IsRdmaEnabled() const;
    
    /**
     * @brief Register GPU buffer for RDMA
     * @param softBody Soft body
     * @param gpuPtr GPU memory pointer
     * @param size Buffer size
     * @return True if registered successfully
     */
    bool RegisterBufferForRdma(PhysXSoftBody* softBody, void* gpuPtr, size_t size);
    
    /**
     * @brief Migrate soft body using RDMA
     * @param softBody Soft body to migrate
     * @param targetNode Target node ID
     */
    void MigrateSoftBodyRdma(PhysXSoftBody* softBody, int targetNode);
    
    /**
     * @brief Sync state using RDMA
     * @param softBody Soft body
     * @param targetNode Target node ID
     */
    void SyncStateRdma(PhysXSoftBody* softBody, int targetNode);
    
    /**
     * @brief Print RDMA statistics
     */
    void PrintRdmaStatistics() const;
    
    // Hierarchical Distribution
    
    /**
     * @brief Get hierarchy configuration
     * @return Hierarchy config
     */
    const HierarchyConfig& GetHierarchyConfig() const;
    
    /**
     * @brief Register regional master
     * @param nodeId Regional master node ID
     * @param region Region identifier
     */
    void RegisterRegionalMaster(int nodeId, const std::string& region);
    
    /**
     * @brief Select region for batch assignment
     * @param softBodies Soft bodies in batch
     * @return Region identifier
     */
    std::string SelectRegionForBatch(const std::vector<PhysXSoftBody*>& softBodies);
    
    /**
     * @brief Assign batch to specific region
     * @param batchId Batch ID
     * @param region Region identifier
     */
    void AssignBatchToRegion(uint32_t batchId, const std::string& region);
    
    /**
     * @brief Broadcast message to all regions
     * @param msg Message to broadcast
     */
    void BroadcastToAllRegions(const NetworkManager::Message& msg);

private:
    struct Impl;
    std::unique_ptr<Impl> m_Impl;
    
    // Internal helpers
    int SelectNodeForBatch(const std::vector<PhysXSoftBody*>& softBodies, uint32_t priority);
    void AssignBatchToNode(uint32_t batchId, int nodeId);
    void HandleBatchResult(int nodeId, uint32_t batchId, const std::vector<uint8_t>& resultData);
    void HandleNodeFailure(int nodeId);
    void ReassignBatches(int failedNodeId);
    void CheckLoadBalance();
    
    // Message handlers
    void HandleMasterMessage(int nodeId, const NetworkManager::Message& msg);
    void HandleWorkerMessage(int nodeId, const NetworkManager::Message& msg);
    
    // Registration
    bool RegisterWithMaster();
    void UnregisterFromMaster();
    void HandleNodeRegistration(int nodeId, const NetworkManager::Message& msg);
    void HandleNodeUnregistration(int nodeId, const NetworkManager::Message& msg);
    
    // Fault tolerance
    void SendHeartbeats();
    void MonitorWorkerHealth();
    void RegisterWorker(int nodeId, const NetworkManager::NodeInfo& info);
    void HandleHeartbeat(int nodeId, const NetworkManager::Message& msg);
    
    // Failover
    void MonitorMasterHealth();
    void StartElection();
    void RequestVotes();
    void WaitForElectionResult();
    void BecomeLeader();
    void AnnounceLeadership();
    void ReconstructMasterState();
    void StartMasterThreads();
    void HandleVoteRequest(int nodeId, const NetworkManager::Message& msg);
    void HandleVoteResponse(int nodeId, const NetworkManager::Message& msg);
    void HandleLeaderAnnouncement(int nodeId, const NetworkManager::Message& msg);
    
    // State synchronization
    void StartStateSynchronization();
    void SynchronizeState();
    void SyncSoftBodyState(PhysXSoftBody* softBody);
    void HandleStateSyncFull(int nodeId, const NetworkManager::Message& msg);
    void HandleStateSyncDelta(int nodeId, const NetworkManager::Message& msg);
    void BroadcastStateUpdate(PhysXSoftBody* softBody);
    
    // Result handling
    std::vector<uint8_t> SerializeBatchResults(const std::vector<PhysXSoftBody*>& softBodies);
    void SendBatchResults(uint32_t batchId, const std::vector<uint8_t>& resultData,
                         uint64_t processingTimeMs, bool success, const std::string& errorMessage);
    void DeserializeBatchResults(const std::vector<PhysXSoftBody*>& softBodies,
                                 const std::vector<uint8_t>& resultData);
    void ParseBatchResultMessage(const NetworkManager::Message& msg);
    void MonitorResultTimeouts();
    
    // Load balancing helpers
    void UpdateWorkerLoad(int nodeId, float load);
    float PredictNodeLoad(int nodeId, size_t additionalBatches);
    void MigrateBatchesForBalance(int sourceNode, int targetNode);
    void PerformBatchMigration(uint32_t batchId, int fromNode, int toNode);
    float CalculateLoadImbalance() const;
};
