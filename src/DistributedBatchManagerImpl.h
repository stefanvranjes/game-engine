#pragma once

#include "DistributedBatchManager.h"
#include "NetworkManager.h"
#include "StateSerializer.h"
#include "GpuBatchManager.h"
#ifdef HAS_RDMA
#include "RdmaManager.h"
#endif

#include <iostream>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <thread>
#include <atomic>
#include <chrono>
#include <random>
#include <string>

// Structs used by Impl
struct BatchResult {
    uint32_t batchId;
    int workerNodeId;
    uint64_t processingTimeMs;
    uint64_t timestamp;
    std::vector<uint8_t> serializedStates;  // Serialized soft body states
    bool success;
    std::string errorMessage;
};

struct MigrationRecord {
    uint32_t batchId;
    int sourceNode;
    int targetNode;
    uint64_t startTime;
    bool completed;
};

struct LoadHistory {
    std::vector<float> samples;
    size_t maxSamples = 60;  // Keep 60 samples
    
    void AddSample(float load) {
        samples.push_back(load);
        if (samples.size() > maxSamples) {
            samples.erase(samples.begin());
        }
    }
    
    float GetAverage() const {
        if (samples.empty()) return 0.0f;
        float sum = 0.0f;
        for (float s : samples) sum += s;
        return sum / samples.size();
    }
    
    float GetTrend() const {
        if (samples.size() < 2) return 0.0f;
        // Simple linear trend: (last - first) / samples
        return (samples.back() - samples.front()) / samples.size();
    }
};

struct DistributedBatchManager::Impl {
    NodeRole role = NodeRole::STANDALONE;
    LoadBalancingStrategy loadBalancingStrategy = LoadBalancingStrategy::LEAST_LOADED;
    
    // Networking
    std::unique_ptr<NetworkManager> networkManager;
    std::unique_ptr<StateSerializer> stateSerializer;
    
    // Master state
    std::unordered_map<int, WorkerNode> workers;
    std::unordered_map<uint32_t, BatchAssignment> batchAssignments;
    std::unordered_map<PhysXSoftBody*, uint32_t> softBodyToBatch;
    uint32_t nextBatchId = 1;
    
    // Worker state
    std::unique_ptr<GpuBatchManager> localBatchManager;
    std::unordered_map<uint32_t, std::vector<PhysXSoftBody*>> assignedBatches;
    
    // Load balancing
    bool autoLoadBalancing = false;
    uint32_t loadBalanceInterval = 1000;
    float loadImbalanceThreshold = 0.2f;  // 20% imbalance triggers rebalancing
    std::thread loadBalanceThread;
    std::atomic<bool> running{false};
    
    // Load monitoring
    std::unordered_map<int, LoadHistory> loadHistory;
    
    // Migration tracking
    std::vector<MigrationRecord> activeMigrations;
    std::mutex migrationMutex;
    
    // Fault tolerance
    uint32_t heartbeatIntervalMs = 1000;     // Send heartbeat every 1 second
    uint32_t heartbeatTimeoutMs = 5000;      // Timeout after 5 seconds
    std::thread heartbeatThread;
    std::thread monitorThread;
    
    // Statistics
    DistributedStats stats;
    std::mutex statsMutex;

    // Master failover (from Failover.cpp)
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

    // Hierarchical distribution (from Hierarchical.cpp)
    HierarchyConfig hierarchyConfig;
    std::unordered_map<std::string, int> regionalMasters;  // region -> node ID
    std::unordered_map<std::string, std::vector<int>> regionalWorkers;  // region -> workers
    std::unordered_map<int, std::string> nodeRegions;  // node ID -> region

    // RDMA support (from Rdma.cpp)
#ifdef HAS_RDMA
    std::unique_ptr<RdmaManager> rdmaManager;
    bool rdmaEnabled = false;
    std::unordered_map<PhysXSoftBody*, void*> gpuBuffers;  // Soft body -> GPU buffer mapping
#endif

    // Result handling (from Results.cpp)
    std::unordered_map<uint32_t, BatchResult> pendingResults;
    std::mutex resultMutex;
    uint32_t resultTimeoutMs = 5000;  // 5 second timeout for results

    // State synchronization (from Sync.cpp)
    uint32_t syncIntervalMs = 100;           // Sync every 100ms (10 Hz)
    bool useDeltaSync = true;                // Use delta encoding
    std::thread syncThread;
    std::unordered_map<PhysXSoftBody*, StateSerializer::SoftBodyState> lastSyncedStates;
    std::mutex syncMutex;
    uint32_t currentFrame = 0;

    ~Impl() {
        running = false;
        if (loadBalanceThread.joinable()) loadBalanceThread.join();
        if (heartbeatThread.joinable()) heartbeatThread.join();
        if (monitorThread.joinable()) monitorThread.join();
        if (electionThread.joinable()) electionThread.join();
        if (syncThread.joinable()) syncThread.join();
    }
};
