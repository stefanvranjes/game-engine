// Integration Tests for Distributed Physics System

#include "DistributedBatchManager.h"
#include "GpuBatchManager.h"
#include "PhysXSoftBody.h"
#include <gtest/gtest.h>
#include <thread>
#include <chrono>

class DistributedPhysicsTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup test environment
    }
    
    void TearDown() override {
        // Cleanup
    }
};

// ============================================================================
// Test 1: Basic Master-Worker Connection
// ============================================================================

TEST_F(DistributedPhysicsTest, MasterWorkerConnection) {
    // Start master
    DistributedBatchManager master;
    ASSERT_TRUE(master.InitializeAsMaster(8080));
    
    // Start worker
    DistributedBatchManager worker;
    ASSERT_TRUE(worker.InitializeAsWorker("localhost", 8080));
    
    // Wait for connection
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    // Verify connection
    auto workers = master.GetWorkerNodes();
    EXPECT_EQ(workers.size(), 1);
    EXPECT_TRUE(workers[0].isHealthy);
    
    // Cleanup
    worker.Shutdown();
    master.Shutdown();
}

// ============================================================================
// Test 2: Batch Distribution
// ============================================================================

TEST_F(DistributedPhysicsTest, BatchDistribution) {
    // Start master
    DistributedBatchManager master;
    master.InitializeAsMaster(8080);
    master.SetLoadBalancingStrategy(
        DistributedBatchManager::LoadBalancingStrategy::ROUND_ROBIN);
    
    // Start 2 workers
    DistributedBatchManager worker1, worker2;
    worker1.InitializeAsWorker("localhost", 8080);
    worker2.InitializeAsWorker("localhost", 8080);
    
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    
    // Add soft bodies
    std::vector<PhysXSoftBody*> softBodies;
    for (int i = 0; i < 10; i++) {
        auto* sb = new PhysXSoftBody();
        softBodies.push_back(sb);
        master.AddSoftBody(sb, 5);
    }
    
    // Process batches
    master.ProcessBatches(0.016f);
    
    // Wait for distribution
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    
    // Verify distribution
    auto stats = master.GetStatistics();
    EXPECT_GT(stats.batchesAssigned, 0);
    
    // Cleanup
    for (auto* sb : softBodies) delete sb;
    worker1.Shutdown();
    worker2.Shutdown();
    master.Shutdown();
}

// ============================================================================
// Test 3: Load Balancing Strategies
// ============================================================================

TEST_F(DistributedPhysicsTest, LoadBalancingStrategies) {
    DistributedBatchManager master;
    master.InitializeAsMaster(8080);
    
    DistributedBatchManager worker1, worker2;
    worker1.InitializeAsWorker("localhost", 8080);
    worker2.InitializeAsWorker("localhost", 8080);
    
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    
    // Test each strategy
    std::vector<DistributedBatchManager::LoadBalancingStrategy> strategies = {
        DistributedBatchManager::LoadBalancingStrategy::ROUND_ROBIN,
        DistributedBatchManager::LoadBalancingStrategy::LEAST_LOADED,
        DistributedBatchManager::LoadBalancingStrategy::CAPABILITY_BASED,
        DistributedBatchManager::LoadBalancingStrategy::PRIORITY_AWARE
    };
    
    for (auto strategy : strategies) {
        master.SetLoadBalancingStrategy(strategy);
        
        // Add soft bodies
        std::vector<PhysXSoftBody*> softBodies;
        for (int i = 0; i < 20; i++) {
            auto* sb = new PhysXSoftBody();
            softBodies.push_back(sb);
            master.AddSoftBody(sb, i % 10);  // Varying priorities
        }
        
        // Process
        master.ProcessBatches(0.016f);
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        
        // Verify batches were assigned
        auto stats = master.GetStatistics();
        EXPECT_GT(stats.batchesAssigned, 0);
        
        // Cleanup
        for (auto* sb : softBodies) {
            master.RemoveSoftBody(sb);
            delete sb;
        }
    }
    
    worker1.Shutdown();
    worker2.Shutdown();
    master.Shutdown();
}

// ============================================================================
// Test 4: Worker Failure and Recovery
// ============================================================================

TEST_F(DistributedPhysicsTest, WorkerFailureRecovery) {
    DistributedBatchManager master;
    master.InitializeAsMaster(8080);
    master.SetHeartbeatConfig(500, 2000);  // Fast timeout for testing
    
    DistributedBatchManager worker1, worker2;
    worker1.InitializeAsWorker("localhost", 8080);
    worker2.InitializeAsWorker("localhost", 8080);
    
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    
    // Add soft bodies
    std::vector<PhysXSoftBody*> softBodies;
    for (int i = 0; i < 10; i++) {
        auto* sb = new PhysXSoftBody();
        softBodies.push_back(sb);
        master.AddSoftBody(sb, 5);
    }
    
    master.ProcessBatches(0.016f);
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    
    auto initialStats = master.GetStatistics();
    EXPECT_EQ(initialStats.activeWorkers, 2);
    
    // Simulate worker1 failure
    worker1.Shutdown();
    
    // Wait for failure detection
    std::this_thread::sleep_for(std::chrono::milliseconds(2500));
    
    // Verify failure detected
    auto failureStats = master.GetStatistics();
    EXPECT_EQ(failureStats.activeWorkers, 1);
    EXPECT_GT(failureStats.failedNodes, 0);
    
    // Verify batches reassigned
    EXPECT_GT(failureStats.batchesFailed, 0);
    
    // Cleanup
    for (auto* sb : softBodies) delete sb;
    worker2.Shutdown();
    master.Shutdown();
}

// ============================================================================
// Test 5: Master Failover
// ============================================================================

TEST_F(DistributedPhysicsTest, MasterFailover) {
    DistributedBatchManager master;
    master.InitializeAsMaster(8080);
    master.EnableMasterFailover(true);
    
    DistributedBatchManager worker1, worker2;
    worker1.InitializeAsWorker("localhost", 8080);
    worker1.EnableMasterFailover(true);
    worker2.InitializeAsWorker("localhost", 8080);
    worker2.EnableMasterFailover(true);
    
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    
    // Verify master
    EXPECT_TRUE(master.IsMaster());
    EXPECT_FALSE(worker1.IsMaster());
    EXPECT_FALSE(worker2.IsMaster());
    
    // Simulate master failure
    master.Shutdown();
    
    // Wait for election
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    
    // One worker should become master
    bool hasNewMaster = worker1.IsMaster() || worker2.IsMaster();
    EXPECT_TRUE(hasNewMaster);
    
    // Cleanup
    worker1.Shutdown();
    worker2.Shutdown();
}

// ============================================================================
// Test 6: State Synchronization
// ============================================================================

TEST_F(DistributedPhysicsTest, StateSynchronization) {
    DistributedBatchManager master;
    master.InitializeAsMaster(8080);
    
    DistributedBatchManager worker;
    worker.InitializeAsWorker("localhost", 8080);
    
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    
    // Add soft body
    PhysXSoftBody* softBody = new PhysXSoftBody();
    master.AddSoftBody(softBody, 5);
    
    // Process
    master.ProcessBatches(0.016f);
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    
    // Verify state sync occurred
    auto stats = master.GetStatistics();
    EXPECT_GT(stats.batchesCompleted, 0);
    
    // Cleanup
    delete softBody;
    worker.Shutdown();
    master.Shutdown();
}

// ============================================================================
// Test 7: Dynamic Load Balancing
// ============================================================================

TEST_F(DistributedPhysicsTest, DynamicLoadBalancing) {
    DistributedBatchManager master;
    master.InitializeAsMaster(8080);
    master.EnableAutoLoadBalancing(true, 500);  // Check every 500ms
    
    DistributedBatchManager worker1, worker2;
    worker1.InitializeAsWorker("localhost", 8080);
    worker2.InitializeAsWorker("localhost", 8080);
    
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    
    // Add many soft bodies to create imbalance
    std::vector<PhysXSoftBody*> softBodies;
    for (int i = 0; i < 50; i++) {
        auto* sb = new PhysXSoftBody();
        softBodies.push_back(sb);
        master.AddSoftBody(sb, 5);
    }
    
    master.ProcessBatches(0.016f);
    
    // Wait for load balancing
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    
    // Verify migrations occurred
    auto stats = master.GetStatistics();
    EXPECT_GT(stats.totalMigrations, 0);
    
    // Cleanup
    for (auto* sb : softBodies) delete sb;
    worker1.Shutdown();
    worker2.Shutdown();
    master.Shutdown();
}

// ============================================================================
// Test 8: Batch Result Timeout
// ============================================================================

TEST_F(DistributedPhysicsTest, BatchResultTimeout) {
    DistributedBatchManager master;
    master.InitializeAsMaster(8080);
    master.SetResultTimeout(1000);  // 1 second timeout
    
    DistributedBatchManager worker;
    worker.InitializeAsWorker("localhost", 8080);
    
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    
    // Add soft body
    PhysXSoftBody* softBody = new PhysXSoftBody();
    master.AddSoftBody(softBody, 5);
    
    master.ProcessBatches(0.016f);
    
    // Simulate worker not responding (shutdown without unregister)
    // This would normally cause timeout
    
    // Wait for timeout
    std::this_thread::sleep_for(std::chrono::milliseconds(1500));
    
    // Verify timeout handling
    auto stats = master.GetStatistics();
    // Batch should be marked as failed or reassigned
    
    // Cleanup
    delete softBody;
    worker.Shutdown();
    master.Shutdown();
}

// ============================================================================
// Test 9: Multi-Node Scalability
// ============================================================================

TEST_F(DistributedPhysicsTest, MultiNodeScalability) {
    DistributedBatchManager master;
    master.InitializeAsMaster(8080);
    
    // Start 4 workers
    std::vector<std::unique_ptr<DistributedBatchManager>> workers;
    for (int i = 0; i < 4; i++) {
        auto worker = std::make_unique<DistributedBatchManager>();
        worker->InitializeAsWorker("localhost", 8080);
        workers.push_back(std::move(worker));
    }
    
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    
    // Verify all workers connected
    auto workerNodes = master.GetWorkerNodes();
    EXPECT_EQ(workerNodes.size(), 4);
    
    // Add many soft bodies
    std::vector<PhysXSoftBody*> softBodies;
    for (int i = 0; i < 100; i++) {
        auto* sb = new PhysXSoftBody();
        softBodies.push_back(sb);
        master.AddSoftBody(sb, i % 10);
    }
    
    // Process
    auto start = std::chrono::high_resolution_clock::now();
    master.ProcessBatches(0.016f);
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end - start).count();
    
    // Verify distribution
    auto stats = master.GetStatistics();
    EXPECT_GT(stats.batchesAssigned, 0);
    EXPECT_EQ(stats.activeWorkers, 4);
    
    std::cout << "Processed " << stats.batchesCompleted 
              << " batches across 4 workers in " << duration << "ms" << std::endl;
    
    // Cleanup
    for (auto* sb : softBodies) delete sb;
    for (auto& worker : workers) worker->Shutdown();
    master.Shutdown();
}

// ============================================================================
// Test 10: Graceful Shutdown
// ============================================================================

TEST_F(DistributedPhysicsTest, GracefulShutdown) {
    DistributedBatchManager master;
    master.InitializeAsMaster(8080);
    
    DistributedBatchManager worker;
    worker.InitializeAsWorker("localhost", 8080);
    
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    
    auto initialStats = master.GetStatistics();
    EXPECT_EQ(initialStats.activeWorkers, 1);
    
    // Graceful shutdown
    worker.Shutdown();
    
    // Wait briefly
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    
    // Verify worker removed
    auto finalStats = master.GetStatistics();
    EXPECT_EQ(finalStats.activeWorkers, 0);
    
    master.Shutdown();
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
