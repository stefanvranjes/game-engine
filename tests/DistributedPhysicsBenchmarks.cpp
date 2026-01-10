// Performance Benchmarks for Distributed Physics

#include "DistributedBatchManager.h"
#include "GpuBatchManager.h"
#include "PhysXSoftBody.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <fstream>

using namespace std::chrono;

struct BenchmarkResult {
    std::string testName;
    int nodeCount;
    int softBodyCount;
    double avgProcessingTimeMs;
    double throughput;  // soft bodies per second
    double efficiency;  // percentage
    size_t totalBandwidthKB;
};

class DistributedPhysicsBenchmark {
public:
    void RunAllBenchmarks() {
        std::cout << "=== Distributed Physics Performance Benchmarks ===" << std::endl;
        
        BenchmarkScalability();
        BenchmarkLoadBalancing();
        BenchmarkFailover();
        BenchmarkBandwidth();
        
        PrintResults();
        SaveResultsToCSV("benchmark_results.csv");
    }
    
private:
    std::vector<BenchmarkResult> results;
    
    void BenchmarkScalability() {
        std::cout << "\n[1/4] Scalability Benchmark" << std::endl;
        
        std::vector<int> nodeCounts = {1, 2, 4, 8};
        int softBodyCount = 100;
        
        for (int nodes : nodeCounts) {
            std::cout << "  Testing with " << nodes << " nodes..." << std::endl;
            
            // Setup
            DistributedBatchManager master;
            master.InitializeAsMaster(8080);
            
            std::vector<std::unique_ptr<DistributedBatchManager>> workers;
            for (int i = 0; i < nodes; i++) {
                auto worker = std::make_unique<DistributedBatchManager>();
                worker->InitializeAsWorker("localhost", 8080);
                workers.push_back(std::move(worker));
            }
            
            std::this_thread::sleep_for(milliseconds(500));
            
            // Create soft bodies
            std::vector<PhysXSoftBody*> softBodies;
            for (int i = 0; i < softBodyCount; i++) {
                auto* sb = new PhysXSoftBody();
                softBodies.push_back(sb);
                master.AddSoftBody(sb, 5);
            }
            
            // Benchmark
            auto start = high_resolution_clock::now();
            
            for (int frame = 0; frame < 100; frame++) {
                master.ProcessBatches(0.016f);
                std::this_thread::sleep_for(milliseconds(16));
            }
            
            auto end = high_resolution_clock::now();
            auto duration = duration_cast<milliseconds>(end - start).count();
            
            // Calculate metrics
            double avgTime = duration / 100.0;
            double throughput = (softBodyCount * 100.0) / (duration / 1000.0);
            double efficiency = (throughput / nodes) / (throughput / 1.0) * 100.0;
            
            BenchmarkResult result;
            result.testName = "Scalability";
            result.nodeCount = nodes;
            result.softBodyCount = softBodyCount;
            result.avgProcessingTimeMs = avgTime;
            result.throughput = throughput;
            result.efficiency = efficiency;
            
            results.push_back(result);
            
            // Cleanup
            for (auto* sb : softBodies) delete sb;
            for (auto& worker : workers) worker->Shutdown();
            master.Shutdown();
            
            std::this_thread::sleep_for(milliseconds(500));
        }
    }
    
    void BenchmarkLoadBalancing() {
        std::cout << "\n[2/4] Load Balancing Benchmark" << std::endl;
        
        std::vector<std::string> strategies = {
            "ROUND_ROBIN", "LEAST_LOADED", "CAPABILITY_BASED", "PRIORITY_AWARE"
        };
        
        for (size_t i = 0; i < strategies.size(); i++) {
            std::cout << "  Testing " << strategies[i] << "..." << std::endl;
            
            DistributedBatchManager master;
            master.InitializeAsMaster(8080);
            master.SetLoadBalancingStrategy(
                static_cast<DistributedBatchManager::LoadBalancingStrategy>(i));
            
            // 4 workers
            std::vector<std::unique_ptr<DistributedBatchManager>> workers;
            for (int j = 0; j < 4; j++) {
                auto worker = std::make_unique<DistributedBatchManager>();
                worker->InitializeAsWorker("localhost", 8080);
                workers.push_back(std::move(worker));
            }
            
            std::this_thread::sleep_for(milliseconds(500));
            
            // 100 soft bodies with varying priorities
            std::vector<PhysXSoftBody*> softBodies;
            for (int j = 0; j < 100; j++) {
                auto* sb = new PhysXSoftBody();
                softBodies.push_back(sb);
                master.AddSoftBody(sb, j % 10);
            }
            
            // Benchmark
            auto start = high_resolution_clock::now();
            
            for (int frame = 0; frame < 50; frame++) {
                master.ProcessBatches(0.016f);
                std::this_thread::sleep_for(milliseconds(16));
            }
            
            auto end = high_resolution_clock::now();
            auto duration = duration_cast<milliseconds>(end - start).count();
            
            BenchmarkResult result;
            result.testName = "LoadBalancing_" + strategies[i];
            result.nodeCount = 4;
            result.softBodyCount = 100;
            result.avgProcessingTimeMs = duration / 50.0;
            result.throughput = (100.0 * 50.0) / (duration / 1000.0);
            
            results.push_back(result);
            
            // Cleanup
            for (auto* sb : softBodies) delete sb;
            for (auto& worker : workers) worker->Shutdown();
            master.Shutdown();
            
            std::this_thread::sleep_for(milliseconds(500));
        }
    }
    
    void BenchmarkFailover() {
        std::cout << "\n[3/4] Failover Benchmark" << std::endl;
        
        DistributedBatchManager master;
        master.InitializeAsMaster(8080);
        master.EnableMasterFailover(true);
        
        std::vector<std::unique_ptr<DistributedBatchManager>> workers;
        for (int i = 0; i < 3; i++) {
            auto worker = std::make_unique<DistributedBatchManager>();
            worker->InitializeAsWorker("localhost", 8080);
            worker->EnableMasterFailover(true);
            workers.push_back(std::move(worker));
        }
        
        std::this_thread::sleep_for(milliseconds(500));
        
        // Measure failover time
        auto start = high_resolution_clock::now();
        master.Shutdown();
        
        // Wait for election
        std::this_thread::sleep_for(milliseconds(500));
        
        auto end = high_resolution_clock::now();
        auto failoverTime = duration_cast<milliseconds>(end - start).count();
        
        std::cout << "  Failover time: " << failoverTime << "ms" << std::endl;
        
        // Cleanup
        for (auto& worker : workers) worker->Shutdown();
    }
    
    void BenchmarkBandwidth() {
        std::cout << "\n[4/4] Bandwidth Benchmark" << std::endl;
        
        DistributedBatchManager master;
        master.InitializeAsMaster(8080);
        master.EnableDeltaSync(true);
        master.SetSyncInterval(100);  // 10 Hz
        
        DistributedBatchManager worker;
        worker.InitializeAsWorker("localhost", 8080);
        
        std::this_thread::sleep_for(milliseconds(500));
        
        // 50 soft bodies
        std::vector<PhysXSoftBody*> softBodies;
        for (int i = 0; i < 50; i++) {
            auto* sb = new PhysXSoftBody();
            softBodies.push_back(sb);
            master.AddSoftBody(sb, 5);
        }
        
        // Run for 10 seconds
        auto start = high_resolution_clock::now();
        
        for (int i = 0; i < 100; i++) {
            master.ProcessBatches(0.016f);
            std::this_thread::sleep_for(milliseconds(100));
        }
        
        auto end = high_resolution_clock::now();
        
        auto stats = master.GetStatistics();
        
        std::cout << "  Avg network latency: " << stats.avgNetworkLatencyMs << "ms" << std::endl;
        std::cout << "  Batches completed: " << stats.batchesCompleted << std::endl;
        
        // Cleanup
        for (auto* sb : softBodies) delete sb;
        worker.Shutdown();
        master.Shutdown();
    }
    
    void PrintResults() {
        std::cout << "\n=== Benchmark Results ===" << std::endl;
        std::cout << std::endl;
        
        printf("%-30s | %5s | %8s | %10s | %12s | %10s\n",
               "Test", "Nodes", "Bodies", "Avg Time", "Throughput", "Efficiency");
        printf("-------------------------------|-------|----------|------------|--------------|----------\n");
        
        for (const auto& result : results) {
            printf("%-30s | %5d | %8d | %8.2f ms | %10.1f/s | %8.1f%%\n",
                   result.testName.c_str(),
                   result.nodeCount,
                   result.softBodyCount,
                   result.avgProcessingTimeMs,
                   result.throughput,
                   result.efficiency);
        }
    }
    
    void SaveResultsToCSV(const std::string& filename) {
        std::ofstream file(filename);
        
        file << "Test,Nodes,Bodies,AvgTimeMs,Throughput,Efficiency\n";
        
        for (const auto& result : results) {
            file << result.testName << ","
                 << result.nodeCount << ","
                 << result.softBodyCount << ","
                 << result.avgProcessingTimeMs << ","
                 << result.throughput << ","
                 << result.efficiency << "\n";
        }
        
        file.close();
        std::cout << "\nResults saved to " << filename << std::endl;
    }
};

int main() {
    DistributedPhysicsBenchmark benchmark;
    benchmark.RunAllBenchmarks();
    return 0;
}
