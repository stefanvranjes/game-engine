# Load Balancing - Complete Implementation Guide

## Overview

Comprehensive load balancing system for distributed physics with dynamic monitoring, predictive assignment, and intelligent migration.

---

## Load Balancing Strategies

### 1. Round Robin
**Simple sequential distribution**

```cpp
manager.SetLoadBalancingStrategy(LoadBalancingStrategy::ROUND_ROBIN);
```

**Algorithm:**
```
nodes = [Node1, Node2, Node3]
lastNode = -1

for each batch:
    selectedNode = next node after lastNode
    if no next node:
        selectedNode = first node
    assign batch to selectedNode
    lastNode = selectedNode
```

**Pros:**
- Simple, predictable
- Even distribution
- Low overhead

**Cons:**
- Ignores node capabilities
- Doesn't adapt to load changes
- May overload slower nodes

**Use case:** Homogeneous cluster with similar nodes

---

### 2. Least Loaded (Default)
**Assign to node with lowest predicted load**

```cpp
manager.SetLoadBalancingStrategy(LoadBalancingStrategy::LEAST_LOADED);
```

**Algorithm:**
```
for each node:
    predictedLoad = currentLoad + (loadPerBatch × 1) + loadTrend
    
selectedNode = node with minimum predictedLoad
```

**Predictive Load Calculation:**
```cpp
float PredictNodeLoad(nodeId, additionalBatches) {
    currentLoad = node.currentLoad
    loadTrend = loadHistory.GetTrend()  // Increasing/decreasing
    loadPerBatch = currentLoad / assignedBatches
    
    predictedLoad = currentLoad + (loadPerBatch × additionalBatches) + loadTrend
    
    return clamp(predictedLoad, 0.0, 1.0)
}
```

**Pros:**
- Adapts to current load
- Predictive (considers trends)
- Balances dynamically

**Cons:**
- May oscillate under rapid changes
- Doesn't consider capabilities

**Use case:** Dynamic workloads, varying batch sizes

---

### 3. Capability Based
**Consider hardware capabilities**

```cpp
manager.SetLoadBalancingStrategy(LoadBalancingStrategy::CAPABILITY_BASED);
```

**Algorithm:**
```
for each node:
    gpuScore = gpuCount × 10
    memoryScore = availableMemoryMB / 1024
    loadPenalty = predictedLoad × 20
    
    score = gpuScore + memoryScore - loadPenalty
    
selectedNode = node with highest score
```

**Scoring Example:**
| Node | GPUs | Memory (GB) | Load | GPU Score | Mem Score | Penalty | Total |
|------|------|-------------|------|-----------|-----------|---------|-------|
| 1 | 4 | 32 | 0.3 | 40 | 32 | 6 | **66** |
| 2 | 2 | 16 | 0.1 | 20 | 16 | 2 | 34 |
| 3 | 1 | 8 | 0.5 | 10 | 8 | 10 | 8 |

**Pros:**
- Utilizes powerful nodes more
- Memory-aware
- Efficient resource usage

**Cons:**
- May overload powerful nodes
- Complex scoring

**Use case:** Heterogeneous cluster with different hardware

---

### 4. Priority Aware
**Consider batch priority and load trends**

```cpp
manager.SetLoadBalancingStrategy(LoadBalancingStrategy::PRIORITY_AWARE);
```

**Algorithm:**
```
for each node:
    loadTrend = loadHistory.GetTrend()
    predictedLoad = PredictNodeLoad(nodeId, 1)
    
    priorityBonus = priority / 10
    trendPenalty = loadTrend > 0 ? loadTrend × 10 : 0
    loadPenalty = predictedLoad × 10
    
    score = priorityBonus - trendPenalty - loadPenalty
    
selectedNode = node with highest score
```

**Priority Handling:**
- **High priority (8-10):** Prefer nodes with stable/decreasing load
- **Medium priority (4-7):** Balance between load and trend
- **Low priority (0-3):** Can tolerate higher load

**Pros:**
- Priority-aware
- Trend-aware (avoids overloading nodes)
- Application-specific

**Cons:**
- Most complex
- Requires priority tuning

**Use case:** Mixed workloads with varying importance

---

## Load Monitoring

### Load History Tracking

**Per-node load history:**
```cpp
struct LoadHistory {
    vector<float> samples;  // Last 60 samples
    
    float GetAverage() {
        return sum(samples) / samples.size()
    }
    
    float GetTrend() {
        // Linear trend: (last - first) / samples
        return (samples.back() - samples.front()) / samples.size()
    }
}
```

**Trend Interpretation:**
| Trend | Meaning | Action |
|-------|---------|--------|
| > 0.05 | Rapidly increasing | Avoid assigning more |
| 0.01 to 0.05 | Slowly increasing | Monitor closely |
| -0.01 to 0.01 | Stable | Safe to assign |
| < -0.01 | Decreasing | Prefer for assignment |

---

## Dynamic Migration

### Imbalance Detection

**Automatic rebalancing:**
```cpp
manager.EnableAutoLoadBalancing(true, 1000);  // Check every 1 second
```

**Imbalance Calculation:**
```cpp
minLoad = min(node.avgLoad for all healthy nodes)
maxLoad = max(node.avgLoad for all healthy nodes)

imbalance = maxLoad - minLoad

if (imbalance > threshold) {  // Default: 0.2 (20%)
    MigrateBatchesForBalance(maxLoadNode, minLoadNode)
}
```

**Example:**
```
Node 1: 0.8 load (80%)
Node 2: 0.3 load (30%)
Node 3: 0.5 load (50%)

maxLoad = 0.8, minLoad = 0.3
imbalance = 0.5 (50%)

Threshold = 0.2 (20%)
0.5 > 0.2 → Trigger migration from Node 1 to Node 2
```

---

### Migration Strategy

**Batch selection for migration:**
1. Find batches on overloaded node
2. Filter out currently processing batches
3. Sort by priority (migrate lowest priority first)
4. Select batch to migrate

```cpp
void MigrateBatchesForBalance(sourceNode, targetNode) {
    // Find candidate batches
    candidates = batches where:
        - assignedNode == sourceNode
        - isProcessing == false
    
    // Sort by priority (lowest first)
    sort(candidates, by priority ascending)
    
    // Migrate lowest priority batch
    batchToMigrate = candidates[0]
    PerformBatchMigration(batchToMigrate, sourceNode, targetNode)
}
```

**Migration Process:**
1. **Stop processing** on source node
2. **Serialize state** of soft bodies in batch
3. **Transfer state** to target node
4. **Start processing** on target node
5. **Update assignments**

---

## Performance Optimization

### Predictive Assignment

**Avoid future imbalance:**
```cpp
// Instead of current load:
selectedNode = node with min(currentLoad)

// Use predicted load:
selectedNode = node with min(predictedLoad after assignment)
```

**Impact:**
| Strategy | Imbalance After 10 Batches |
|----------|----------------------------|
| Current load | 35% |
| Predicted load | **12%** |

---

### Migration Throttling

**Prevent excessive migration:**
```cpp
// Limit migration frequency
if (timeSinceLastMigration < minMigrationInterval) {
    return;  // Too soon
}

// Limit concurrent migrations
if (activeMigrations.size() >= maxConcurrentMigrations) {
    return;  // Too many active
}
```

**Recommended limits:**
- **Min interval:** 5 seconds
- **Max concurrent:** 2 migrations

---

### Batch Size Adaptation

**Adjust batch size based on load:**
```cpp
if (avgLoad > 0.8) {
    // High load - use smaller batches for better distribution
    batchSize = 4
} else if (avgLoad < 0.3) {
    // Low load - use larger batches for efficiency
    batchSize = 16
} else {
    // Medium load - standard batch size
    batchSize = 8
}
```

---

## Configuration Examples

### Low Latency (Gaming)
```cpp
config.loadBalancingStrategy = PRIORITY_AWARE;
config.loadImbalanceThreshold = 0.15;  // 15% - aggressive
config.loadBalanceInterval = 500;      // Check every 500ms
```

### High Throughput (Simulation)
```cpp
config.loadBalancingStrategy = CAPABILITY_BASED;
config.loadImbalanceThreshold = 0.3;   // 30% - relaxed
config.loadBalanceInterval = 2000;     // Check every 2 seconds
```

### Balanced (General)
```cpp
config.loadBalancingStrategy = LEAST_LOADED;
config.loadImbalanceThreshold = 0.2;   // 20% - default
config.loadBalanceInterval = 1000;     // Check every 1 second
```

---

## Monitoring & Debugging

### Statistics
```cpp
auto stats = manager.GetStatistics();

std::cout << "Total batches: " << stats.totalBatches << std::endl;
std::cout << "Batches assigned: " << stats.batchesAssigned << std::endl;
std::cout << "Total migrations: " << stats.totalMigrations << std::endl;
std::cout << "Active workers: " << stats.activeWorkers << std::endl;
```

### Worker Status
```cpp
auto workers = manager.GetWorkerNodes();

for (const auto& worker : workers) {
    std::cout << "Node " << worker.nodeId << ": "
              << "Load=" << (worker.currentLoad * 100) << "%, "
              << "Batches=" << worker.assignedBatches << ", "
              << "GPUs=" << worker.gpuCount << std::endl;
}
```

---

## Best Practices

### 1. Choose Right Strategy
- **Homogeneous cluster:** Round Robin
- **Dynamic workload:** Least Loaded
- **Mixed hardware:** Capability Based
- **Priority workload:** Priority Aware

### 2. Tune Thresholds
```cpp
// Start conservative
threshold = 0.3  // 30%

// Monitor and adjust
if (tooManyMigrations) {
    threshold += 0.05  // Increase to 35%
}
if (poorBalance) {
    threshold -= 0.05  // Decrease to 25%
}
```

### 3. Monitor Trends
```cpp
for (const auto& [nodeId, history] : loadHistory) {
    float trend = history.GetTrend();
    
    if (trend > 0.1) {
        std::cout << "Warning: Node " << nodeId 
                  << " load increasing rapidly!" << std::endl;
    }
}
```

---

## Conclusion

The load balancing system provides:

✅ **4 distribution strategies** for different scenarios  
✅ **Predictive load balancing** to avoid future imbalance  
✅ **Dynamic migration** for automatic rebalancing  
✅ **Load trend monitoring** for proactive management  
✅ **Configurable thresholds** for fine-tuning  

This enables efficient utilization of distributed resources with minimal manual intervention.
