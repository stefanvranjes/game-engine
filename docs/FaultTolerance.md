# Fault Tolerance - Complete Implementation Guide

## Overview

Comprehensive fault tolerance system for distributed physics with heartbeat monitoring, automatic failure detection, and batch reassignment.

---

## Heartbeat System

### Worker Heartbeats

**Workers send periodic heartbeats to master:**
```cpp
// Worker side
void SendHeartbeats() {
    while (running) {
        Message heartbeat;
        heartbeat.type = HEARTBEAT;
        heartbeat.sourceNode = localNodeId;
        heartbeat.targetNode = 0;  // Master
        
        networkManager.SendMessage(0, heartbeat, false);  // Unreliable
        
        sleep(heartbeatIntervalMs);  // Default: 1000ms
    }
}
```

**Configuration:**
```cpp
manager.SetHeartbeatConfig(1000, 5000);
// interval: 1000ms (1 second)
// timeout: 5000ms (5 seconds)
```

**Why unreliable?**
- Heartbeats are periodic - loss acceptable
- Next heartbeat will arrive soon
- Reduces network overhead
- Timeout handles multiple losses

---

## Failure Detection

### Master Monitoring

**Master monitors all workers:**
```cpp
void MonitorWorkerHealth() {
    while (running) {
        now = getCurrentTime();
        
        for (worker in workers) {
            timeSinceHeartbeat = now - worker.lastHeartbeat;
            
            if (timeSinceHeartbeat > heartbeatTimeoutMs) {
                // Node failed!
                worker.isHealthy = false;
                HandleNodeFailure(worker.nodeId);
            }
        }
        
        sleep(1000);  // Check every second
    }
}
```

**Failure criteria:**
```
Timeout = 5 seconds (default)

Heartbeat interval = 1 second
Expected heartbeats in timeout = 5

If 5+ consecutive heartbeats missed → Node failed
```

**False positive prevention:**
- **Generous timeout** (5× heartbeat interval)
- **Multiple missed heartbeats** required
- **Network jitter tolerance**

---

## Batch Reassignment

### Automatic Reassignment

**When node fails, reassign all its batches:**
```cpp
void HandleNodeFailure(nodeId) {
    // 1. Find batches on failed node
    batchesToReassign = findBatches(nodeId);
    
    // 2. Reassign each batch
    for (batch in batchesToReassign) {
        batch.assignedNode = -1;  // Unassign
        batch.isProcessing = false;
        
        newNode = SelectNodeForBatch(batch);
        
        if (newNode >= 0) {
            AssignBatchToNode(batch, newNode);
        }
    }
    
    // 3. Disconnect failed node
    networkManager.DisconnectNode(nodeId);
}
```

**Reassignment strategy:**
1. **Mark batches as unassigned**
2. **Use normal load balancing** to select new nodes
3. **Preserve batch priority**
4. **Update statistics**

---

## State Recovery

### Batch State Handling

**Processing batches (lost on failure):**
```
Batch state: PROCESSING
Node fails → Batch lost

Solution:
1. Mark batch as unassigned
2. Reassign to healthy node
3. Restart processing from last known state
```

**Idle batches (safe):**
```
Batch state: IDLE
Node fails → Batch safe

Solution:
1. Simply reassign to new node
2. No state loss
```

**Completed batches (results may be lost):**
```
Batch state: COMPLETED
Node fails before sending results → Results lost

Solution:
1. Timeout waiting for results
2. Reassign and reprocess
3. OR: Use result caching (advanced)
```

---

## Worker Recovery

### Automatic Recovery

**Worker reconnects after failure:**
```cpp
void HandleHeartbeat(nodeId, msg) {
    worker = workers[nodeId];
    worker.lastHeartbeat = now();
    
    if (!worker.isHealthy) {
        // Node recovered!
        worker.isHealthy = true;
        activeWorkers++;
        
        log("Worker " + nodeId + " recovered");
    }
}
```

**Recovery process:**
1. **Worker restarts** and reconnects
2. **Sends heartbeat** to master
3. **Master detects recovery**
4. **Node marked healthy** again
5. **Eligible for new batches**

**Note:** Previous batches already reassigned, not restored

---

## Configuration

### Heartbeat Tuning

**Low latency network (LAN):**
```cpp
config.heartbeatIntervalMs = 500;   // 500ms
config.heartbeatTimeoutMs = 2000;   // 2 seconds
// Fast detection (2 seconds)
```

**High latency network (WAN):**
```cpp
config.heartbeatIntervalMs = 2000;  // 2 seconds
config.heartbeatTimeoutMs = 10000;  // 10 seconds
// Avoid false positives
```

**Unreliable network:**
```cpp
config.heartbeatIntervalMs = 1000;  // 1 second
config.heartbeatTimeoutMs = 15000;  // 15 seconds
// Very tolerant
```

**Rule of thumb:**
```
timeout = interval × 5  (minimum)
timeout = interval × 10 (recommended for WAN)
```

---

## Failure Scenarios

### Scenario 1: Worker Crash

```
Timeline:
t=0s:  Worker processing batch
t=1s:  Worker crashes (no heartbeat sent)
t=2s:  Master: "No heartbeat yet, waiting..."
t=3s:  Master: "Still no heartbeat..."
t=4s:  Master: "Still no heartbeat..."
t=5s:  Master: "Timeout! Node failed"
t=5s:  Master reassigns batches
t=6s:  New worker starts processing
```

**Total downtime:** 5-6 seconds

### Scenario 2: Network Partition

```
Timeline:
t=0s:  Network partition (worker isolated)
t=1s:  Worker sends heartbeat (doesn't reach master)
t=2s:  Worker sends heartbeat (doesn't reach master)
...
t=5s:  Master: "Timeout! Node failed"
t=5s:  Master reassigns batches
t=10s: Network recovers
t=10s: Worker heartbeat reaches master
t=10s: Master: "Node recovered"
```

**Result:** 
- Batches reassigned during partition
- Worker recovers but doesn't get old batches back
- Worker available for new batches

### Scenario 3: Master Failure

**Current implementation:**
- Master failure = **cluster failure**
- All workers lose coordination

**Future enhancement (master failover):**
```
1. Workers detect master failure
2. Elect new master (e.g., lowest node ID)
3. New master reconstructs state
4. Resume operation
```

---

## Statistics & Monitoring

### Fault Tolerance Metrics

```cpp
auto stats = manager.GetStatistics();

std::cout << "Total workers: " << stats.totalWorkers << std::endl;
std::cout << "Active workers: " << stats.activeWorkers << std::endl;
std::cout << "Failed nodes: " << stats.failedNodes << std::endl;
std::cout << "Batches failed: " << stats.batchesFailed << std::endl;
std::cout << "Batches completed: " << stats.batchesCompleted << std::endl;

// Calculate availability
float availability = (float)stats.activeWorkers / stats.totalWorkers;
std::cout << "Cluster availability: " << (availability * 100) << "%" << std::endl;
```

### Health Monitoring

```cpp
auto workers = manager.GetWorkerNodes();

for (const auto& worker : workers) {
    auto now = getCurrentTime();
    auto timeSinceHeartbeat = now - worker.lastHeartbeat;
    
    std::cout << "Node " << worker.nodeId << ": "
              << (worker.isHealthy ? "HEALTHY" : "FAILED") << ", "
              << "Last heartbeat: " << timeSinceHeartbeat << "ms ago"
              << std::endl;
}
```

---

## Best Practices

### 1. Tune Timeouts for Network

```cpp
// Measure network latency first
float avgLatency = measureLatency();

// Set timeout based on latency
config.heartbeatTimeoutMs = avgLatency * 10;
config.heartbeatIntervalMs = avgLatency * 2;
```

### 2. Monitor Failure Rate

```cpp
float failureRate = stats.failedNodes / stats.totalWorkers;

if (failureRate > 0.1) {  // > 10%
    std::cerr << "High failure rate! Check network/hardware" << std::endl;
}
```

### 3. Graceful Degradation

```cpp
if (stats.activeWorkers < 2) {
    std::cerr << "Warning: Low worker count, performance degraded" << std::endl;
}

if (stats.activeWorkers == 0) {
    std::cerr << "Critical: No workers available!" << std::endl;
    // Fall back to local processing
    manager.InitializeAsStandalone();
}
```

### 4. Batch Priority on Failure

```cpp
// High priority batches get reassigned first
void ReassignBatches(failedNodeId) {
    batches = findBatches(failedNodeId);
    
    // Sort by priority (highest first)
    sort(batches, by priority descending);
    
    for (batch in batches) {
        reassign(batch);
    }
}
```

---

## Performance Impact

### Overhead

**Heartbeat bandwidth:**
```
Message size: ~50 bytes
Interval: 1 second
Workers: 10

Bandwidth = 10 workers × 50 bytes × 1 Hz = 500 bytes/s = 0.5 KB/s

Negligible!
```

**Monitoring CPU:**
```
Check interval: 1 second
Workers: 100
Time per check: ~1ms

CPU usage = 1ms / 1000ms = 0.1%

Negligible!
```

### Failure Recovery Time

| Timeout | Detection Time | Reassignment Time | Total Downtime |
|---------|----------------|-------------------|----------------|
| 2s | 2s | 0.1s | **2.1s** |
| 5s | 5s | 0.1s | **5.1s** |
| 10s | 10s | 0.1s | **10.1s** |

**Trade-off:**
- **Short timeout:** Fast recovery, more false positives
- **Long timeout:** Fewer false positives, slower recovery

---

## Limitations & Future Work

### Current Limitations

1. **No master failover** - Master is single point of failure
2. **No state checkpointing** - Processing batches lost on failure
3. **No result caching** - Completed results may be lost
4. **No partial batch recovery** - Entire batch reprocessed

### Future Enhancements

**State Checkpointing:**
```cpp
// Periodic state snapshots
every 10 seconds:
    checkpoint = captureState();
    saveCheckpoint(checkpoint);

on failure:
    state = loadCheckpoint();
    resumeFrom(state);
```

**Result Caching:**
```cpp
// Cache results before sending
result = processBatch(batch);
cache.store(batchId, result);
sendResult(result);

on failure:
    if (cache.has(batchId)) {
        return cache.get(batchId);  // No reprocessing!
    }
```

---

## Conclusion

The fault tolerance system provides:

✅ **Automatic failure detection** (heartbeat monitoring)  
✅ **Fast recovery** (5-10 second detection)  
✅ **Automatic batch reassignment** (no manual intervention)  
✅ **Worker recovery** (automatic reintegration)  
✅ **Minimal overhead** (<1% CPU, <1 KB/s bandwidth)  
✅ **Configurable timeouts** (tune for your network)  

This ensures distributed physics simulation continues even when individual nodes fail, providing high availability and reliability.
