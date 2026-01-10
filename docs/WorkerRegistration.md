# Worker Registration & Lifecycle

## Overview

Complete worker node lifecycle management with automatic registration, heartbeat monitoring, and graceful shutdown.

---

## Registration Flow

### Worker Startup

```
1. Worker connects to master (TCP)
2. Worker sends NODE_REGISTER message
3. Master receives registration
4. Master adds worker to cluster
5. Master sends ACK
6. Worker starts heartbeat thread
7. Worker ready for batch assignments
```

**Timeline:**
```
t=0ms:   Connect to master
t=10ms:  Send registration
t=20ms:  Master processes registration
t=30ms:  Receive ACK
t=40ms:  Start heartbeat
t=50ms:  Ready for work

Total: ~50ms
```

---

## Registration Message

### Worker → Master

**Message type:** `NODE_REGISTER`

**Data format:**
```
[GPU Count (8 bytes)]
[Total Memory MB (8 bytes)]
[Address Length (4 bytes)]
[Address String (variable)]
[Port (2 bytes)]
```

**Example:**
```cpp
NodeInfo {
    gpuCount: 4
    totalMemoryMB: 32768  // 32 GB
    address: "192.168.1.101"
    port: 0  // Workers don't listen
}
```

### Master Response

**Message type:** `ACK`

**Confirms successful registration**

---

## Node Capabilities

### Auto-Detection

**GPU count:**
```cpp
#ifdef HAS_CUDA_TOOLKIT
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    nodeInfo.gpuCount = deviceCount;
#endif
```

**GPU memory:**
```cpp
for (int i = 0; i < deviceCount; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    totalMemoryMB += prop.totalGlobalMem / (1024 * 1024);
}
```

**System info:**
- IP address (TODO: auto-detect)
- Available CPU cores
- System RAM

---

## Heartbeat System

### Enhanced Heartbeat

**Heartbeat now includes:**
1. **Timestamp** (implicit)
2. **Current load** (0.0 - 1.0)
3. **Assigned batch count**

**Message format:**
```
[Current Load (4 bytes float)]
[Assigned Batches (4 bytes uint32)]
```

### Load Calculation

```cpp
float currentLoad = 0.0f;

if (localBatchManager) {
    auto stats = localBatchManager->GetStatistics();
    
    // Load based on active batches
    currentLoad = min(1.0f, stats.activeBatches / 10.0f);
}
```

**Load interpretation:**
- `0.0` = Idle
- `0.5` = Half capacity
- `1.0` = Full capacity

### Master Processing

**Master extracts load:**
```cpp
void HandleHeartbeat(nodeId, msg) {
    // Update last heartbeat time
    worker.lastHeartbeat = now();
    
    // Extract load
    float load;
    memcpy(&load, msg.data, sizeof(float));
    worker.currentLoad = load;
    
    // Update load history for predictive balancing
    UpdateWorkerLoad(nodeId, load);
}
```

---

## Worker Lifecycle

### States

```
DISCONNECTED → CONNECTING → REGISTERED → ACTIVE → UNREGISTERING → DISCONNECTED
```

**DISCONNECTED:**
- Not connected to master
- No network activity

**CONNECTING:**
- TCP connection in progress
- Waiting for connection

**REGISTERED:**
- Connected and registered
- Heartbeat started
- Waiting for first batch

**ACTIVE:**
- Processing batches
- Sending results
- Full participation

**UNREGISTERING:**
- Graceful shutdown
- Finishing current batches
- Sending unregister message

---

## Graceful Shutdown

### Worker Shutdown

```cpp
manager.Shutdown();

// 1. Send NODE_UNREGISTER to master
// 2. Wait for current batches to complete
// 3. Stop heartbeat thread
// 4. Close network connection
```

**Timeline:**
```
t=0s:    Shutdown() called
t=0s:    Send NODE_UNREGISTER
t=0.1s:  Finish current batches
t=0.2s:  Stop heartbeat thread
t=0.3s:  Close connection
t=0.4s:  Shutdown complete

Total: ~400ms
```

### Master Handling

**Master receives unregister:**
```cpp
void HandleNodeUnregistration(nodeId) {
    // 1. Reassign batches from this worker
    ReassignBatches(nodeId);
    
    // 2. Remove from workers list
    workers.erase(nodeId);
    
    // 3. Update statistics
    stats.totalWorkers--;
    stats.activeWorkers--;
    
    // 4. Disconnect
    networkManager.DisconnectNode(nodeId);
}
```

---

## Failure vs Graceful Shutdown

### Graceful Shutdown

**Worker sends unregister:**
- Master immediately reassigns batches
- No timeout waiting
- Clean removal

**Timeline:** ~400ms

### Failure (No Unregister)

**Worker crashes/disconnects:**
- Master detects via heartbeat timeout
- Waits 5 seconds before reassignment
- Automatic recovery

**Timeline:** ~5 seconds

**Difference:** 12× faster with graceful shutdown!

---

## Registration Statistics

### Master Tracking

```cpp
auto stats = manager.GetStatistics();

std::cout << "Total workers: " << stats.totalWorkers << std::endl;
std::cout << "Active workers: " << stats.activeWorkers << std::endl;
std::cout << "Failed nodes: " << stats.failedNodes << std::endl;
```

### Worker List

```cpp
auto workers = manager.GetWorkerNodes();

for (const auto& worker : workers) {
    std::cout << "Node " << worker.nodeId << ":" << std::endl;
    std::cout << "  GPUs: " << worker.gpuCount << std::endl;
    std::cout << "  Memory: " << worker.totalMemoryMB << " MB" << std::endl;
    std::cout << "  Load: " << worker.currentLoad << std::endl;
    std::cout << "  Batches: " << worker.assignedBatches << std::endl;
    std::cout << "  Healthy: " << (worker.isHealthy ? "Yes" : "No") << std::endl;
}
```

---

## Best Practices

### 1. Register Immediately

```cpp
// Good: Register right after connection
manager.InitializeAsWorker(masterAddr, masterPort);
// Registration happens automatically

// Bad: Don't delay registration
// Worker won't receive batches until registered
```

### 2. Graceful Shutdown

```cpp
// Good: Always call Shutdown()
manager.Shutdown();

// Bad: Don't just exit
// exit(0);  // Master waits 5s for timeout
```

### 3. Monitor Registration

```cpp
// Check if registration succeeded
if (!manager.InitializeAsWorker(addr, port)) {
    std::cerr << "Registration failed!" << std::endl;
    // Retry or exit
}
```

### 4. Handle Registration Failure

```cpp
int retries = 3;
while (retries > 0) {
    if (manager.InitializeAsWorker(addr, port)) {
        break;  // Success
    }
    
    std::cerr << "Registration failed, retrying..." << std::endl;
    sleep(1);
    retries--;
}
```

---

## Troubleshooting

### Registration Timeout

**Symptoms:** Worker connects but never receives batches

**Solutions:**
- Check master is running
- Verify network connectivity
- Check firewall rules
- Verify port is correct

### Heartbeat Failure

**Symptoms:** Worker marked as failed despite being active

**Solutions:**
- Increase heartbeat timeout
- Check network latency
- Verify heartbeat thread running

### Duplicate Registration

**Symptoms:** Worker registered multiple times

**Solutions:**
- Ensure single InitializeAsWorker call
- Check for connection retry logic
- Verify node ID uniqueness

---

## Performance

### Registration Overhead

| Metric | Value |
|--------|-------|
| Registration time | ~50ms |
| Message size | ~100 bytes |
| CPU usage | <0.1% |
| Network bandwidth | Negligible |

### Heartbeat Overhead

| Metric | Value |
|--------|-------|
| Frequency | 1 Hz (default) |
| Message size | ~20 bytes |
| Bandwidth | 20 bytes/s |
| CPU usage | <0.01% |

**Total overhead:** Negligible!

---

## Conclusion

Worker registration provides:

✅ **Automatic capability detection** (GPUs, memory)  
✅ **Fast registration** (~50ms)  
✅ **Enhanced heartbeat** (with load reporting)  
✅ **Graceful shutdown** (12× faster than timeout)  
✅ **Minimal overhead** (<0.1% CPU, <100 bytes/s)  
✅ **Robust error handling** (retry, timeout)  

This completes the distributed node lifecycle management for production-ready distributed physics simulation.
