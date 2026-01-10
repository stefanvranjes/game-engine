# State Synchronization - Complete Guide

## Overview

Efficient state synchronization for distributed physics with full/delta sync, bandwidth optimization, and conflict resolution.

---

## Synchronization Modes

### 1. Full State Sync

**When to use:**
- Initial synchronization
- After node failure/recovery
- Periodic checkpoints

**Data sent:**
- Complete vertex positions
- Complete vertex velocities
- All simulation flags

**Bandwidth:**
```
Soft body: 1000 vertices
Position: 3 floats × 4 bytes = 12 bytes
Velocity: 3 floats × 4 bytes = 12 bytes
Total: 24 bytes × 1000 = 24 KB per soft body

With LZ4 compression: ~6 KB (75% reduction)
```

### 2. Delta Sync (Default)

**When to use:**
- Continuous updates
- Bandwidth-constrained networks
- High-frequency sync

**Data sent:**
- Only changed vertices
- Compressed deltas

**Bandwidth:**
```
Changed vertices: ~10% (100 vertices)
Delta encoding: 3 bytes per vertex
Total: 3 bytes × 100 = 300 bytes

95% bandwidth reduction vs full sync!
```

---

## Sync Frequency

### Configuration

```cpp
// Set sync interval
manager.SetSyncInterval(100);  // 100ms = 10 Hz

// Enable delta sync
manager.EnableDeltaSync(true);
```

### Frequency Trade-offs

| Frequency | Latency | Bandwidth | Use Case |
|-----------|---------|-----------|----------|
| 60 Hz (16ms) | 16ms | High | Real-time gameplay |
| 30 Hz (33ms) | 33ms | Medium | Standard simulation |
| 10 Hz (100ms) | 100ms | Low | Background processing |
| 1 Hz (1000ms) | 1000ms | Very low | Checkpoints only |

**Recommended:** 10-30 Hz for most applications

---

## Bandwidth Optimization

### Automatic Optimization

```cpp
manager.OptimizeSyncBandwidth();

// Automatically adjusts sync frequency to stay under limit
// Default limit: 10 MB/s
```

**Algorithm:**
```cpp
estimatedBandwidth = softBodyCount × stateSize × syncHz

if (estimatedBandwidth > maxBandwidth) {
    newSyncHz = maxBandwidth / (softBodyCount × stateSize)
    setSyncInterval(1000 / newSyncHz)
}
```

### Priority-Based Sync

**Sync high-priority objects more frequently:**

```cpp
manager.SyncHighPriorityFirst();

// High priority (8-10): Every frame
// Medium priority (4-7): Every 2 frames
// Low priority (0-3): Every 5 frames
```

**Bandwidth savings:**
```
Without priority: All objects @ 10 Hz = 100% bandwidth
With priority:
- 20% high @ 10 Hz = 20%
- 30% medium @ 5 Hz = 15%
- 50% low @ 2 Hz = 10%
Total: 45% bandwidth (55% savings!)
```

---

## Conflict Resolution

### Conflict Scenarios

**Scenario 1: Concurrent Modification**
```
t=0: Master and Worker both modify same soft body
t=1: Both send updates
t=2: Conflict detected!
```

**Scenario 2: Network Partition**
```
Partition occurs
Master and Worker diverge
Partition heals
States conflict
```

### Resolution Strategies

**1. Master Wins (Default)**
```cpp
void ResolveConflict(masterState, workerState) {
    return masterState;  // Always use master
}
```

**Pros:** Simple, consistent
**Cons:** Worker changes lost

**2. Timestamp-Based**
```cpp
void ResolveConflict(masterState, workerState) {
    if (masterState.timestamp > workerState.timestamp) {
        return masterState;
    } else {
        return workerState;
    }
}
```

**Pros:** Latest wins
**Cons:** Clock synchronization required

**3. Version Vector**
```cpp
struct VersionVector {
    map<nodeId, version> versions;
};

void ResolveConflict(masterState, workerState) {
    if (masterState.version.dominates(workerState.version)) {
        return masterState;
    } else if (workerState.version.dominates(masterState.version)) {
        return workerState;
    } else {
        // Concurrent modification - merge or use application logic
        return merge(masterState, workerState);
    }
}
```

**Pros:** Detects concurrent modifications
**Cons:** Complex, overhead

---

## Bandwidth Estimation

### Per Soft Body

**Full state:**
```
Vertices: 1000
Position: 12 bytes
Velocity: 12 bytes
Flags: 4 bytes
Total: 28 KB uncompressed

With LZ4: ~7 KB (75% compression)
```

**Delta state:**
```
Changed vertices: 10% (100)
Delta encoding: 3 bytes per vertex
Total: 300 bytes

With LZ4: ~150 bytes (50% compression)
```

### Total Bandwidth

**Example: 100 soft bodies @ 10 Hz**

**Full sync:**
```
100 bodies × 7 KB × 10 Hz = 7 MB/s
```

**Delta sync:**
```
100 bodies × 150 bytes × 10 Hz = 150 KB/s

98% reduction!
```

---

## Implementation Details

### Worker → Master Sync

**Worker sends updates:**
```cpp
void SyncSoftBodyState(softBody) {
    if (useDeltaSync && hasLastState) {
        // Send delta
        deltaData = SerializeDelta(softBody, lastState);
        sendMessage(DELTA_SYNC, deltaData, unreliable);
    } else {
        // Send full state
        fullData = SerializeFull(softBody);
        sendMessage(FULL_SYNC, fullData, reliable);
    }
    
    lastState = CaptureState(softBody);
}
```

**Reliability:**
- **Full sync:** Reliable (TCP-like)
- **Delta sync:** Unreliable (UDP-like)

**Why unreliable for deltas?**
- High frequency (10+ Hz)
- Loss acceptable (next delta will correct)
- Lower latency
- Reduced overhead

### Master → Worker Broadcast

**Master broadcasts state changes:**
```cpp
void BroadcastStateUpdate(softBody) {
    stateData = SerializeFull(softBody);
    
    for (worker : workers) {
        sendMessage(worker, FULL_SYNC, stateData, unreliable);
    }
}
```

**Use cases:**
- Soft body created/destroyed
- Major state change
- Synchronization checkpoint

---

## Performance Tuning

### Adaptive Sync Rate

**Adjust based on network conditions:**

```cpp
void AdaptiveSyncRate() {
    if (avgLatency > 100ms) {
        // High latency - reduce frequency
        setSyncInterval(200);  // 5 Hz
    } else if (avgLatency < 20ms) {
        // Low latency - increase frequency
        setSyncInterval(33);   // 30 Hz
    }
}
```

### Selective Sync

**Only sync visible/active objects:**

```cpp
void SelectiveSync() {
    for (softBody : softBodies) {
        if (softBody.isVisible() && softBody.isActive()) {
            SyncSoftBodyState(softBody);
        }
    }
}
```

**Bandwidth savings:** 50-80% (depending on visibility)

---

## Best Practices

### 1. Start with Delta Sync

```cpp
manager.EnableDeltaSync(true);
manager.SetSyncInterval(100);  // 10 Hz
```

### 2. Monitor Bandwidth

```cpp
size_t bandwidth = manager.EstimateSyncBandwidth();
std::cout << "Sync bandwidth: " << (bandwidth / 1024) << " KB/s" << std::endl;

if (bandwidth > targetBandwidth) {
    manager.OptimizeSyncBandwidth();
}
```

### 3. Use Priority-Based Sync

```cpp
// High priority for player-visible objects
manager.AddSoftBody(playerCloth, 10);

// Low priority for background objects
manager.AddSoftBody(backgroundCloth, 2);

manager.SyncHighPriorityFirst();
```

### 4. Periodic Full Sync

```cpp
// Send full state every 10 seconds as checkpoint
if (currentTime % 10000 == 0) {
    for (softBody : softBodies) {
        SendFullState(softBody);
    }
}
```

---

## Comparison

### Full vs Delta Sync

| Metric | Full Sync | Delta Sync |
|--------|-----------|------------|
| Bandwidth | 7 MB/s | 150 KB/s |
| Latency | Low | Very low |
| Reliability | High | Medium |
| CPU usage | Low | Medium |
| Packet loss tolerance | High | Low |

### Sync Frequency

| Frequency | Latency | Bandwidth | CPU |
|-----------|---------|-----------|-----|
| 60 Hz | 16ms | High | High |
| 30 Hz | 33ms | Medium | Medium |
| 10 Hz | 100ms | Low | Low |
| 1 Hz | 1000ms | Very low | Very low |

---

## Limitations & Future Work

### Current Limitations

1. **No partial vertex sync** - All vertices or none
2. **Simple conflict resolution** - Master always wins
3. **No state prediction** - No client-side prediction

### Future Enhancements

**Partial Vertex Sync:**
```cpp
// Only sync vertices that moved significantly
for (vertex : vertices) {
    if (distance(vertex, lastVertex) > threshold) {
        syncVertex(vertex);
    }
}
```

**State Prediction:**
```cpp
// Predict state between syncs
predictedState = lastState + velocity * deltaTime

// Smooth to actual state when received
smoothedState = lerp(predictedState, actualState, 0.1)
```

**Compression Improvements:**
```cpp
// Quantize positions (16-bit instead of 32-bit)
quantizedPos = (pos / worldSize) * 65535

// 50% size reduction with minimal quality loss
```

---

## Conclusion

State synchronization provides:

✅ **Efficient bandwidth usage** (98% reduction with delta sync)  
✅ **Configurable sync rate** (1-60 Hz)  
✅ **Priority-based sync** (important objects first)  
✅ **Automatic optimization** (bandwidth limiting)  
✅ **Conflict resolution** (master wins)  
✅ **Compression** (LZ4 for 75% reduction)  

This ensures distributed physics simulation stays synchronized across all nodes with minimal network overhead.
