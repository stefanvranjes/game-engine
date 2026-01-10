# Master Failover - Implementation Guide

## Overview

Raft-inspired leader election for master failover, eliminating single point of failure in distributed physics system.

---

## Architecture

### Roles

**MASTER:**
- Coordinates work distribution
- Manages worker nodes
- Single active master at any time

**WORKER:**
- Processes assigned batches
- Monitors master health
- Participates in elections

**CANDIDATE:**
- Temporary role during election
- Requests votes from other nodes
- Becomes master if wins majority

---

## Leader Election (Raft-Inspired)

### Election Trigger

**Master failure detected:**
```cpp
if (timeSinceHeartbeat > heartbeatTimeoutMs) {
    std::cerr << "Master failed! Starting election..." << std::endl;
    StartElection();
}
```

### Election Process

**1. Become Candidate**
```cpp
role = CANDIDATE
currentTerm++
votedFor = self
voteCount = 1  // Vote for self
```

**2. Request Votes**
```cpp
for (node in allNodes) {
    sendVoteRequest(node, currentTerm, candidateId)
}
```

**3. Wait for Majority**
```cpp
majority = (totalNodes / 2) + 1

if (voteCount >= majority) {
    BecomeLeader()
} else if (timeout) {
    StartElection()  // Retry
}
```

**4. Become Leader**
```cpp
role = MASTER
AnnounceLeadership()
ReconstructMasterState()
StartMasterThreads()
```

---

## Election Timeline

### Example: 5-Node Cluster

```
t=0s:  Master (Node 1) fails
t=0s:  Node 2 detects failure, becomes CANDIDATE
t=0s:  Node 2 sends vote requests (term=2)

t=0.01s: Node 3 receives request, grants vote
t=0.02s: Node 4 receives request, grants vote
t=0.03s: Node 5 receives request, grants vote

t=0.03s: Node 2 has 4 votes (including self)
t=0.03s: Majority reached (4 >= 3)
t=0.03s: Node 2 becomes MASTER

t=0.04s: Node 2 announces leadership
t=0.05s: All nodes acknowledge new master

Total election time: ~50ms
```

---

## Split Vote Handling

### Scenario: Multiple Candidates

```
5 nodes: [1, 2, 3, 4, 5]
Master (Node 1) fails

Node 2 and Node 3 both become candidates simultaneously:

Node 2 votes: [2, 4]     (2 votes)
Node 3 votes: [3, 5]     (2 votes)
Node 1: (failed)

No majority! (need 3 votes)

Solution: Random election timeout
- Node 2 timeout: 150ms
- Node 3 timeout: 250ms

Node 2 times out first, starts new election
Node 3 receives new request, votes for Node 2
Node 2 wins with 3+ votes
```

**Random timeout prevents repeated splits:**
```cpp
timeout = random(150ms, 300ms)
```

---

## State Reconstruction

### Master State

**What needs to be reconstructed:**
1. **Batch assignments** (which batches on which workers)
2. **Worker information** (load, capabilities)
3. **Soft body mappings** (soft body → batch ID)

**Reconstruction process:**
```cpp
void ReconstructMasterState() {
    // 1. Request state from all workers
    for (worker in workers) {
        sendStateRequest(worker)
    }
    
    // 2. Collect responses
    for (response in responses) {
        batches = response.assignedBatches
        load = response.currentLoad
        
        // Rebuild assignments
        for (batch in batches) {
            batchAssignments[batch.id] = batch
        }
    }
    
    // 3. Verify consistency
    validateState()
}
```

---

## Configuration

### Enable Failover

```cpp
manager.EnableMasterFailover(true);
```

**When to enable:**
- Production deployments
- High availability required
- Multiple worker nodes

**When to disable:**
- Development/testing
- Single worker setups
- Standalone mode

---

## Election Parameters

### Timeouts

**Heartbeat timeout:**
```cpp
heartbeatTimeoutMs = 5000  // 5 seconds
```

**Election timeout (random):**
```cpp
electionTimeoutMs = random(150, 300)  // 150-300ms
```

**Why random?**
- Prevents split votes
- Ensures eventual leader
- Raft recommendation

### Tuning

**Fast election (LAN):**
```cpp
heartbeatTimeoutMs = 2000   // 2 seconds
electionTimeout = (100, 200) // 100-200ms
```

**Slow election (WAN):**
```cpp
heartbeatTimeoutMs = 10000  // 10 seconds
electionTimeout = (500, 1000) // 500-1000ms
```

---

## Failure Scenarios

### Scenario 1: Master Crash

```
t=0s:  Master crashes
t=5s:  Workers detect failure (heartbeat timeout)
t=5s:  Worker 2 starts election
t=5.05s: Worker 2 wins election
t=5.1s: Worker 2 reconstructs state
t=5.2s: System fully operational

Downtime: ~5.2 seconds
```

### Scenario 2: Network Partition

```
Cluster: [M, W1, W2, W3, W4]
Partition: [M, W1] | [W2, W3, W4]

Partition 1 (minority):
- M and W1 isolated
- Cannot reach majority
- M continues but ineffective

Partition 2 (majority):
- W2, W3, W4 detect M failure
- W2 starts election
- W2 wins (3 votes >= 3 majority)
- W2 becomes new master

When partition heals:
- Old M receives leadership announcement
- Old M steps down to worker
- W2 remains master
```

**Split-brain prevention:**
- Majority vote required
- Only one partition can elect leader

### Scenario 3: Cascading Failures

```
t=0s:  Master fails
t=5s:  Worker 2 starts election
t=5.05s: Worker 2 becomes master
t=10s: Worker 2 (new master) fails!
t=15s: Worker 3 starts election
t=15.05s: Worker 3 becomes master

Multiple failovers handled automatically
```

---

## Comparison with Other Approaches

### Raft vs Paxos

| Feature | Raft | Paxos |
|---------|------|-------|
| Complexity | Simple | Complex |
| Leader election | Yes | No (multi-Paxos) |
| Log replication | Built-in | Separate |
| Understandability | High | Low |

**Why Raft-inspired:**
- Easier to understand
- Proven in production (etcd, Consul)
- Good balance of simplicity and correctness

### Raft vs Zookeeper

| Feature | Raft | Zookeeper (ZAB) |
|---------|------|-----------------|
| External dependency | No | Yes |
| Setup complexity | Low | High |
| Performance | Good | Excellent |

**Why not Zookeeper:**
- Avoid external dependency
- Simpler deployment
- Sufficient for our needs

---

## Best Practices

### 1. Odd Number of Nodes

```cpp
// Good: 3, 5, 7 nodes
totalNodes = 5
majority = 3

// Bad: 2, 4, 6 nodes
totalNodes = 4
majority = 3  // Same fault tolerance as 5!
```

**Why odd?**
- Better fault tolerance per node
- 3 nodes: tolerate 1 failure
- 5 nodes: tolerate 2 failures
- 4 nodes: tolerate 1 failure (same as 3!)

### 2. Monitor Election Frequency

```cpp
if (electionsPerHour > 10) {
    std::cerr << "Too many elections! Check network stability" << std::endl;
}
```

**Frequent elections indicate:**
- Network instability
- Timeout too aggressive
- Hardware issues

### 3. Graceful Master Shutdown

```cpp
void GracefulShutdown() {
    // Announce stepping down
    announceResignation();
    
    // Wait for new election
    sleep(electionTimeout * 2);
    
    // Shutdown
    shutdown();
}
```

### 4. State Validation

```cpp
void ValidateReconstructedState() {
    // Check for inconsistencies
    for (batch in batchAssignments) {
        if (!workerExists(batch.assignedNode)) {
            std::cerr << "Orphaned batch detected!" << std::endl;
            reassignBatch(batch);
        }
    }
}
```

---

## Performance Impact

### Election Overhead

**Normal operation:**
- No overhead (failover disabled until needed)

**During election:**
- Vote requests: ~50 bytes × N nodes
- Vote responses: ~50 bytes × N nodes
- Total: ~100 bytes × N nodes

**Example (10 nodes):**
- Total traffic: ~1 KB
- Duration: ~50-300ms
- Frequency: Only on master failure

**Negligible!**

### State Reconstruction

**Time:**
| Workers | State Size | Reconstruction Time |
|---------|------------|---------------------|
| 5 | 10 KB | ~50ms |
| 10 | 50 KB | ~100ms |
| 20 | 100 KB | ~200ms |

**Fast enough for production!**

---

## Limitations & Future Work

### Current Limitations

1. **No log replication** - State may be slightly stale
2. **No persistent storage** - State lost on full cluster failure
3. **Simple majority** - No weighted voting

### Future Enhancements

**Log Replication:**
```cpp
// Replicate all state changes
onChange(state) {
    logEntry = createLogEntry(state)
    replicateToMajority(logEntry)
}
```

**Persistent Storage:**
```cpp
// Checkpoint state to disk
every 60 seconds:
    saveStateToDisk(currentState)

on startup:
    state = loadStateFromDisk()
```

**Weighted Voting:**
```cpp
// More powerful nodes get more votes
voteWeight = nodeGpuCount + (nodeMemoryGB / 10)
```

---

## Conclusion

Master failover provides:

✅ **Automatic leader election** (Raft-inspired)  
✅ **Fast failover** (50-300ms election)  
✅ **Split-brain prevention** (majority vote)  
✅ **State reconstruction** (from workers)  
✅ **No single point of failure** (any node can become master)  
✅ **Minimal overhead** (~1 KB per election)  

This ensures distributed physics simulation continues even when the master node fails, providing true high availability.
