# Distributed Physics - Testing & Deployment Guide

## Testing Strategy

### Unit Tests
- Individual component testing
- Network message serialization
- State synchronization accuracy
- Load balancing algorithms

### Integration Tests
- Multi-node communication
- Failure scenarios
- State consistency
- Performance under load

### Performance Benchmarks
- Scalability (1-8 nodes)
- Latency measurements
- Bandwidth utilization
- Efficiency metrics

---

## Running Tests

### Prerequisites

```bash
# Install Google Test
vcpkg install gtest

# Build with tests enabled
cmake -DBUILD_TESTS=ON ..
cmake --build .
```

### Run Integration Tests

```bash
# Run all tests
./tests/DistributedPhysicsIntegrationTests

# Run specific test
./tests/DistributedPhysicsIntegrationTests --gtest_filter=DistributedPhysicsTest.MasterWorkerConnection
```

### Run Benchmarks

```bash
./tests/DistributedPhysicsBenchmarks
```

**Output:** `benchmark_results.csv`

---

## Deployment

### Single Machine (Development)

```bash
# Terminal 1 - Master
./GameEngine --distributed-master --port 8080

# Terminal 2 - Worker 1
./GameEngine --distributed-worker --master localhost:8080

# Terminal 3 - Worker 2
./GameEngine --distributed-worker --master localhost:8080
```

### Multi-Machine (Production)

**Machine 1 (Master):**
```bash
./GameEngine --distributed-master --port 8080 --bind 0.0.0.0
```

**Machine 2-N (Workers):**
```bash
./GameEngine --distributed-worker --master 192.168.1.100:8080
```

---

## Configuration

### Network Settings

```cpp
// Heartbeat
manager.SetHeartbeatConfig(1000, 5000);

// Result timeout
manager.SetResultTimeout(5000);

// Sync interval
manager.SetSyncInterval(100);  // 10 Hz
```

### Load Balancing

```cpp
manager.SetLoadBalancingStrategy(
    DistributedBatchManager::LoadBalancingStrategy::LEAST_LOADED);
manager.EnableAutoLoadBalancing(true, 1000);
```

### Fault Tolerance

```cpp
manager.EnableMasterFailover(true);
```

---

## Monitoring

### Statistics

```cpp
auto stats = manager.GetStatistics();

std::cout << "Active workers: " << stats.activeWorkers << std::endl;
std::cout << "Batches completed: " << stats.batchesCompleted << std::endl;
std::cout << "Avg latency: " << stats.avgNetworkLatencyMs << "ms" << std::endl;
```

### Worker Health

```cpp
auto workers = manager.GetWorkerNodes();

for (const auto& worker : workers) {
    std::cout << "Node " << worker.nodeId 
              << ": " << (worker.isHealthy ? "HEALTHY" : "FAILED")
              << ", Load: " << worker.currentLoad << std::endl;
}
```

---

## Troubleshooting

### Connection Issues

**Problem:** Workers can't connect to master

**Solutions:**
- Check firewall rules
- Verify port is open
- Check network connectivity
- Verify master is running

### High Latency

**Problem:** `avgNetworkLatencyMs > 100ms`

**Solutions:**
- Reduce sync frequency
- Enable delta sync
- Check network congestion
- Use faster network (10 GbE+)

### Batch Failures

**Problem:** `batchesFailed > 10%`

**Solutions:**
- Increase result timeout
- Check worker stability
- Verify GPU memory
- Monitor system resources

---

## Performance Tuning

### For LAN (Low Latency)

```cpp
heartbeatInterval = 500ms
heartbeatTimeout = 2000ms
syncInterval = 33ms  // 30 Hz
resultTimeout = 2000ms
```

### For WAN (High Latency)

```cpp
heartbeatInterval = 2000ms
heartbeatTimeout = 10000ms
syncInterval = 100ms  // 10 Hz
resultTimeout = 10000ms
```

### For Maximum Throughput

```cpp
manager.EnableDeltaSync(true);
manager.OptimizeSyncBandwidth();
manager.SetLoadBalancingStrategy(LEAST_LOADED);
manager.EnableAutoLoadBalancing(true);
```

---

## Expected Performance

### Scalability

| Nodes | Throughput | Efficiency |
|-------|------------|------------|
| 1 | 100/s | 100% |
| 2 | 190/s | 95% |
| 4 | 360/s | 90% |
| 8 | 680/s | 85% |

### Latency

| Component | Latency |
|-----------|---------|
| Network RTT | 1-10ms (LAN) |
| Serialization | 1-2ms |
| Batch processing | 10-50ms |
| Total | 15-65ms |

### Bandwidth

| Mode | Per Worker |
|------|------------|
| Full sync | 7 MB/s |
| Delta sync | 150 KB/s |
| Heartbeat | 0.5 KB/s |

---

## Production Checklist

- [ ] All integration tests passing
- [ ] Performance benchmarks meet targets
- [ ] Fault tolerance tested
- [ ] Master failover verified
- [ ] Network configured correctly
- [ ] Monitoring in place
- [ ] Backup strategy defined
- [ ] Documentation complete

---

## Support

For issues or questions:
1. Check logs for error messages
2. Review statistics for anomalies
3. Run integration tests
4. Consult documentation
