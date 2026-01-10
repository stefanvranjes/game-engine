# Hierarchical Distribution - Usage Guide

## Quick Start

### Setup: 3-Tier Hierarchy

**Tier 0: Global Master**
```cpp
DistributedBatchManager globalMaster;
globalMaster.InitializeAsGlobalMaster(8080);
```

**Tier 1: Regional Masters**
```cpp
// US-West Regional Master
DistributedBatchManager regionalUSWest;
regionalUSWest.InitializeAsRegionalMaster(
    "global.example.com", 8080,  // Global master
    9001,                         // Local port for workers
    "us-west"                     // Region
);

// Europe Regional Master
DistributedBatchManager regionalEurope;
regionalEurope.InitializeAsRegionalMaster(
    "global.example.com", 8080,
    9002,
    "europe"
);
```

**Tier 2: Workers**
```cpp
// Worker in US-West
DistributedBatchManager worker1;
worker1.InitializeAsWorkerInRegion(
    "regional-us-west.example.com", 9001
);

// Worker in Europe
DistributedBatchManager worker2;
worker2.InitializeAsWorkerInRegion(
    "regional-europe.example.com", 9002
);
```

---

## Architecture

```
Global Master (8080)
├── Regional Master US-West (9001)
│   ├── Worker 1
│   ├── Worker 2
│   └── Worker 3
└── Regional Master Europe (9002)
    ├── Worker 4
    ├── Worker 5
    └── Worker 6
```

---

## Benefits

### Scalability
- **Flat:** 10-20 workers max
- **Hierarchical:** 100+ workers (10 regions × 10 workers)

### Latency
- **Intra-region:** 5-10ms (workers → regional master)
- **Cross-region:** 50-150ms (regional → global)

### Fault Tolerance
- Regional master failure → Promote worker
- Global master failure → Elect new global master

---

## Use Cases

### Multi-Data Center
```
Global Master (Primary DC)
├── Regional Master DC1 → 20 workers
├── Regional Master DC2 → 20 workers
└── Regional Master DC3 → 20 workers
```

### Geographic Distribution
```
Global Master (US)
├── Regional Master US-West → 10 workers
├── Regional Master US-East → 10 workers
├── Regional Master Europe → 10 workers
└── Regional Master Asia → 10 workers
```

---

## Configuration

```cpp
// Get hierarchy info
auto config = manager.GetHierarchyConfig();
std::cout << "Role: " << (int)config.role << std::endl;
std::cout << "Tier: " << config.tier << std::endl;
std::cout << "Region: " << config.region << std::endl;
```

---

## Conclusion

Hierarchical distribution enables massive-scale distributed physics with:
- 100+ workers across multiple regions
- Low intra-region latency (<10ms)
- Geographic distribution
- Fault tolerance at all tiers
