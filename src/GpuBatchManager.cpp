#include "GpuBatchManager.h"
#include "PhysXSoftBody.h"
#include "GpuProfiler.h"
#include <algorithm>
#include <chrono>
#include <iostream>

#ifdef USE_PHYSX

struct GpuBatchManager::Impl {
    std::vector<GpuBatchManager::SoftBodyEntry> entries;
    
#ifdef HAS_CUDA_TOOLKIT
    std::vector<cudaStream_t> streams;
    std::vector<cudaEvent_t> events;
#endif
    
    BatchStats stats;
    size_t currentStreamIndex = 0;
    size_t currentFrame = 0;
    bool initialized = false;
    bool autoPriorityEnabled = false;
    
    // Adaptive priority settings
    bool adaptiveEnabled = false;
    float targetFrameTimeMs = 16.67f;  // 60 FPS default
    float frameBudgetMs = 8.0f;        // Budget for batch operations
    size_t lowPrioritySkipRate = 1;    // Process every frame initially
    size_t deferredSkipRate = 2;       // Process every other frame initially
    
    // Predictive priority settings
    bool predictiveEnabled = false;
    float predictionTimeSeconds = 0.5f;  // Predict 0.5 seconds ahead
    Vec3 lastCameraPosition;
    Vec3 cameraVelocity;
    size_t predictiveBoosted = 0;
    
    // Multi-GPU settings
    struct GpuDevice {
        int deviceId;
        std::string name;
        size_t totalMemoryMB;
        size_t freeMemoryMB;
        int computeCapability;
        std::vector<cudaStream_t> streams;
        std::vector<cudaEvent_t> events;
        size_t batchesProcessed;
        float currentLoad;
    };
    
    std::vector<GpuDevice> gpuDevices;
    GpuDistribution distribution = GpuDistribution::ROUND_ROBIN;
    bool multiGpuEnabled = false;
    size_t currentGpuIndex = 0;
    bool peerAccessEnabled = false;
    
    // Dynamic migration settings
    bool migrationEnabled = false;
    float loadThreshold = 0.3f;          // 30% load imbalance threshold
    size_t migrationInterval = 60;       // Check every 60 frames
    size_t framesSinceLastMigration = 0;
    size_t totalMigrations = 0;
    size_t migrationsThisFrame = 0;
    
    ~Impl() {
#ifdef HAS_CUDA_TOOLKIT
        for (auto stream : streams) {
            if (stream) {
                cudaStreamDestroy(stream);
            }
        }
        for (auto event : events) {
            if (event) {
                cudaEventDestroy(event);
            }
        }
#endif
    }
};

GpuBatchManager::GpuBatchManager()
    : m_Impl(std::make_unique<Impl>()) {
}

GpuBatchManager::~GpuBatchManager() {
    Shutdown();
}

void GpuBatchManager::Initialize(size_t streamCount) {
    GPU_PROFILE_SCOPE("GpuBatchManager::Initialize");
    
    if (m_Impl->initialized) {
        std::cerr << "GpuBatchManager: Already initialized" << std::endl;
        return;
    }
    
    if (streamCount == 0) {
        streamCount = DEFAULT_STREAM_COUNT;
    }
    
#ifdef HAS_CUDA_TOOLKIT
    // Create CUDA streams for concurrent operations
    m_Impl->streams.resize(streamCount);
    m_Impl->events.resize(streamCount);
    
    for (size_t i = 0; i < streamCount; ++i) {
        cudaError_t err1 = cudaStreamCreate(&m_Impl->streams[i]);
        cudaError_t err2 = cudaEventCreate(&m_Impl->events[i]);
        
        if (err1 != cudaSuccess || err2 != cudaSuccess) {
            std::cerr << "GpuBatchManager: Failed to create CUDA stream/event " << i << std::endl;
            Shutdown();
            return;
        }
    }
    
    m_Impl->stats.streamCount = streamCount;
    m_Impl->initialized = true;
    
    std::cout << "GpuBatchManager: Initialized with " << streamCount << " CUDA streams" << std::endl;
#else
    std::cout << "GpuBatchManager: CUDA not available, batch operations disabled" << std::endl;
    m_Impl->initialized = false;
#endif
}

void GpuBatchManager::Shutdown() {
    if (!m_Impl->initialized) {
        return;
    }
    
    // Synchronize before shutdown
    Synchronize();
    
    m_Impl->softBodies.clear();
    m_Impl->initialized = false;
}

void GpuBatchManager::AddSoftBody(PhysXSoftBody* softBody) {
    if (!softBody) {
        return;
    }
    
    // Check if already added
    auto it = std::find_if(m_Impl->entries.begin(), m_Impl->entries.end(),
        [softBody](const SoftBodyEntry& e) { return e.softBody == softBody; });
    
    if (it != m_Impl->entries.end()) {
        return;
    }
    
    // Add with default MEDIUM priority
    SoftBodyEntry entry;
    entry.softBody = softBody;
    entry.priority = Priority::MEDIUM;
    entry.priorityScore = 50.0f;
    entry.lastProcessedFrame = 0;
    
    // Initialize priority decay
    entry.temporaryBoost = 0.0f;
    entry.boostDecayRate = 10.0f;  // Default: 10 points per second
    entry.timeSinceBoost = 0.0f;
    
    // Initialize multi-GPU
    entry.assignedGpu = -1;  // Auto-assign
    
    m_Impl->entries.push_back(entry);
    m_Impl->stats.softBodyCount = m_Impl->entries.size();
}

void GpuBatchManager::RemoveSoftBody(PhysXSoftBody* softBody) {
    auto it = std::find_if(m_Impl->entries.begin(), m_Impl->entries.end(),
        [softBody](const SoftBodyEntry& e) { return e.softBody == softBody; });
    
    if (it != m_Impl->entries.end()) {
        m_Impl->entries.erase(it);
        m_Impl->stats.softBodyCount = m_Impl->entries.size();
    }
}

void GpuBatchManager::SetSoftBodyPriority(PhysXSoftBody* softBody, Priority priority) {
    auto it = std::find_if(m_Impl->entries.begin(), m_Impl->entries.end(),
        [softBody](const SoftBodyEntry& e) { return e.softBody == softBody; });
    
    if (it != m_Impl->entries.end()) {
        it->priority = priority;
        // Recalculate priority score will happen on next UpdatePriorities()
    }
}

void GpuBatchManager::UpdatePriorities(const Vec3& cameraPosition) {
    GPU_PROFILE_SCOPE("GpuBatchManager::UpdatePriorities");
    
    for (auto& entry : m_Impl->entries) {
        entry.priorityScore = CalculatePriorityScore(entry, cameraPosition);
    }
    
    // Sort by priority score (highest first)
    SortByPriority();
}

void GpuBatchManager::EnableAutoPriority(bool enable) {
    m_Impl->autoPriorityEnabled = enable;
}

void GpuBatchManager::EnableAdaptivePriority(bool enable, float targetFrameTimeMs) {
    m_Impl->adaptiveEnabled = enable;
    m_Impl->targetFrameTimeMs = targetFrameTimeMs;
    
    // Initialize frame budget (allocate portion of frame time to batching)
    m_Impl->frameBudgetMs = targetFrameTimeMs * 0.5f; // 50% of frame time max
    
    // Reset skip rates to defaults
    m_Impl->lowPrioritySkipRate = 1;
    m_Impl->deferredSkipRate = 2;
    
    // Update stats
    m_Impl->stats.adaptiveEnabled = enable;
    m_Impl->stats.currentFrameBudgetMs = m_Impl->frameBudgetMs;
    m_Impl->stats.lowPrioritySkipRate = m_Impl->lowPrioritySkipRate;
    m_Impl->stats.deferredSkipRate = m_Impl->deferredSkipRate;
    
    if (enable) {
        std::cout << "Adaptive priority enabled: target=" << targetFrameTimeMs 
                  << "ms, budget=" << m_Impl->frameBudgetMs << "ms" << std::endl;
    }
}

void GpuBatchManager::UpdateAdaptivePriority(float lastFrameTimeMs) {
    if (!m_Impl->adaptiveEnabled) {
        return;
    }
    
    GPU_PROFILE_SCOPE("GpuBatchManager::UpdateAdaptivePriority");
    
    // Calculate how much we're over/under budget
    float frameOverhead = lastFrameTimeMs - m_Impl->targetFrameTimeMs;
    
    // Adjust frame budget based on performance
    if (frameOverhead > 2.0f) {
        // Running slow - reduce batch budget
        m_Impl->frameBudgetMs = std::max(2.0f, m_Impl->frameBudgetMs * 0.9f);
        
        // Increase skip rates for low priority
        m_Impl->lowPrioritySkipRate = std::min(size_t(4), m_Impl->lowPrioritySkipRate + 1);
        m_Impl->deferredSkipRate = std::min(size_t(8), m_Impl->deferredSkipRate + 1);
        
    } else if (frameOverhead < -2.0f) {
        // Running fast - can afford more batch processing
        m_Impl->frameBudgetMs = std::min(m_Impl->targetFrameTimeMs * 0.6f, 
                                         m_Impl->frameBudgetMs * 1.1f);
        
        // Decrease skip rates (process more frequently)
        m_Impl->lowPrioritySkipRate = std::max(size_t(1), m_Impl->lowPrioritySkipRate - 1);
        m_Impl->deferredSkipRate = std::max(size_t(2), m_Impl->deferredSkipRate - 1);
    }
    
    // Update stats
    m_Impl->stats.currentFrameBudgetMs = m_Impl->frameBudgetMs;
    m_Impl->stats.lowPrioritySkipRate = m_Impl->lowPrioritySkipRate;
    m_Impl->stats.deferredSkipRate = m_Impl->deferredSkipRate;
}

void GpuBatchManager::EnablePredictivePriority(bool enable, float predictionTimeSeconds) {
    m_Impl->predictiveEnabled = enable;
    m_Impl->predictionTimeSeconds = predictionTimeSeconds;
    
    // Update stats
    m_Impl->stats.predictiveEnabled = enable;
    m_Impl->stats.predictionTimeSeconds = predictionTimeSeconds;
    
    if (enable) {
        std::cout << "Predictive priority enabled: prediction=" << predictionTimeSeconds 
                  << "s ahead" << std::endl;
    }
}

void GpuBatchManager::UpdatePredictivePriority(const Vec3& currentCameraPosition, 
                                                const Vec3& cameraVelocity) {
    if (!m_Impl->predictiveEnabled) {
        return;
    }
    
    GPU_PROFILE_SCOPE("GpuBatchManager::UpdatePredictivePriority");
    
    // Store camera velocity for prediction
    m_Impl->cameraVelocity = cameraVelocity;
    m_Impl->lastCameraPosition = currentCameraPosition;
    
    // Predict future camera position
    Vec3 predictedPosition = currentCameraPosition + 
                             (cameraVelocity * m_Impl->predictionTimeSeconds);
    
    size_t boostedCount = 0;
    
    // Boost priority for soft bodies that will be close to predicted position
    for (auto& entry : m_Impl->entries) {
        Vec3 sbPosition = entry.softBody->GetPosition();
        
        // Calculate distance to predicted camera position
        float predictedDistance = (sbPosition - predictedPosition).Length();
        
        // Calculate current distance
        float currentDistance = (sbPosition - currentCameraPosition).Length();
        
        // If object will be significantly closer in the future, boost priority
        if (predictedDistance < currentDistance * 0.7f) {
            // Object is approaching - boost priority
            float approachBonus = 15.0f; // Significant boost
            
            // More boost if it will be very close
            if (predictedDistance < 20.0f) {
                approachBonus = 25.0f;
            }
            
            entry.priorityScore += approachBonus;
            boostedCount++;
        }
        // If object is moving away but still close, maintain priority
        else if (currentDistance < 30.0f && predictedDistance > currentDistance) {
            // Keep priority high even though moving away (avoid pop-out)
            entry.priorityScore += 5.0f;
        }
    }
    
    m_Impl->predictiveBoosted = boostedCount;
    m_Impl->stats.predictiveBoosted = boostedCount;
    
    // Re-sort after prediction boost
    if (boostedCount > 0) {
        SortByPriority();
    }
}

void GpuBatchManager::UpdatePriorityDecay(float deltaTime) {
    GPU_PROFILE_SCOPE("GpuBatchManager::UpdatePriorityDecay");
    
    bool needsResort = false;
    
    for (auto& entry : m_Impl->entries) {
        if (entry.temporaryBoost > 0.0f) {
            // Update time since boost
            entry.timeSinceBoost += deltaTime;
            
            // Calculate decay amount
            float decayAmount = entry.boostDecayRate * deltaTime;
            
            // Apply decay
            entry.temporaryBoost = std::max(0.0f, entry.temporaryBoost - decayAmount);
            
            // Recalculate priority score with decayed boost
            // Note: Base score will be recalculated in UpdatePriorities
            // Here we just apply the remaining boost
            needsResort = true;
        }
    }
    
    // Re-sort if any boosts decayed
    if (needsResort) {
        SortByPriority();
    }
}

void GpuBatchManager::ApplyTemporaryBoost(PhysXSoftBody* softBody, float boostAmount, float decayRate) {
    auto it = std::find_if(m_Impl->entries.begin(), m_Impl->entries.end(),
        [softBody](const SoftBodyEntry& e) { return e.softBody == softBody; });
    
    if (it != m_Impl->entries.end()) {
        // Apply boost
        it->temporaryBoost = boostAmount;
        it->boostDecayRate = decayRate;
        it->timeSinceBoost = 0.0f;
        
        // Immediately add to priority score
        it->priorityScore += boostAmount;
        
        // Re-sort to reflect new priority
        SortByPriority();
    }
}

// Multi-GPU implementation methods

void GpuBatchManager::EnableMultiGpu(bool enable, const std::vector<int>& gpuIds, GpuDistribution distribution) {
    m_Impl->multiGpuEnabled = enable;
    m_Impl->distribution = distribution;
    
    if (!enable) {
        std::cout << "Multi-GPU disabled" << std::endl;
        return;
    }
    
#ifdef HAS_CUDA_TOOLKIT
    // Detect available GPUs
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    
    if (deviceCount <= 1) {
        std::cout << "Only 1 GPU detected, multi-GPU disabled" << std::endl;
        m_Impl->multiGpuEnabled = false;
        return;
    }
    
    // Use specified GPUs or all available
    std::vector<int> targetGpus;
    if (gpuIds.empty()) {
        targetGpus.resize(deviceCount);
        std::iota(targetGpus.begin(), targetGpus.end(), 0);
    } else {
        targetGpus = gpuIds;
    }
    
    // Initialize GPU devices
    for (int gpuId : targetGpus) {
        if (gpuId >= deviceCount) {
            std::cerr << "Invalid GPU ID: " << gpuId << std::endl;
            continue;
        }
        
        Impl::GpuDevice device;
        device.deviceId = gpuId;
        
        cudaSetDevice(gpuId);
        
        // Get device properties
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, gpuId);
        device.name = prop.name;
        device.computeCapability = prop.major * 10 + prop.minor;
        
        // Get memory info
        size_t free, total;
        cudaMemGetInfo(&free, &total);
        device.totalMemoryMB = total / (1024 * 1024);
        device.freeMemoryMB = free / (1024 * 1024);
        
        // Create streams for this GPU
        device.streams.resize(m_Impl->stats.streamCount);
        device.events.resize(m_Impl->stats.streamCount);
        
        for (size_t i = 0; i < m_Impl->stats.streamCount; ++i) {
            cudaStreamCreate(&device.streams[i]);
            cudaEventCreate(&device.events[i]);
        }
        
        device.batchesProcessed = 0;
        device.currentLoad = 0.0f;
        
        m_Impl->gpuDevices.push_back(device);
        
        std::cout << "GPU " << gpuId << ": " << device.name 
                  << " (" << device.totalMemoryMB << "MB, CC " 
                  << device.computeCapability << ")" << std::endl;
    }
    
    // Enable peer-to-peer access
    EnablePeerAccess();
    
    std::cout << "Multi-GPU enabled with " << m_Impl->gpuDevices.size() 
              << " GPUs using " << (distribution == GpuDistribution::ROUND_ROBIN ? "ROUND_ROBIN" :
                                   distribution == GpuDistribution::LOAD_BALANCED ? "LOAD_BALANCED" :
                                   distribution == GpuDistribution::PRIORITY_BASED ? "PRIORITY_BASED" :
                                   distribution == GpuDistribution::AUTO ? "AUTO" : "MANUAL")
              << " distribution" << std::endl;
#else
    std::cout << "CUDA not available, multi-GPU disabled" << std::endl;
    m_Impl->multiGpuEnabled = false;
#endif
}

void GpuBatchManager::EnablePeerAccess() {
#ifdef HAS_CUDA_TOOLKIT
    bool anyPeerAccess = false;
    
    for (size_t i = 0; i < m_Impl->gpuDevices.size(); ++i) {
        cudaSetDevice(m_Impl->gpuDevices[i].deviceId);
        
        for (size_t j = 0; j < m_Impl->gpuDevices.size(); ++j) {
            if (i == j) continue;
            
            int canAccess = 0;
            cudaDeviceCanAccessPeer(&canAccess, 
                                    m_Impl->gpuDevices[i].deviceId,
                                    m_Impl->gpuDevices[j].deviceId);
            
            if (canAccess) {
                cudaError_t err = cudaDeviceEnablePeerAccess(
                    m_Impl->gpuDevices[j].deviceId, 0);
                
                if (err == cudaSuccess) {
                    std::cout << "  Peer access enabled: GPU " << i 
                              << " <-> GPU " << j << std::endl;
                    anyPeerAccess = true;
                } else if (err != cudaErrorPeerAccessAlreadyEnabled) {
                    std::cerr << "  Failed to enable peer access: GPU " << i 
                              << " -> GPU " << j << std::endl;
                }
            }
        }
    }
    
    m_Impl->peerAccessEnabled = anyPeerAccess;
#endif
}

void GpuBatchManager::EnableDynamicMigration(bool enable, float loadThreshold, size_t migrationInterval) {
    m_Impl->migrationEnabled = enable;
    m_Impl->loadThreshold = loadThreshold;
    m_Impl->migrationInterval = migrationInterval;
    m_Impl->framesSinceLastMigration = 0;
    
    if (enable) {
        std::cout << "Dynamic migration enabled: threshold=" << (loadThreshold * 100) 
                  << "%, interval=" << migrationInterval << " frames" << std::endl;
    }
}

void GpuBatchManager::MigrateSoftBody(PhysXSoftBody* softBody, int targetGpu) {
    if (!m_Impl->multiGpuEnabled || targetGpu < 0 || 
        targetGpu >= static_cast<int>(m_Impl->gpuDevices.size())) {
        return;
    }
    
    auto it = std::find_if(m_Impl->entries.begin(), m_Impl->entries.end(),
        [softBody](const SoftBodyEntry& e) { return e.softBody == softBody; });
    
    if (it != m_Impl->entries.end()) {
        int oldGpu = it->assignedGpu;
        
        // Update assignment
        it->assignedGpu = targetGpu;
        
        // TODO: If peer access is enabled, could use cudaMemcpyPeer for direct transfer
        // For now, data will be transferred on next batch operation
        
        m_Impl->totalMigrations++;
        m_Impl->migrationsThisFrame++;
        
        std::cout << "Migrated soft body from GPU " << oldGpu 
                  << " to GPU " << targetGpu << std::endl;
    }
}

void GpuBatchManager::CheckAndPerformMigration() {
    if (!m_Impl->migrationEnabled || !m_Impl->multiGpuEnabled || 
        m_Impl->gpuDevices.size() < 2) {
        return;
    }
    
    m_Impl->framesSinceLastMigration++;
    m_Impl->migrationsThisFrame = 0;
    
    // Only check periodically
    if (m_Impl->framesSinceLastMigration < m_Impl->migrationInterval) {
        return;
    }
    
    m_Impl->framesSinceLastMigration = 0;
    
    GPU_PROFILE_SCOPE("CheckAndPerformMigration");
    
    // Calculate load imbalance
    float imbalance = CalculateLoadImbalance();
    
    if (imbalance < m_Impl->loadThreshold) {
        // Load is balanced, no migration needed
        return;
    }
    
    // Find most and least loaded GPUs
    int maxLoadGpu = 0;
    int minLoadGpu = 0;
    float maxLoad = 0.0f;
    float minLoad = FLT_MAX;
    
    for (size_t i = 0; i < m_Impl->gpuDevices.size(); ++i) {
        float load = m_Impl->gpuDevices[i].currentLoad;
        if (load > maxLoad) {
            maxLoad = load;
            maxLoadGpu = static_cast<int>(i);
        }
        if (load < minLoad) {
            minLoad = load;
            minLoadGpu = static_cast<int>(i);
        }
    }
    
    // Find candidates to migrate from overloaded GPU to underloaded GPU
    // Prefer migrating lower priority soft bodies
    std::vector<SoftBodyEntry*> candidates;
    
    for (auto& entry : m_Impl->entries) {
        if (entry.assignedGpu == maxLoadGpu && entry.priority <= Priority::MEDIUM) {
            candidates.push_back(&entry);
        }
    }
    
    // Sort by priority (lowest first)
    std::sort(candidates.begin(), candidates.end(),
        [](const SoftBodyEntry* a, const SoftBodyEntry* b) {
            return a->priority < b->priority;
        });
    
    // Migrate soft bodies until load is balanced
    size_t migratedCount = 0;
    const size_t maxMigrationsPerCheck = 5;  // Limit migrations per check
    
    for (auto* entry : candidates) {
        if (migratedCount >= maxMigrationsPerCheck) {
            break;
        }
        
        // Migrate to least loaded GPU
        MigrateSoftBody(entry->softBody, minLoadGpu);
        migratedCount++;
        
        // Recalculate imbalance
        imbalance = CalculateLoadImbalance();
        if (imbalance < m_Impl->loadThreshold) {
            break;  // Balanced now
        }
    }
    
    if (migratedCount > 0) {
        std::cout << "Migrated " << migratedCount << " soft bodies from GPU " 
                  << maxLoadGpu << " to GPU " << minLoadGpu 
                  << " (imbalance: " << (imbalance * 100) << "%)" << std::endl;
    }
}

float GpuBatchManager::CalculateLoadImbalance() {
    if (m_Impl->gpuDevices.size() < 2) {
        return 0.0f;
    }
    
    float maxLoad = 0.0f;
    float minLoad = FLT_MAX;
    
    for (const auto& gpu : m_Impl->gpuDevices) {
        maxLoad = std::max(maxLoad, gpu.currentLoad);
        minLoad = std::min(minLoad, gpu.currentLoad);
    }
    
    return maxLoad - minLoad;
}

void GpuBatchManager::AssignToGpu(PhysXSoftBody* softBody, int gpuId) {
    auto it = std::find_if(m_Impl->entries.begin(), m_Impl->entries.end(),
        [softBody](const SoftBodyEntry& e) { return e.softBody == softBody; });
    
    if (it != m_Impl->entries.end()) {
        it->assignedGpu = gpuId;
        std::cout << "Soft body assigned to GPU " << gpuId << std::endl;
    }
}

int GpuBatchManager::SelectGpuForBatch(const SoftBodyEntry& entry) {
    if (!m_Impl->multiGpuEnabled || m_Impl->gpuDevices.empty()) {
        return 0;
    }
    
    // Manual assignment takes precedence
    if (entry.assignedGpu >= 0 && entry.assignedGpu < static_cast<int>(m_Impl->gpuDevices.size())) {
        return entry.assignedGpu;
    }
    
    switch (m_Impl->distribution) {
        case GpuDistribution::ROUND_ROBIN: {
            // Simple round-robin
            int gpu = m_Impl->currentGpuIndex % m_Impl->gpuDevices.size();
            m_Impl->currentGpuIndex++;
            return gpu;
        }
        
        case GpuDistribution::LOAD_BALANCED: {
            // Select GPU with lowest load
            int bestGpu = 0;
            float minLoad = FLT_MAX;
            
            for (size_t i = 0; i < m_Impl->gpuDevices.size(); ++i) {
                if (m_Impl->gpuDevices[i].currentLoad < minLoad) {
                    minLoad = m_Impl->gpuDevices[i].currentLoad;
                    bestGpu = static_cast<int>(i);
                }
            }
            return bestGpu;
        }
        
        case GpuDistribution::PRIORITY_BASED: {
            // High priority on GPU 0 (usually fastest)
            if (entry.priority >= Priority::HIGH) {
                return 0;
            }
            // Distribute others across remaining GPUs
            if (m_Impl->gpuDevices.size() > 1) {
                int gpu = (m_Impl->currentGpuIndex % (m_Impl->gpuDevices.size() - 1)) + 1;
                m_Impl->currentGpuIndex++;
                return gpu;
            }
            return 0;
        }
        
        case GpuDistribution::AUTO: {
            // Automatic selection based on workload complexity and GPU capabilities
            float complexity = CalculateWorkloadComplexity(entry);
            return SelectOptimalGpu(complexity, entry.priority);
        }
        
        case GpuDistribution::MANUAL:
        default:
            return 0;
    }
}

float GpuBatchManager::CalculateWorkloadComplexity(const SoftBodyEntry& entry) {
    float complexity = 0.0f;
    
    // Base complexity from soft body properties
    PhysXSoftBody* softBody = entry.softBody;
    
    // Vertex count factor (0-40 points)
    size_t vertexCount = softBody->GetVertexCount();
    if (vertexCount > 10000) {
        complexity += 40.0f;  // Very complex
    } else if (vertexCount > 5000) {
        complexity += 30.0f;  // Complex
    } else if (vertexCount > 2000) {
        complexity += 20.0f;  // Medium
    } else if (vertexCount > 500) {
        complexity += 10.0f;  // Simple
    }
    // else: Very simple - no points
    
    // Simulation complexity (0-30 points)
    // Higher iteration counts = more complex
    if (softBody->GetSolverIterations() > 10) {
        complexity += 30.0f;
    } else if (softBody->GetSolverIterations() > 5) {
        complexity += 20.0f;
    } else {
        complexity += 10.0f;
    }
    
    // Collision complexity (0-20 points)
    if (softBody->HasCollisionMesh()) {
        complexity += 20.0f;
    } else if (softBody->HasSelfCollision()) {
        complexity += 10.0f;
    }
    
    // Tear/plasticity complexity (0-10 points)
    if (softBody->HasTearing() || softBody->HasPlasticity()) {
        complexity += 10.0f;
    }
    
    return complexity; // Range: 0-100
}

int GpuBatchManager::SelectOptimalGpu(float complexity, Priority priority) {
    if (m_Impl->gpuDevices.empty()) {
        return 0;
    }
    
    // Score each GPU based on capability and current load
    struct GpuScore {
        int gpuIndex;
        float score;
    };
    
    std::vector<GpuScore> gpuScores;
    gpuScores.reserve(m_Impl->gpuDevices.size());
    
    for (size_t i = 0; i < m_Impl->gpuDevices.size(); ++i) {
        const auto& gpu = m_Impl->gpuDevices[i];
        float score = 0.0f;
        
        // Compute capability factor (0-40 points)
        // Higher compute capability = better for complex workloads
        if (gpu.computeCapability >= 80) {
            score += 40.0f;  // Ampere or newer
        } else if (gpu.computeCapability >= 75) {
            score += 35.0f;  // Turing
        } else if (gpu.computeCapability >= 70) {
            score += 30.0f;  // Volta
        } else if (gpu.computeCapability >= 60) {
            score += 25.0f;  // Pascal
        } else {
            score += 20.0f;  // Older
        }
        
        // Available memory factor (0-30 points)
        float memoryUtilization = 1.0f - (static_cast<float>(gpu.freeMemoryMB) / gpu.totalMemoryMB);
        score += (1.0f - memoryUtilization) * 30.0f;
        
        // Current load factor (0-30 points)
        // Lower load = higher score
        score += (1.0f - gpu.currentLoad) * 30.0f;
        
        // Adjust score based on workload complexity
        // Complex workloads prefer powerful GPUs
        if (complexity > 70.0f) {
            // Very complex - strongly prefer high compute capability
            if (gpu.computeCapability >= 75) {
                score += 20.0f;
            }
        } else if (complexity < 30.0f) {
            // Simple workload - can use any GPU, prefer less loaded
            score += (1.0f - gpu.currentLoad) * 10.0f;
        }
        
        // Priority factor
        if (priority >= Priority::HIGH && i == 0) {
            // High priority gets bonus for GPU 0 (primary GPU)
            score += 15.0f;
        }
        
        gpuScores.push_back({static_cast<int>(i), score});
    }
    
    // Select GPU with highest score
    auto best = std::max_element(gpuScores.begin(), gpuScores.end(),
        [](const GpuScore& a, const GpuScore& b) {
            return a.score < b.score;
        });
    
    return best->gpuIndex;
}

GpuBatchManager::MultiGpuStats GpuBatchManager::GetMultiGpuStatistics() const {
    MultiGpuStats stats;
    stats.gpuCount = m_Impl->gpuDevices.size();
    stats.peerAccessEnabled = m_Impl->peerAccessEnabled;
    
    // Migration stats
    stats.migrationEnabled = m_Impl->migrationEnabled;
    stats.totalMigrations = m_Impl->totalMigrations;
    stats.migrationsThisFrame = m_Impl->migrationsThisFrame;
    stats.maxLoadImbalance = CalculateLoadImbalance();
    
    stats.batchesPerGpu.resize(stats.gpuCount);
    stats.loadPerGpu.resize(stats.gpuCount);
    stats.memoryUsedPerGpu.resize(stats.gpuCount);
    
    for (size_t i = 0; i < stats.gpuCount; ++i) {
        stats.batchesPerGpu[i] = m_Impl->gpuDevices[i].batchesProcessed;
        stats.loadPerGpu[i] = m_Impl->gpuDevices[i].currentLoad;
        
#ifdef HAS_CUDA_TOOLKIT
        cudaSetDevice(m_Impl->gpuDevices[i].deviceId);
        size_t free, total;
        cudaMemGetInfo(&free, &total);
        stats.memoryUsedPerGpu[i] = (total - free) / (1024 * 1024);
#else
        stats.memoryUsedPerGpu[i] = 0;
#endif
    }
    
    return stats;
}

void GpuBatchManager::ProcessMultiGpuBatches(physx::PxScene* scene) {
    GPU_PROFILE_SCOPE("ProcessMultiGpuBatches");
    
    // Group entries by assigned GPU
    std::vector<std::vector<SoftBodyEntry*>> gpuBatches(m_Impl->gpuDevices.size());
    
    for (auto& entry : m_Impl->entries) {
        int gpuId = SelectGpuForBatch(entry);
        if (gpuId >= 0 && gpuId < static_cast<int>(m_Impl->gpuDevices.size())) {
            gpuBatches[gpuId].push_back(&entry);
        }
    }
    
    // Process batches on each GPU concurrently
    for (size_t gpuIdx = 0; gpuIdx < m_Impl->gpuDevices.size(); ++gpuIdx) {
        if (gpuBatches[gpuIdx].empty()) {
            continue;
        }
        
        cudaSetDevice(m_Impl->gpuDevices[gpuIdx].deviceId);
        
        // Process batches for this GPU
        size_t totalBatches = (gpuBatches[gpuIdx].size() + MAX_BATCH_SIZE - 1) / MAX_BATCH_SIZE;
        
        for (size_t batchIdx = 0; batchIdx < totalBatches; ++batchIdx) {
            size_t startIdx = batchIdx * MAX_BATCH_SIZE;
            size_t endIdx = std::min(startIdx + MAX_BATCH_SIZE, gpuBatches[gpuIdx].size());
            
            size_t streamIdx = batchIdx % m_Impl->gpuDevices[gpuIdx].streams.size();
            cudaStream_t stream = m_Impl->gpuDevices[gpuIdx].streams[streamIdx];
            
            // TODO: Implement actual PhysX copySoftBodyData call
            // scene->copySoftBodyData(..., stream);
            
            cudaEventRecord(m_Impl->gpuDevices[gpuIdx].events[streamIdx], stream);
            
            // Update last processed frame for entries in this batch
            for (size_t i = startIdx; i < endIdx; ++i) {
                gpuBatches[gpuIdx][i]->lastProcessedFrame = m_Impl->currentFrame;
            }
            
            m_Impl->gpuDevices[gpuIdx].batchesProcessed++;
        }
        
        // Update load estimate
        m_Impl->gpuDevices[gpuIdx].currentLoad = 
            static_cast<float>(gpuBatches[gpuIdx].size()) / m_Impl->entries.size();
    }
    
    m_Impl->currentFrame++;
    
    // Check for load imbalance and perform migration if needed
    CheckAndPerformMigration();
}

void GpuBatchManager::ProcessSingleGpuBatches(physx::PxScene* scene) {
    GPU_PROFILE_SCOPE("ProcessSingleGpuBatches");
    
    // Process soft bodies in batches (already sorted by priority)
    size_t totalBatches = (m_Impl->entries.size() + MAX_BATCH_SIZE - 1) / MAX_BATCH_SIZE;
    
    for (size_t batchIdx = 0; batchIdx < totalBatches; ++batchIdx) {
        size_t startIdx = batchIdx * MAX_BATCH_SIZE;
        size_t endIdx = std::min(startIdx + MAX_BATCH_SIZE, m_Impl->entries.size());
        
        // Priority-based stream assignment
        size_t streamIdx;
        if (batchIdx == 0 && m_Impl->streams.size() > 1) {
            streamIdx = 0; // Highest priority batch gets first stream
        } else {
            streamIdx = (batchIdx % (m_Impl->streams.size() - 1)) + 1;
        }
        
        cudaStream_t stream = m_Impl->streams[streamIdx];
        
        // TODO: Implement actual PhysX copySoftBodyData call
        // scene->copySoftBodyData(..., stream);
        
        cudaEventRecord(m_Impl->events[streamIdx], stream);
        
        // Update last processed frame for entries in this batch
        for (size_t i = startIdx; i < endIdx; ++i) {
            m_Impl->entries[i].lastProcessedFrame = m_Impl->currentFrame;
        }
        
        m_Impl->stats.batchesProcessed++;
    }
    
    m_Impl->currentFrame++;
}

float GpuBatchManager::CalculatePriorityScore(const SoftBodyEntry& entry, const Vec3& cameraPosition) {
    float score = 0.0f;
    
    // Base priority from manual setting (0-80 points)
    score += static_cast<float>(entry.priority) * 20.0f;
    
    // Distance factor (0-20 points)
    Vec3 sbPosition = entry.softBody->GetPosition();
    float distance = (sbPosition - cameraPosition).Length();
    
    if (distance < 10.0f) {
        score += 20.0f;  // Very close - CRITICAL
    } else if (distance < 50.0f) {
        score += 15.0f;  // Close - HIGH
    } else if (distance < 100.0f) {
        score += 10.0f;  // Medium distance
    } else if (distance < 200.0f) {
        score += 5.0f;   // Far
    }
    // else: Very far - no bonus
    
    // Add temporary boost (decays over time)
    score += entry.temporaryBoost;
    
    return score; // Range: 0-100+ (can exceed 100 with boosts)
}

void GpuBatchManager::SortByPriority() {
    std::sort(m_Impl->entries.begin(), m_Impl->entries.end(),
        [](const SoftBodyEntry& a, const SoftBodyEntry& b) {
            return a.priorityScore > b.priorityScore; // Descending order (highest first)
        });
}

void GpuBatchManager::BatchCopyData(physx::PxScene* scene) {
    GPU_PROFILE_SCOPE("GpuBatchManager::BatchCopyData");
    
    if (!m_Impl->initialized || !scene) {
        return;
    }
    
    if (m_Impl->entries.size() < MIN_BATCH_SIZE) {
        // Not worth batching for small counts
        return;
    }
    
#ifdef HAS_CUDA_TOOLKIT
    
    // Ensure sorted by priority if auto-priority is enabled
    if (m_Impl->autoPriorityEnabled) {
        SortByPriority();
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    

    // Process soft bodies in batches (already sorted by priority)
    size_t totalBatches = (m_Impl->entries.size() + MAX_BATCH_SIZE - 1) / MAX_BATCH_SIZE;
    
    for (size_t batchIdx = 0; batchIdx < totalBatches; ++batchIdx) {
        size_t startIdx = batchIdx * MAX_BATCH_SIZE;
        size_t endIdx = std::min(startIdx + MAX_BATCH_SIZE, m_Impl->entries.size());
        size_t batchSize = endIdx - startIdx;
        
        // Priority-based stream assignment
        // First batch (highest priority) gets dedicated stream 0
        // Remaining batches share other streams
        size_t streamIdx;
        if (batchIdx == 0 && m_Impl->streams.size() > 1) {
            streamIdx = 0; // Highest priority batch gets first stream
        } else {
            streamIdx = (batchIdx % (m_Impl->streams.size() - 1)) + 1;
        }
        
        cudaStream_t stream = m_Impl->streams[streamIdx];
        
        // Prepare batch data
        // Note: PhysX copySoftBodyData requires arrays of soft body indices and buffer pointers
        // This is a simplified version - full implementation would use PhysX API
        
        // TODO: Implement actual PhysX copySoftBodyData call
        // scene->copySoftBodyData(..., stream);
        
        // Record event for synchronization
        cudaEventRecord(m_Impl->events[streamIdx], stream);
        
        // Update last processed frame for entries in this batch
        for (size_t i = startIdx; i < endIdx; ++i) {
            m_Impl->entries[i].lastProcessedFrame = m_Impl->currentFrame;
        }
        
        m_Impl->stats.batchesProcessed++;
    }
    
    m_Impl->currentFrame++;
#endif
    
    auto end = std::chrono::high_resolution_clock::now();
    m_Impl->stats.lastBatchTimeMs = 
        std::chrono::duration<float, std::milli>(end - start).count();
}

void GpuBatchManager::BatchApplyData(physx::PxScene* scene) {
    GPU_PROFILE_SCOPE("GpuBatchManager::BatchApplyData");
    
    if (!m_Impl->initialized || !scene) {
        return;
    }
    
    if (m_Impl->entries.size() < MIN_BATCH_SIZE) {
        return;
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
#ifdef HAS_CUDA_TOOLKIT
    // Process soft bodies in batches (already sorted by priority from BatchCopyData)
    size_t totalBatches = (m_Impl->entries.size() + MAX_BATCH_SIZE - 1) / MAX_BATCH_SIZE;
    
    for (size_t batchIdx = 0; batchIdx < totalBatches; ++batchIdx) {
        size_t startIdx = batchIdx * MAX_BATCH_SIZE;
        size_t endIdx = std::min(startIdx + MAX_BATCH_SIZE, m_Impl->entries.size());
        size_t batchSize = endIdx - startIdx;
        
        // Use same stream assignment as BatchCopyData
        size_t streamIdx;
        if (batchIdx == 0 && m_Impl->streams.size() > 1) {
            streamIdx = 0;
        } else {
            streamIdx = (batchIdx % (m_Impl->streams.size() - 1)) + 1;
        }
        
        cudaStream_t stream = m_Impl->streams[streamIdx];
        
        // TODO: Implement actual PhysX applySoftBodyData call
        // scene->applySoftBodyData(..., stream);
        
        // Record event for synchronization
        cudaEventRecord(m_Impl->events[streamIdx], stream);
        
        m_Impl->stats.batchesProcessed++;
    }
#endif
    
    auto end = std::chrono::high_resolution_clock::now();
    m_Impl->stats.lastBatchTimeMs += 
        std::chrono::duration<float, std::milli>(end - start).count();
}

void GpuBatchManager::Synchronize() {
    GPU_PROFILE_SCOPE("GpuBatchManager::Synchronize");
    
    if (!m_Impl->initialized) {
        return;
    }
    
#ifdef HAS_CUDA_TOOLKIT
    // Wait for all streams to complete
    for (auto stream : m_Impl->streams) {
        if (stream) {
            cudaStreamSynchronize(stream);
        }
    }
#endif
}

GpuBatchManager::BatchStats GpuBatchManager::GetStatistics() const {
    return m_Impl->stats;
}

void GpuBatchManager::ResetStatistics() {
    m_Impl->stats.batchesProcessed = 0;
    m_Impl->stats.avgBatchSizeKB = 0.0f;
    m_Impl->stats.lastBatchTimeMs = 0.0f;
    m_Impl->stats.totalDataTransferredMB = 0.0f;
}

bool GpuBatchManager::IsInitialized() const {
    return m_Impl->initialized;
}

#endif // USE_PHYSX