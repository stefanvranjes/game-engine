#pragma once

#include "VirtualFileSystem.h"
#include <memory>
#include <map>
#include <queue>
#include <thread>
#include <mutex>
#include <atomic>
#include <functional>
#include <cstdint>

/**
 * @enum ResourcePriority
 * @brief Loading priority for resources
 */
enum class ResourcePriority {
    Critical = 0,   // Load immediately (player model, critical UI)
    High = 1,       // Load soon (nearby objects, active audio)
    Normal = 2,     // Standard priority
    Low = 3,        // Load when time permits (distant objects)
    Deferred = 4    // Load in background, can be preempted
};

/**
 * @enum ResourceState
 * @brief Current state of a resource
 */
enum class ResourceState {
    Unloaded = 0,
    Loading = 1,
    Loaded = 2,
    Unloading = 3,
    Failed = 4
};

/**
 * @class Resource
 * @brief Base class for all streamable resources
 */
class Resource {
public:
    virtual ~Resource() = default;
    
    const std::string& GetPath() const { return m_Path; }
    ResourceState GetState() const { return m_State; }
    size_t GetMemoryUsage() const { return m_MemoryUsage; }
    uint64_t GetLoadTime() const { return m_LoadTime; }
    
    /**
     * Called when async load completes
     */
    virtual bool OnLoadComplete(const std::vector<uint8_t>& data) = 0;
    
    /**
     * Called when resource is being unloaded
     */
    virtual void OnUnload() = 0;
    
    /**
     * Get reference count for resource pooling
     */
    uint32_t GetRefCount() const { return m_RefCount; }
    void AddRef() { m_RefCount++; }
    void Release() { if (m_RefCount > 0) m_RefCount--; }

protected:
    Resource(const std::string& path);
    
    std::string m_Path;
    ResourceState m_State = ResourceState::Unloaded;
    size_t m_MemoryUsage = 0;
    uint64_t m_LoadTime = 0;
    uint32_t m_RefCount = 0;
};

/**
 * @struct ResourceRequest
 * @brief Request for asynchronous resource loading
 */
struct ResourceRequest {
    std::string path;
    ResourcePriority priority = ResourcePriority::Normal;
    std::shared_ptr<Resource> resource;
    std::function<void(bool success)> callback;
    uint64_t requestTime = 0;
    
    // Comparator for priority queue
    bool operator<(const ResourceRequest& other) const {
        // Higher priority (lower enum value) comes first
        if (static_cast<int>(priority) != static_cast<int>(other.priority)) {
            return static_cast<int>(priority) > static_cast<int>(other.priority);
        }
        // If same priority, older requests come first
        return requestTime > other.requestTime;
    }
};

/**
 * @struct StreamingStatistics
 * @brief Performance metrics for resource streaming
 */
struct StreamingStatistics {
    size_t totalLoadedMemory = 0;
    size_t peakMemoryUsage = 0;
    size_t memoryBudget = 0;
    
    uint32_t resourcesLoaded = 0;
    uint32_t resourcesFailed = 0;
    uint32_t pendingRequests = 0;
    
    float averageLoadTime = 0.0f;
    float bytesPerSecond = 0.0f;
};

/**
 * @class ResourceStreamingManager
 * @brief Manages asynchronous resource loading with priority queuing and memory budgeting
 * 
 * Features:
 * - Priority-based loading queue
 * - Memory budget enforcement with LRU eviction
 * - Async I/O with worker threads
 * - Per-frame load time limits
 * - Resource pooling and reference counting
 * - Streaming statistics and profiling
 * 
 * Usage:
 * @code
 * ResourceStreamingManager rsm;
 * rsm.SetMemoryBudget(512 * 1024 * 1024); // 512 MB
 * 
 * auto myResource = std::make_shared<MyResource>("models/player.gltf");
 * rsm.RequestLoad(
 *     myResource,
 *     ResourcePriority::Critical,
 *     [](bool success) { std::cout << "Loaded! " << (success ? "OK" : "FAIL") << std::endl; }
 * );
 * 
 * // In game loop:
 * rsm.Update(deltaTime);
 * @endcode
 */
class ResourceStreamingManager {
public:
    ResourceStreamingManager();
    ~ResourceStreamingManager();
    
    /**
     * Initialize the streaming manager
     * @param vfs Virtual file system to use for I/O
     * @param maxWorkerThreads Number of background threads for I/O
     */
    void Initialize(VirtualFileSystem* vfs, uint32_t maxWorkerThreads = 4);
    
    /**
     * Request asynchronous loading of a resource
     */
    void RequestLoad(
        std::shared_ptr<Resource> resource,
        ResourcePriority priority = ResourcePriority::Normal,
        std::function<void(bool success)> callback = nullptr);
    
    /**
     * Request unloading of a resource (may be deferred)
     */
    void RequestUnload(std::shared_ptr<Resource> resource);
    
    /**
     * Unload all resources immediately
     */
    void UnloadAll();
    
    /**
     * Process pending load completions and memory management
     * Called once per frame
     * @param deltaTime Frame time in seconds
     */
    void Update(float deltaTime);
    
    /**
     * Set memory budget in bytes
     */
    void SetMemoryBudget(size_t bytes) { m_MemoryBudget = bytes; }
    
    /**
     * Set maximum time per frame for loading in milliseconds
     */
    void SetFrameTimeLimit(float milliseconds) { m_MaxFrameTimeMs = milliseconds; }
    
    /**
     * Get streaming statistics
     */
    StreamingStatistics GetStatistics() const;
    
    /**
     * Check if a specific resource is loaded
     */
    bool IsLoaded(const std::string& path) const;
    
    /**
     * Get resource by path
     */
    std::shared_ptr<Resource> GetResource(const std::string& path);
    
    /**
     * Preload a list of resources
     */
    void PreloadResources(
        const std::vector<std::string>& paths,
        ResourcePriority priority = ResourcePriority::High);
    
    /**
     * Hint that resources within a range will be needed soon
     */
    void PrefetchNearby(
        const glm::vec3& position,
        float radius,
        ResourcePriority priority = ResourcePriority::Normal);
    
    /**
     * Enable/disable loading
     */
    void SetLoadingEnabled(bool enabled) { m_LoadingEnabled = enabled; }

private:
    friend class ResourceStreamingManager; // For worker thread access
    
    struct LoadedResource {
        std::shared_ptr<Resource> resource;
        uint64_t loadTime;
        uint64_t lastAccessTime;
    };
    
    VirtualFileSystem* m_VFS = nullptr;
    
    // Loading pipeline
    std::priority_queue<ResourceRequest> m_LoadQueue;
    std::map<std::string, std::shared_ptr<Resource>> m_Resources;
    std::vector<std::shared_ptr<std::thread>> m_WorkerThreads;
    
    // Memory management
    size_t m_MemoryBudget = 512 * 1024 * 1024; // 512 MB default
    size_t m_CurrentMemoryUsage = 0;
    size_t m_PeakMemoryUsage = 0;
    
    // Frame-time limiting
    float m_MaxFrameTimeMs = 5.0f;
    float m_FrameLoadTime = 0.0f;
    
    // Statistics
    uint32_t m_LoadsCompleted = 0;
    uint32_t m_LoadsFailed = 0;
    float m_AverageLoadTime = 0.0f;
    
    // Threading
    std::atomic<bool> m_IsRunning = true;
    std::atomic<bool> m_LoadingEnabled = true;
    std::mutex m_QueueMutex;
    std::mutex m_ResourceMutex;
    
    // Worker thread function
    void WorkerThread();
    
    // Memory management
    void ManageMemory();
    void EvictLRU(size_t requiredSize);
    
    // Load completion processing
    void ProcessLoadCompletions();
};
