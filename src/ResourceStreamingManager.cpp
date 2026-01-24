#include "ResourceStreamingManager.h"
#include "VirtualFileSystem.h"
#include <iostream>
#include <algorithm>
#include <chrono>
#include <queue>
#include <glm/glm.hpp>

// ============================================================================
// Resource Implementation
// ============================================================================

Resource::Resource(const std::string& path)
    : m_Path(path), m_State(ResourceState::Unloaded) {
}

class GenericResource : public Resource {
public:
    GenericResource(const std::string& path) : Resource(path) {}
    bool OnLoadComplete(const std::vector<uint8_t>& data) override {
        m_MemoryUsage = data.size();
        m_State = ResourceState::Loaded;
        return true;
    }
    void OnUnload() override {
        m_State = ResourceState::Unloaded;
        m_MemoryUsage = 0;
    }
};

// ============================================================================
// ResourceStreamingManager Implementation
// ============================================================================

ResourceStreamingManager::ResourceStreamingManager() = default;

ResourceStreamingManager::~ResourceStreamingManager() {
    m_IsRunning = false;
    
    // Wait for worker threads
    for (auto& thread : m_WorkerThreads) {
        if (thread && thread->joinable()) {
            thread->join();
        }
    }
    
    UnloadAll();
}

void ResourceStreamingManager::Initialize(VirtualFileSystem* vfs, uint32_t maxWorkerThreads) {
    m_VFS = vfs;
    m_IsRunning = true;
    
    // Start worker threads
    for (uint32_t i = 0; i < maxWorkerThreads; ++i) {
        m_WorkerThreads.push_back(
            std::make_unique<std::thread>(&ResourceStreamingManager::WorkerThread, this));
    }
}

void ResourceStreamingManager::RequestLoad(
    std::shared_ptr<Resource> resource,
    ResourcePriority priority,
    std::function<void(bool success)> callback) {
    
    if (!resource) return;
    
    std::lock_guard<std::mutex> lock(m_QueueMutex);
    
    ResourceRequest request;
    request.path = resource->GetPath();
    request.priority = priority;
    request.resource = resource;
    request.callback = callback;
    request.requestTime = std::chrono::system_clock::now().time_since_epoch().count();
    
    m_LoadQueue.push(request);
}

void ResourceStreamingManager::RequestUnload(std::shared_ptr<Resource> resource) {
    if (!resource) return;
    
    std::lock_guard<std::mutex> lock(m_ResourceMutex);
    
    auto it = m_Resources.find(resource->GetPath());
    if (it != m_Resources.end()) {
        it->second->OnUnload();
        m_CurrentMemoryUsage -= it->second->GetMemoryUsage();
        m_Resources.erase(it);
    }
}

void ResourceStreamingManager::UnloadAll() {
    std::lock_guard<std::mutex> lock(m_ResourceMutex);
    
    for (auto& pair : m_Resources) {
        pair.second->OnUnload();
    }
    
    m_Resources.clear();
    m_CurrentMemoryUsage = 0;
}

void ResourceStreamingManager::Update(float deltaTime) {
    if (!m_LoadingEnabled) return;
    
    // Process load completions
    ProcessLoadCompletions();
    
    // Manage memory budget
    ManageMemory();
    
    // Update statistics
    m_FrameLoadTime = 0.0f;
}

StreamingStatistics ResourceStreamingManager::GetStatistics() const {
    StreamingStatistics stats;
    stats.totalLoadedMemory = m_CurrentMemoryUsage;
    stats.peakMemoryUsage = m_PeakMemoryUsage;
    stats.memoryBudget = m_MemoryBudget;
    stats.resourcesLoaded = m_LoadsCompleted;
    stats.resourcesFailed = m_LoadsFailed;
    
    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(m_QueueMutex));
    stats.pendingRequests = m_LoadQueue.size();
    
    if (m_LoadsCompleted > 0) {
        stats.averageLoadTime = m_AverageLoadTime;
    }
    
    return stats;
}

bool ResourceStreamingManager::IsLoaded(const std::string& path) const {
    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(m_ResourceMutex));
    auto it = m_Resources.find(path);
    return it != m_Resources.end() && it->second->GetState() == ResourceState::Loaded;
}

std::shared_ptr<Resource> ResourceStreamingManager::GetResource(const std::string& path) {
    std::lock_guard<std::mutex> lock(m_ResourceMutex);
    auto it = m_Resources.find(path);
    return it != m_Resources.end() ? it->second : nullptr;
}

void ResourceStreamingManager::PreloadResources(
    const std::vector<std::string>& paths,
    ResourcePriority priority) {
    
    for (const auto& path : paths) {
        // Create generic resource wrapper
        auto resource = std::make_shared<GenericResource>(path);
        RequestLoad(resource, priority);
    }
}

void ResourceStreamingManager::PrefetchNearby(
    const glm::vec3& position,
    float radius,
    ResourcePriority priority) {
    
    // This would be called from game code with camera position
    // Implementation would depend on spatial index structure
    // For now, just a placeholder for future LOD streaming
}

void ResourceStreamingManager::WorkerThread() {
    while (m_IsRunning) {
        ResourceRequest request;
        
        // Get next request from queue
        {
            std::lock_guard<std::mutex> lock(m_QueueMutex);
            
            if (m_LoadQueue.empty()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }
            
            request = m_LoadQueue.top();
            m_LoadQueue.pop();
        }
        
        if (!m_VFS) continue;
        
        // Load file asynchronously
        auto startTime = std::chrono::high_resolution_clock::now();
        
        std::vector<uint8_t> data = m_VFS->ReadFile(request.path);
        
        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        
        bool success = !data.empty();
        
        if (success && request.resource) {
            // Call resource's load completion handler
            success = request.resource->OnLoadComplete(data);
        }
        
        // Update statistics
        if (success) {
            m_LoadsCompleted++;
            
            // Update average load time
            float loadTime = static_cast<float>(duration.count());
            m_AverageLoadTime = (m_AverageLoadTime * (m_LoadsCompleted - 1) + loadTime) / m_LoadsCompleted;
            
            // Register resource
            {
                std::lock_guard<std::mutex> lock(m_ResourceMutex);
                m_Resources[request.path] = request.resource;
                m_CurrentMemoryUsage += request.resource->GetMemoryUsage();
                
                if (m_CurrentMemoryUsage > m_PeakMemoryUsage) {
                    m_PeakMemoryUsage = m_CurrentMemoryUsage;
                }
            }
        } else {
            m_LoadsFailed++;
        }
        
        // Invoke callback
        if (request.callback) {
            request.callback(success);
        }
    }
}

void ResourceStreamingManager::ManageMemory() {
    if (m_CurrentMemoryUsage <= m_MemoryBudget) {
        return;
    }
    
    // Need to free up memory
    size_t requiredSize = m_CurrentMemoryUsage - m_MemoryBudget;
    EvictLRU(requiredSize);
}

void ResourceStreamingManager::EvictLRU(size_t requiredSize) {
    std::lock_guard<std::mutex> lock(m_ResourceMutex);
    
    // Collect all resources and sort by LRU
    std::vector<std::pair<uint64_t, std::string>> lruList;
    
    for (auto& pair : m_Resources) {
        if (pair.second && pair.second->GetLoadTime() > 0) {
            lruList.push_back({pair.second->GetLoadTime(), pair.first});
        }
    }
    
    // Sort by load time (oldest first)
    std::sort(lruList.begin(), lruList.end());
    
    // Evict resources until we free enough memory
    size_t freed = 0;
    for (const auto& item : lruList) {
        if (freed >= requiredSize) break;
        
        auto it = m_Resources.find(item.second);
        if (it != m_Resources.end()) {
            size_t memSize = it->second->GetMemoryUsage();
            it->second->OnUnload();
            m_Resources.erase(it);
            m_CurrentMemoryUsage -= memSize;
            freed += memSize;
        }
    }
}

void ResourceStreamingManager::ProcessLoadCompletions() {
    // Process any pending async operations
    // This is called from main thread, safe to update resources
}

// Custom comparator for priority queue
struct ResourceRequestComparator {
    bool operator()(const ResourceRequest& a, const ResourceRequest& b) const {
        // Higher priority (lower enum value) comes first
        if (static_cast<int>(a.priority) != static_cast<int>(b.priority)) {
            return static_cast<int>(a.priority) > static_cast<int>(b.priority);
        }
        // If same priority, older requests come first
        return a.requestTime > b.requestTime;
    }
};

// Fix the priority queue declaration
// Note: We need to redefine the priority queue in the header with the correct comparator
