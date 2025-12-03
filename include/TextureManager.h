#pragma once
#pragma once

#include "Texture.h"
#include <map>
#include <string>
#include <memory>

#include <thread>
#include <mutex>
#include <queue>
#include <atomic>
#include <vector> // Added for std::vector

class TextureManager {
public:
    TextureManager();
    ~TextureManager();

    // Request texture load (Async)
    std::shared_ptr<Texture> LoadTexture(const std::string& path);

    // Get existing texture, returns nullptr if not found
    std::shared_ptr<Texture> GetTexture(const std::string& path);

    // Process pending uploads and manage memory (Call once per frame)
    void Update();

    // Clear all textures
    void Clear();
    
    // Memory Budget
    void SetMemoryBudget(size_t bytes) { m_MemoryBudget = bytes; }
    size_t GetMemoryUsage() const { return m_CurrentMemoryUsage; }

private:
    std::map<std::string, std::shared_ptr<Texture>> m_Textures;
    
    // Async Loading
    struct LoadRequest {
        std::string path;
        std::shared_ptr<Texture> texture;
    };
    
    std::queue<LoadRequest> m_LoadQueue;
    std::mutex m_QueueMutex;
    std::thread m_WorkerThread;
    std::atomic<bool> m_IsRunning;
    
    // Pending uploads (Ready to upload to GPU)
    std::vector<std::shared_ptr<Texture>> m_PendingUploads;
    std::mutex m_UploadMutex;
    
    // Memory Management
    size_t m_MemoryBudget;
    size_t m_CurrentMemoryUsage;
    
    void WorkerThreadLoop();
    void ManageMemory();
};
