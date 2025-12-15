#pragma once
#pragma once

#include "Texture.h"
#include "FileWatcher.h"
#include <map>
#include <string>
#include <memory>

#include <thread>
#include <mutex>
#include <queue>
#include <atomic>
#include <vector> // Added for std::vector
#include <functional>

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
    
    // Quality Settings
    void SetGlobalAnisotropy(float level);
    
    // Resource Listing
    std::vector<std::string> GetTextureNames() const;
    
    // Hot-Reload Support
    void SetHotReloadEnabled(bool enabled);
    bool IsHotReloadEnabled() const { return m_HotReloadEnabled; }
    
    /**
     * Watch directory for texture changes
     * @param directory Directory to watch (e.g., "assets/textures/")
     */
    void WatchTextureDirectory(const std::string& directory);
    
    /**
     * Register callback for texture changes
     * @param callback Called when a texture is reloaded (receives path)
     */
    using TextureChangeCallback = std::function<void(const std::string& path)>;
    void SetOnTextureReloaded(TextureChangeCallback callback) { m_OnTextureReloaded = callback; }

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
    // Memory Management
    size_t m_MemoryBudget;
    size_t m_CurrentMemoryUsage;
    
    float m_GlobalAnisotropy;
    
    // Hot-Reload Support
    std::unique_ptr<FileWatcher> m_FileWatcher;
    bool m_HotReloadEnabled = false;
    TextureChangeCallback m_OnTextureReloaded;
    
    void WorkerThreadLoop();
    void ManageMemory();
    void ReloadTexture(const std::string& path);
};
