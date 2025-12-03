#include "TextureManager.h"
#include <iostream>

#include "TextureManager.h"
#include <iostream>
#include <algorithm>

TextureManager::TextureManager() 
    : m_IsRunning(true)
    , m_MemoryBudget(1024 * 1024 * 512) // 512 MB default budget
    , m_CurrentMemoryUsage(0)
{
    // Start worker thread
    m_WorkerThread = std::thread(&TextureManager::WorkerThreadLoop, this);
}

TextureManager::~TextureManager() {
    m_IsRunning = false;
    if (m_WorkerThread.joinable()) {
        m_WorkerThread.join();
    }
    Clear();
}

std::shared_ptr<Texture> TextureManager::LoadTexture(const std::string& path) {
    // Check if texture is already loaded or pending
    auto it = m_Textures.find(path);
    if (it != m_Textures.end()) {
        return it->second;
    }

    // Create new texture object
    std::shared_ptr<Texture> texture = std::make_shared<Texture>();
    m_Textures[path] = texture;
    
    // Queue for async loading
    {
        std::lock_guard<std::mutex> lock(m_QueueMutex);
        m_LoadQueue.push({path, texture});
    }

    return texture;
}

std::shared_ptr<Texture> TextureManager::GetTexture(const std::string& path) {
    auto it = m_Textures.find(path);
    if (it != m_Textures.end()) {
        return it->second;
    }
    return nullptr;
}

void TextureManager::Update() {
    // Process pending uploads (Main thread)
    std::vector<std::shared_ptr<Texture>> uploads;
    {
        std::lock_guard<std::mutex> lock(m_UploadMutex);
        uploads = std::move(m_PendingUploads);
        m_PendingUploads.clear();
    }
    
    for (auto& texture : uploads) {
        if (texture->UploadToGPU()) {
            m_CurrentMemoryUsage += texture->GetMemorySize();
            // std::cout << "Texture uploaded. Usage: " << m_CurrentMemoryUsage / 1024 / 1024 << " MB" << std::endl;
        }
    }
    
    // Manage memory budget
    ManageMemory();
}

void TextureManager::ManageMemory() {
    if (m_CurrentMemoryUsage <= m_MemoryBudget) return;
    
    // Find LRU textures
    std::vector<std::pair<double, std::string>> lruList;
    
    for (auto& pair : m_Textures) {
        // Only consider loaded textures
        if (pair.second->GetState() == TextureState::Loaded) {
            lruList.push_back({pair.second->GetLastUsedTime(), pair.first});
        }
    }
    
    // Sort by time (oldest first)
    std::sort(lruList.begin(), lruList.end());
    
    // Unload until within budget
    for (auto& item : lruList) {
        if (m_CurrentMemoryUsage <= m_MemoryBudget) break;
        
        auto it = m_Textures.find(item.second);
        if (it != m_Textures.end()) {
            size_t size = it->second->GetMemorySize();
            it->second->Unload();
            m_CurrentMemoryUsage -= size;
            // std::cout << "Unloaded texture: " << item.second << " (" << size / 1024 << " KB)" << std::endl;
        }
    }
}

void TextureManager::WorkerThreadLoop() {
    while (m_IsRunning) {
        LoadRequest request;
        bool hasRequest = false;
        
        {
            std::lock_guard<std::mutex> lock(m_QueueMutex);
            if (!m_LoadQueue.empty()) {
                request = m_LoadQueue.front();
                m_LoadQueue.pop();
                hasRequest = true;
            }
        }
        
        if (hasRequest) {
            // Load data from disk (Slow operation)
            request.texture->LoadDataAsync(request.path);
            
            // Move to pending uploads
            {
                std::lock_guard<std::mutex> lock(m_UploadMutex);
                m_PendingUploads.push_back(request.texture);
            }
        } else {
            // Sleep to avoid busy waiting
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
}

void TextureManager::Clear() {
    m_Textures.clear();
    m_CurrentMemoryUsage = 0;
}
