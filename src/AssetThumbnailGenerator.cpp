#include "AssetThumbnailGenerator.h"
#include "Texture.h"
#include <GL/glew.h>
#include <filesystem>
#include <fstream>
#include <iostream>

AssetThumbnailGenerator::AssetThumbnailGenerator()
    : m_MaxCacheSize(256)
    , m_IsRunning(false)
{
    m_Stats.cachedCount = 0;
    m_Stats.diskCacheCount = 0;
    m_Stats.pendingRequests = 0;
    m_Stats.totalGenerated = 0;
    m_Stats.cacheHits = 0;
    m_Stats.cacheMisses = 0;
}

AssetThumbnailGenerator::~AssetThumbnailGenerator() {
    Shutdown();
}

bool AssetThumbnailGenerator::Initialize(const std::string& cacheDir, size_t maxCacheSize) {
    m_CacheDir = cacheDir;
    m_MaxCacheSize = maxCacheSize;

    // Create cache directory
    std::filesystem::create_directories(m_CacheDir);

    // Start worker thread
    m_IsRunning = true;
    m_WorkerThread = std::thread(&AssetThumbnailGenerator::WorkerThreadMain, this);

    return true;
}

void AssetThumbnailGenerator::Shutdown() {
    if (m_IsRunning) {
        m_IsRunning = false;
        if (m_WorkerThread.joinable()) {
            m_WorkerThread.join();
        }
    }

    // Clean up cached textures
    std::lock_guard<std::mutex> lock(m_CacheMutex);
    for (auto& pair : m_Cache) {
        if (pair.second.textureID != 0) {
            DeleteTexture(pair.second.textureID);
        }
    }
    m_Cache.clear();
}

void AssetThumbnailGenerator::RequestThumbnail(const ThumbnailRequest& request) {
    std::string cacheKey = GetCacheKey(request.assetPath, request.size);

    // Check if already cached
    {
        std::lock_guard<std::mutex> lock(m_CacheMutex);
        auto it = m_Cache.find(cacheKey);
        if (it != m_Cache.end()) {
            m_Stats.cacheHits++;
            UpdateAccessTime(cacheKey);
            if (request.callback) {
                request.callback(it->second.textureID);
            }
            return;
        }
    }

    m_Stats.cacheMisses++;

    // Try to load from disk cache
    unsigned int textureID = 0;
    if (!request.forceRegenerate && LoadFromDiskCache(request.assetPath, request.size, textureID)) {
        // Add to memory cache
        CacheEntry entry;
        entry.textureID = textureID;
        entry.assetPath = request.assetPath;
        entry.cacheFilePath = GetCacheFilePath(request.assetPath, request.size);
        entry.lastAccessTime = std::time(nullptr);
        entry.size = request.size;

        {
            std::lock_guard<std::mutex> lock(m_CacheMutex);
            m_Cache[cacheKey] = entry;
            m_Stats.cachedCount = m_Cache.size();
        }

        if (request.callback) {
            request.callback(textureID);
        }
        return;
    }

    // Queue for generation
    {
        std::lock_guard<std::mutex> lock(m_QueueMutex);
        m_RequestQueue.push(request);
        m_Stats.pendingRequests = m_RequestQueue.size();
    }
}

unsigned int AssetThumbnailGenerator::GetThumbnail(const std::string& assetPath, int size) {
    std::string cacheKey = GetCacheKey(assetPath, size);

    std::lock_guard<std::mutex> lock(m_CacheMutex);
    auto it = m_Cache.find(cacheKey);
    if (it != m_Cache.end()) {
        UpdateAccessTime(cacheKey);
        return it->second.textureID;
    }

    return 0;
}

bool AssetThumbnailGenerator::IsCached(const std::string& assetPath, int size) const {
    std::string cacheKey = GetCacheKey(assetPath, size);
    std::lock_guard<std::mutex> lock(m_CacheMutex);
    return m_Cache.find(cacheKey) != m_Cache.end();
}

void AssetThumbnailGenerator::ClearCache() {
    std::lock_guard<std::mutex> lock(m_CacheMutex);
    
    for (auto& pair : m_Cache) {
        if (pair.second.textureID != 0) {
            DeleteTexture(pair.second.textureID);
        }
    }
    
    m_Cache.clear();
    m_Stats.cachedCount = 0;
}

void AssetThumbnailGenerator::ClearAssetCache(const std::string& assetPath) {
    std::lock_guard<std::mutex> lock(m_CacheMutex);
    
    // Remove all cache entries for this asset (different sizes)
    auto it = m_Cache.begin();
    while (it != m_Cache.end()) {
        if (it->second.assetPath == assetPath) {
            if (it->second.textureID != 0) {
                DeleteTexture(it->second.textureID);
            }
            it = m_Cache.erase(it);
        } else {
            ++it;
        }
    }
    
    m_Stats.cachedCount = m_Cache.size();
}

AssetThumbnailGenerator::Statistics AssetThumbnailGenerator::GetStatistics() const {
    return m_Stats;
}

void AssetThumbnailGenerator::SetDefaultIcon(const std::string& assetType, unsigned int textureID) {
    m_DefaultIcons[assetType] = textureID;
}

unsigned int AssetThumbnailGenerator::GetDefaultIcon(const std::string& assetType) const {
    auto it = m_DefaultIcons.find(assetType);
    if (it != m_DefaultIcons.end()) {
        return it->second;
    }
    return 0;
}

void AssetThumbnailGenerator::WorkerThreadMain() {
    while (m_IsRunning) {
        ThumbnailRequest request;
        
        // Get next request
        {
            std::lock_guard<std::mutex> lock(m_QueueMutex);
            if (m_RequestQueue.empty()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }
            request = m_RequestQueue.front();
            m_RequestQueue.pop();
            m_Stats.pendingRequests = m_RequestQueue.size();
        }

        // Process request
        ProcessRequest(request);
    }
}

void AssetThumbnailGenerator::ProcessRequest(const ThumbnailRequest& request) {
    unsigned int textureID = 0;

    // Generate thumbnail based on asset type
    if (request.assetType == "texture") {
        textureID = GenerateTextureThumbnail(request.assetPath, request.size);
    } else if (request.assetType == "model") {
        textureID = GenerateModelThumbnail(request.assetPath, request.size);
    } else if (request.assetType == "prefab") {
        textureID = GeneratePrefabThumbnail(request.assetPath, request.size);
    } else if (request.assetType == "scene") {
        textureID = GenerateSceneThumbnail(request.assetPath, request.size);
    } else {
        textureID = GetFallbackIcon(request.assetType);
    }

    if (textureID == 0) {
        textureID = GetFallbackIcon(request.assetType);
    }

    // Save to disk cache
    if (textureID != 0) {
        SaveToDiskCache(request.assetPath, request.size, textureID);
    }

    // Add to memory cache
    std::string cacheKey = GetCacheKey(request.assetPath, request.size);
    CacheEntry entry;
    entry.textureID = textureID;
    entry.assetPath = request.assetPath;
    entry.cacheFilePath = GetCacheFilePath(request.assetPath, request.size);
    entry.lastAccessTime = std::time(nullptr);
    entry.size = request.size;

    {
        std::lock_guard<std::mutex> lock(m_CacheMutex);
        
        // Check if we need to evict
        if (m_Cache.size() >= m_MaxCacheSize) {
            EvictLRU();
        }

        m_Cache[cacheKey] = entry;
        m_Stats.cachedCount = m_Cache.size();
        m_Stats.totalGenerated++;
    }

    // Call callback
    if (request.callback) {
        request.callback(textureID);
    }
}

unsigned int AssetThumbnailGenerator::GenerateTextureThumbnail(const std::string& assetPath, int size) {
    // Load texture using existing Texture class
    try {
        auto texture = std::make_shared<Texture>();
        if (texture->LoadFromFile(assetPath)) {
            // TODO: Resize texture to thumbnail size
            // For now, return the original texture ID
            return texture->GetID();
        }
    } catch (const std::exception& e) {
        std::cerr << "Failed to generate texture thumbnail: " << e.what() << std::endl;
    }
    
    return 0;
}

unsigned int AssetThumbnailGenerator::GenerateModelThumbnail(const std::string& assetPath, int size) {
    // TODO: Render model to offscreen framebuffer
    // For now, return fallback icon
    return GetFallbackIcon("model");
}

unsigned int AssetThumbnailGenerator::GeneratePrefabThumbnail(const std::string& assetPath, int size) {
    // TODO: Generate prefab preview
    return GetFallbackIcon("prefab");
}

unsigned int AssetThumbnailGenerator::GenerateSceneThumbnail(const std::string& assetPath, int size) {
    // TODO: Generate scene preview
    return GetFallbackIcon("scene");
}

unsigned int AssetThumbnailGenerator::GetFallbackIcon(const std::string& assetType) {
    auto it = m_DefaultIcons.find(assetType);
    if (it != m_DefaultIcons.end()) {
        return it->second;
    }
    
    // Return generic file icon
    it = m_DefaultIcons.find("file");
    if (it != m_DefaultIcons.end()) {
        return it->second;
    }
    
    return 0;
}

std::string AssetThumbnailGenerator::GetCacheKey(const std::string& assetPath, int size) const {
    return assetPath + "_" + std::to_string(size);
}

std::string AssetThumbnailGenerator::GetCacheFilePath(const std::string& assetPath, int size) const {
    // Create a safe filename from asset path
    std::string safeName = assetPath;
    std::replace(safeName.begin(), safeName.end(), '/', '_');
    std::replace(safeName.begin(), safeName.end(), '\\', '_');
    std::replace(safeName.begin(), safeName.end(), ':', '_');
    
    return m_CacheDir + "/" + safeName + "_" + std::to_string(size) + ".png";
}

bool AssetThumbnailGenerator::LoadFromDiskCache(const std::string& assetPath, int size, unsigned int& outTextureID) {
    std::string cacheFile = GetCacheFilePath(assetPath, size);
    
    if (!std::filesystem::exists(cacheFile)) {
        return false;
    }

    // Load texture from cache file
    outTextureID = LoadTextureFromFile(cacheFile);
    return outTextureID != 0;
}

bool AssetThumbnailGenerator::SaveToDiskCache(const std::string& assetPath, int size, unsigned int textureID) {
    std::string cacheFile = GetCacheFilePath(assetPath, size);
    return SaveTextureToFile(textureID, cacheFile, size, size);
}

void AssetThumbnailGenerator::EvictLRU() {
    if (m_Cache.empty()) return;

    // Find entry with oldest access time
    auto oldest = m_Cache.begin();
    for (auto it = m_Cache.begin(); it != m_Cache.end(); ++it) {
        if (it->second.lastAccessTime < oldest->second.lastAccessTime) {
            oldest = it;
        }
    }

    // Delete texture and remove from cache
    if (oldest->second.textureID != 0) {
        DeleteTexture(oldest->second.textureID);
    }
    m_Cache.erase(oldest);
}

void AssetThumbnailGenerator::UpdateAccessTime(const std::string& cacheKey) {
    auto it = m_Cache.find(cacheKey);
    if (it != m_Cache.end()) {
        it->second.lastAccessTime = std::time(nullptr);
    }
}

unsigned int AssetThumbnailGenerator::LoadTextureFromFile(const std::string& filePath) {
    // Use existing Texture class
    auto texture = std::make_shared<Texture>();
    if (texture->LoadFromFile(filePath)) {
        return texture->GetID();
    }
    return 0;
}

bool AssetThumbnailGenerator::SaveTextureToFile(unsigned int textureID, const std::string& filePath, int width, int height) {
    // TODO: Implement texture saving
    // This would require reading pixels from GPU and saving to PNG
    return false;
}

unsigned int AssetThumbnailGenerator::CreateTextureFromData(const unsigned char* data, int width, int height, int channels) {
    GLuint textureID;
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);

    GLenum format = (channels == 4) ? GL_RGBA : GL_RGB;
    glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glBindTexture(GL_TEXTURE_2D, 0);

    return textureID;
}

void AssetThumbnailGenerator::DeleteTexture(unsigned int textureID) {
    if (textureID != 0) {
        glDeleteTextures(1, &textureID);
    }
}
