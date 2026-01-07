#pragma once

#include <string>
#include <vector>
#include <map>
#include <queue>
#include <memory>
#include <functional>
#include <mutex>
#include <thread>
#include <atomic>

/**
 * @brief Generates and caches thumbnails for various asset types
 * 
 * Supports:
 * - Texture thumbnails (PNG, JPG, TGA, etc.)
 * - Model thumbnails (rendered preview)
 * - Prefab/Scene thumbnails (icon or preview)
 * - Default icons for unsupported types
 * 
 * Features:
 * - Asynchronous thumbnail generation
 * - Persistent disk cache
 * - Memory cache with LRU eviction
 * - Thread-safe operations
 */
class AssetThumbnailGenerator {
public:
    /**
     * @brief Thumbnail generation request
     */
    struct ThumbnailRequest {
        std::string assetPath;      // Path to asset
        std::string assetType;      // Asset type (texture, model, etc.)
        int size = 128;             // Thumbnail size in pixels
        bool forceRegenerate = false; // Force regeneration even if cached
        std::function<void(unsigned int textureID)> callback; // Completion callback
    };

    /**
     * @brief Thumbnail cache entry
     */
    struct CacheEntry {
        unsigned int textureID;     // OpenGL texture ID
        std::string assetPath;      // Asset path
        std::string cacheFilePath;  // Path to cached thumbnail file
        size_t lastAccessTime;      // For LRU eviction
        int size;                   // Thumbnail size
    };

    AssetThumbnailGenerator();
    ~AssetThumbnailGenerator();

    /**
     * @brief Initialize the thumbnail generator
     * @param cacheDir Directory for thumbnail cache
     * @param maxCacheSize Maximum number of thumbnails to keep in memory
     * @return true if successful
     */
    bool Initialize(const std::string& cacheDir, size_t maxCacheSize = 256);

    /**
     * @brief Shutdown and cleanup
     */
    void Shutdown();

    /**
     * @brief Request thumbnail generation
     * @param request Thumbnail request
     * 
     * If thumbnail is cached, callback is called immediately.
     * Otherwise, generation is queued and callback called when complete.
     */
    void RequestThumbnail(const ThumbnailRequest& request);

    /**
     * @brief Get thumbnail texture ID (synchronous)
     * @param assetPath Path to asset
     * @param size Thumbnail size
     * @return Texture ID, or 0 if not available
     * 
     * Returns immediately with cached thumbnail or 0 if not ready.
     * Use RequestThumbnail for asynchronous generation.
     */
    unsigned int GetThumbnail(const std::string& assetPath, int size = 128);

    /**
     * @brief Check if thumbnail is cached
     * @param assetPath Path to asset
     * @param size Thumbnail size
     * @return true if thumbnail is in cache
     */
    bool IsCached(const std::string& assetPath, int size = 128) const;

    /**
     * @brief Clear all cached thumbnails
     */
    void ClearCache();

    /**
     * @brief Clear thumbnails for specific asset
     * @param assetPath Path to asset
     */
    void ClearAssetCache(const std::string& assetPath);

    /**
     * @brief Get cache statistics
     */
    struct Statistics {
        size_t cachedCount;         // Number of thumbnails in memory
        size_t diskCacheCount;      // Number of thumbnails on disk
        size_t pendingRequests;     // Number of pending generation requests
        size_t totalGenerated;      // Total thumbnails generated
        size_t cacheHits;           // Cache hit count
        size_t cacheMisses;         // Cache miss count
    };

    Statistics GetStatistics() const;

    /**
     * @brief Set default icon for asset type
     * @param assetType Asset type
     * @param textureID OpenGL texture ID for icon
     */
    void SetDefaultIcon(const std::string& assetType, unsigned int textureID);

    /**
     * @brief Get default icon for asset type
     * @param assetType Asset type
     * @return Texture ID, or 0 if no default icon
     */
    unsigned int GetDefaultIcon(const std::string& assetType) const;

private:
    // Configuration
    std::string m_CacheDir;
    size_t m_MaxCacheSize;

    // Cache
    std::map<std::string, CacheEntry> m_Cache;  // Key: assetPath_size
    std::map<std::string, unsigned int> m_DefaultIcons;
    mutable std::mutex m_CacheMutex;

    // Generation queue
    std::queue<ThumbnailRequest> m_RequestQueue;
    std::mutex m_QueueMutex;
    std::thread m_WorkerThread;
    std::atomic<bool> m_IsRunning;

    // Statistics
    mutable Statistics m_Stats;

    // Private methods
    void WorkerThreadMain();
    void ProcessRequest(const ThumbnailRequest& request);
    
    unsigned int GenerateTextureThumbnail(const std::string& assetPath, int size);
    unsigned int GenerateModelThumbnail(const std::string& assetPath, int size);
    unsigned int GeneratePrefabThumbnail(const std::string& assetPath, int size);
    unsigned int GenerateSceneThumbnail(const std::string& assetPath, int size);
    unsigned int GetFallbackIcon(const std::string& assetType);

    std::string GetCacheKey(const std::string& assetPath, int size) const;
    std::string GetCacheFilePath(const std::string& assetPath, int size) const;
    
    bool LoadFromDiskCache(const std::string& assetPath, int size, unsigned int& outTextureID);
    bool SaveToDiskCache(const std::string& assetPath, int size, unsigned int textureID);
    
    void EvictLRU();
    void UpdateAccessTime(const std::string& cacheKey);

    // Utility methods
    unsigned int LoadTextureFromFile(const std::string& filePath);
    bool SaveTextureToFile(unsigned int textureID, const std::string& filePath, int width, int height);
    unsigned int CreateTextureFromData(const unsigned char* data, int width, int height, int channels);
    void DeleteTexture(unsigned int textureID);
};
