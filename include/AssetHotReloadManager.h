#pragma once

#include "FileWatcher.h"
#include <memory>
#include <vector>
#include <string>
#include <functional>
#include <unordered_map>

// Forward declarations
class TextureManager;
class Renderer;
class Shader;

/**
 * @class AssetHotReloadManager
 * @brief Centralized system for managing hot-reload of all engine assets
 * 
 * Provides unified interface for:
 * - Shader hot-reload
 * - Texture hot-reload  
 * - Material hot-reload
 * 
 * Enable hot-reload in editor mode for rapid iteration.
 */
class AssetHotReloadManager {
public:
    AssetHotReloadManager();
    ~AssetHotReloadManager();

    /**
     * Initialize the hot-reload system
     * @param renderer Reference to renderer for shader hot-reload
     * @param textureManager Reference to texture manager
     */
    void Initialize(Renderer* renderer, TextureManager* textureManager);

    /**
     * Enable/disable hot-reload globally
     * @param enabled True to enable hot-reload
     */
    void SetEnabled(bool enabled);
    bool IsEnabled() const { return m_Enabled; }

    /**
     * Watch shader directory for changes
     * @param shaderDirectory Path to shader directory (e.g., "shaders/")
     */
    void WatchShaderDirectory(const std::string& shaderDirectory);

    /**
     * Watch texture directory for changes
     * @param textureDirectory Path to texture directory (e.g., "assets/textures/")
     */
    void WatchTextureDirectory(const std::string& textureDirectory);

    /**
     * Update hot-reload system (call once per frame)
     */
    void Update();

    /**
     * Register callback for when any asset is reloaded
     * @param callback Invoked with asset type and path
     */
    using AssetChangeCallback = std::function<void(const std::string& assetType, const std::string& path)>;
    void SetOnAssetReloaded(AssetChangeCallback callback) { m_OnAssetReloaded = callback; }

    /**
     * Get number of watched files
     */
    size_t GetWatchedFileCount() const;

    /**
     * Clear all watches
     */
    void Clear();

    // Statistics
    uint32_t GetReloadCount() const { return m_ReloadCount; }
    void ResetReloadCount() { m_ReloadCount = 0; }

private:
    std::unique_ptr<FileWatcher> m_FileWatcher;
    Renderer* m_Renderer = nullptr;
    TextureManager* m_TextureManager = nullptr;
    
    bool m_Enabled = false;
    uint32_t m_ReloadCount = 0;
    
    AssetChangeCallback m_OnAssetReloaded;

    void OnShaderChanged(const std::string& path);
    void OnTextureChanged(const std::string& path);
};
