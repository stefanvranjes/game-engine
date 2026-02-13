#pragma once

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <functional>
#include <filesystem>
#include "AssetDatabase.h"
#include "FuzzyMatcher.h"

/**
 * @brief Visual asset browser with thumbnail preview and drag-and-drop support
 * 
 * Provides an ImGui-based interface for browsing, searching, and managing assets.
 * Integrates with the existing AssetDatabase and AssetPipeline infrastructure.
 * 
 * Features:
 * - Folder tree navigation
 * - Grid/list view with thumbnails
 * - Search and filtering
 * - Multi-select support
 * - Drag-and-drop for easy asset integration
 * - Context menus for asset operations
 * - Real-time asset monitoring
 */
class AssetBrowser {
public:
    /**
     * @brief Browser configuration
     */
    struct Config {
        std::string assetRootDir = "assets";           // Root directory for assets
        std::string thumbnailCacheDir = ".thumbnails"; // Cache directory for thumbnails
        int thumbnailSize = 128;                       // Thumbnail size in pixels
        bool showHiddenFiles = false;                  // Show hidden files/folders
        bool autoRefresh = true;                       // Auto-refresh on file changes
        bool enableDragDrop = true;                    // Enable drag-and-drop
        bool showFileExtensions = true;                // Show file extensions
        int maxRecentAssets = 10;                      // Number of recent assets to track
    };

    /**
     * @brief View mode for asset display
     */
    enum class ViewMode {
        Grid,   // Thumbnail grid view
        List    // Detailed list view
    };

    /**
     * @brief Sort order for assets
     */
    enum class SortOrder {
        NameAsc,
        NameDesc,
        DateAsc,
        DateDesc,
        SizeAsc,
        SizeDesc,
        TypeAsc,
        TypeDesc
    };

    /**
     * @brief Asset item information
     */
    struct AssetItem {
        std::string path;           // Relative path from asset root
        std::string name;           // Display name
        std::string type;           // Asset type (texture, model, etc.)
        std::string extension;      // File extension
        size_t fileSize;            // File size in bytes
        std::string lastModified;   // Last modification time
        bool isDirectory;           // Is this a directory?
        unsigned int thumbnailID;   // OpenGL texture ID for thumbnail
        bool thumbnailLoaded;       // Has thumbnail been generated?
    };

    AssetBrowser();
    ~AssetBrowser();

    /**
     * @brief Initialize the asset browser
     * @param config Browser configuration
     * @param database Pointer to asset database (optional, can be null)
     * @return true if successful
     */
    bool Initialize(const Config& config, AssetDatabase* database = nullptr);

    /**
     * @brief Shutdown and cleanup resources
     */
    void Shutdown();

    /**
     * @brief Render the asset browser UI
     * Called each frame to draw the ImGui interface
     */
    void Render();

    /**
     * @brief Refresh asset list from disk
     */
    void Refresh();

    /**
     * @brief Set current directory
     * @param path Relative path from asset root
     */
    void SetCurrentDirectory(const std::string& path);

    /**
     * @brief Get current directory
     * @return Current directory path
     */
    const std::string& GetCurrentDirectory() const { return m_CurrentDirectory; }

    /**
     * @brief Set view mode
     * @param mode Grid or List view
     */
    void SetViewMode(ViewMode mode) { m_ViewMode = mode; }

    /**
     * @brief Get current view mode
     */
    ViewMode GetViewMode() const { return m_ViewMode; }

    /**
     * @brief Set sort order
     * @param order Sort order
     */
    void SetSortOrder(SortOrder order);

    /**
     * @brief Get selected asset path
     * @return Path to selected asset, or empty if none selected
     */
    std::string GetSelectedAsset() const;

    /**
     * @brief Get all selected assets
     * @return Vector of selected asset paths
     */
    std::vector<std::string> GetSelectedAssets() const;

    /**
     * @brief Check if currently dragging an asset
     */
    bool IsDragging() const { return m_IsDragging; }

    /**
     * @brief Get path of asset being dragged
     */
    const std::string& GetDraggedAsset() const { return m_DraggedAsset; }

    /**
     * @brief Set search filter
     * @param filter Search string
     */
    void SetSearchFilter(const std::string& filter);

    /**
     * @brief Set type filter
     * @param type Asset type to filter (empty for all)
     */
    void SetTypeFilter(const std::string& type);

    /**
     * @brief Advanced search filters
     */
    struct SearchFilters {
        std::string namePattern;           // File name pattern (fuzzy)
        std::string typeFilter;            // Filter by type (texture, model, etc.)
        size_t minFileSize = 0;            // Minimum file size in bytes
        size_t maxFileSize = SIZE_MAX;     // Maximum file size in bytes
        bool labelsEnabled = false;        // Filter by asset labels (if supported)
        std::vector<std::string> labels;   // Required labels
        bool fuzzyMatch = true;            // Use fuzzy matching instead of exact
        float minFuzzyScore = 0.3f;        // Minimum fuzzy match score (0.0-1.0)
    };

    /**
     * @brief Set advanced search filters
     * @param filters Search filter configuration
     */
    void SetSearchFilters(const SearchFilters& filters) { m_AdvancedFilters = filters; }

    /**
     * @brief Get current advanced filters
     */
    const SearchFilters& GetSearchFilters() const { return m_AdvancedFilters; }

    /**
     * @brief Reset all filters to default
     */
    void ResetFilters();

    /**
     * @brief Filter by file size range
     * @param minBytes Minimum file size in bytes
     * @param maxBytes Maximum file size in bytes
     */
    void SetFileSizeFilter(size_t minBytes, size_t maxBytes);

    /**
     * @brief Enable/disable fuzzy matching for search
     */
    void SetFuzzySearch(bool enabled) { m_AdvancedFilters.fuzzyMatch = enabled; }

    /**
     * @brief Set minimum score threshold for fuzzy matching
     * @param score Threshold from 0.0 (loose) to 1.0 (strict)
     */
    void SetFuzzyThreshold(float score) { m_AdvancedFilters.minFuzzyScore = score; }

    /**
     * @brief Set asset selection callback
     * @param callback Function called when asset is selected
     */
    void SetSelectionCallback(std::function<void(const std::string&)> callback) {
        m_SelectionCallback = callback;
    }

    /**
     * @brief Set asset double-click callback
     * @param callback Function called when asset is double-clicked
     */
    void SetDoubleClickCallback(std::function<void(const std::string&)> callback) {
        m_DoubleClickCallback = callback;
    }

    /**
     * @brief Show/hide the browser window
     */
    void SetVisible(bool visible) { m_IsVisible = visible; }

    /**
     * @brief Check if browser is visible
     */
    bool IsVisible() const { return m_IsVisible; }

    /**
     * @brief Get configuration
     */
    const Config& GetConfig() const { return m_Config; }

private:
    // Configuration
    Config m_Config;
    AssetDatabase* m_Database;
    bool m_IsVisible;

    // Current state
    std::string m_CurrentDirectory;
    ViewMode m_ViewMode;
    SortOrder m_SortOrder;
    std::string m_SearchFilter;
    std::string m_TypeFilter;
    SearchFilters m_AdvancedFilters;  // Advanced search configuration

    // Asset items
    std::vector<AssetItem> m_CurrentAssets;
    std::vector<std::string> m_DirectoryHistory;
    std::vector<std::string> m_RecentAssets;

    // Selection
    std::vector<size_t> m_SelectedIndices;
    size_t m_LastClickedIndex;

    // Drag-and-drop
    bool m_IsDragging;
    std::string m_DraggedAsset;

    // Callbacks
    std::function<void(const std::string&)> m_SelectionCallback;
    std::function<void(const std::string&)> m_DoubleClickCallback;

    // UI state
    char m_SearchBuffer[256];
    float m_ThumbnailScale;
    bool m_ShowContextMenu;
    std::string m_ContextMenuAsset;

    // Private methods
    void RenderToolbar();
    void RenderFolderTree();
    void RenderAssetGrid();
    void RenderAssetList();
    void RenderStatusBar();
    void RenderContextMenu();

    void ScanCurrentDirectory();
    void SortAssets();
    bool MatchesFilter(const AssetItem& item) const;
    
    void SelectAsset(size_t index, bool addToSelection = false);
    void ClearSelection();
    bool IsSelected(size_t index) const;

    void NavigateUp();
    void NavigateToDirectory(const std::string& path);
    void AddToRecent(const std::string& path);

    std::string GetAssetIcon(const std::string& type) const;
    std::string FormatFileSize(size_t bytes) const;
    std::string GetRelativePath(const std::string& fullPath) const;

    // Thumbnail management
    void RequestThumbnail(AssetItem& item);
    void LoadDefaultIcons();
    std::map<std::string, unsigned int> m_DefaultIcons;
};
