#pragma once

#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <chrono>
#include "GameObject.h"
#include "FuzzyMatcher.h"

/**
 * @brief Quick open dialog (Ctrl+P style) for rapid asset and object selection
 * 
 * Provides a modal dialog that allows users to:
 * - Search and open assets by name (fuzzy matching)
 * - Search and navigate to GameObjects in the scene
 * - Use keyboard shortcuts for quick navigation
 * - Display recent selections
 * 
 * Similar to VSCode's Quick Open (Ctrl+P), with features like:
 * - Type prefix filtering (@ for objects, # for asset types)
 * - Quick preview or navigation
 * - Command history
 */
class QuickOpen {
public:
    /**
     * @brief Result of quick open selection
     */
    struct Result {
        enum class Type {
            None,        // No selection
            Asset,       // Asset file
            GameObject,  // Scene object
            Command      // System command
        };

        Type resultType = Type::None;
        std::string path;                    // Asset path
        std::string name;                    // Display name
        std::shared_ptr<GameObject> object;  // GameObject reference
        float matchScore = 0.0f;             // Fuzzy match score
    };

    /**
     * @brief Quick open configuration
     */
    struct Config {
        bool showAssets = true;              // Include assets in search
        bool showObjects = true;             // Include scene objects
        int maxResults = 20;                 // Maximum results to show
        float minFuzzyScore = 0.0f;          // Minimum fuzzy match score
        bool showSearchHint = true;          // Show search type hints
        int recentItemsCount = 5;            // Number of recent items to show
    };

    QuickOpen();
    ~QuickOpen();

    /**
     * @brief Initialize the quick open dialog
     * @param config Configuration options
     */
    void Initialize(const Config& config);

    /**
     * @brief Set the root object for scene searching
     * @param rootObject Root GameObject of the scene
     */
    void SetSceneRoot(std::shared_ptr<GameObject> rootObject) { m_SceneRoot = rootObject; }

    /**
     * @brief Show/open the quick open dialog
     */
    void Show();

    /**
     * @brief Hide/close the quick open dialog
     */
    void Hide();

    /**
     * @brief Check if dialog is visible
     */
    bool IsVisible() const { return m_IsVisible; }

    /**
     * @brief Render the quick open dialog
     * Called each frame to draw ImGui interface
     * @return true if dialog is still open, false if user cancelled
     */
    bool Render();

    /**
     * @brief Get the last selected result
     */
    const Result& GetLastResult() const { return m_LastResult; }

    /**
     * @brief Set selection callback
     * @param callback Called when user makes a selection
     */
    void SetSelectionCallback(std::function<void(const Result&)> callback) {
        m_OnSelection = callback;
    }

    /**
     * @brief Set cancellation callback
     * @param callback Called when user cancels (ESC)
     */
    void SetCancellationCallback(std::function<void()> callback) {
        m_OnCancellation = callback;
    }

    /**
     * @brief Clear search history
     */
    void ClearHistory();

    /**
     * @brief Set asset root directory for asset search
     * @param assetRoot Root directory for assets
     */
    void SetAssetRoot(const std::string& assetRoot) { m_AssetRoot = assetRoot; }

    /**
     * @brief Get current configuration
     */
    const Config& GetConfig() const { return m_Config; }

    /**
     * @brief Set configuration at runtime
     */
    void SetConfig(const Config& config) { m_Config = config; }

private:
    // Configuration
    Config m_Config;
    std::string m_AssetRoot = "assets";
    std::shared_ptr<GameObject> m_SceneRoot;
    bool m_IsVisible = false;

    // Search state
    char m_SearchBuffer[256] = {};
    std::vector<Result> m_CurrentResults;
    size_t m_SelectedResultIndex = 0;

    // History tracking
    std::vector<Result> m_RecentResults;
    std::vector<std::string> m_SearchHistory;

    // UI state
    double m_LastOpenTime = 0.0;
    bool m_FocusSearchInput = true;
    bool m_HasSearched = false;
    Result m_LastResult;

    // Callbacks
    std::function<void(const Result&)> m_OnSelection;
    std::function<void()> m_OnCancellation;

    // Private rendering methods
    void RenderSearchInput();
    void RenderResultsList();
    void RenderSearchHints();
    void RenderRecentItems();

    // Search logic
    void PerformSearch(const std::string& query);
    void SearchAssets(const std::string& pattern, std::vector<Result>& results);
    void SearchGameObjects(const std::string& pattern, std::vector<Result>& results);
    void SearchGameObjectsRecursive(std::shared_ptr<GameObject> object, 
                                    const std::string& pattern, 
                                    std::vector<Result>& results);

    // Result handling
    void SelectResult(const Result& result);
    void HandleKeyInput();
    void AddToRecent(const Result& result);

    // Utility methods
    std::string ExtractSearchType(std::string& query);
    float CalculateResultScore(const Result& result, const std::string& query);
    std::string GetResultIcon(const Result& result) const;
    std::string GetResultDescription(const Result& result) const;
};
