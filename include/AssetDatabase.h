#pragma once

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <filesystem>
#include <nlohmann/json.hpp>
#include "AssetHash.h"

/**
 * @brief Asset database for tracking metadata, dependencies, and build state
 * 
 * Maintains a JSON-based database of all imported assets, their hashes,
 * dependencies, and conversion state. Enables incremental builds by
 * tracking which assets need reprocessing.
 */
class AssetDatabase {
public:
    /**
     * @brief Asset metadata entry
     */
    struct AssetEntry {
        std::string path;                           // Relative asset path
        std::string type;                           // Asset type (texture, model, shader, etc.)
        AssetHash::Hash sourceHash;                 // Hash of source asset
        AssetHash::Hash processedHash;              // Hash of processed/converted asset
        std::string processedPath;                  // Path to converted asset
        std::vector<std::string> dependencies;      // Referenced assets
        std::string lastProcessedTime;              // Timestamp of last processing
        bool isDirty;                               // Needs reprocessing
        std::map<std::string, std::string> metadata; // Custom metadata
        
        nlohmann::json ToJson() const;
        static AssetEntry FromJson(const nlohmann::json& j);
    };

    /**
     * @brief Initialize database from file or create new
     * @param dbPath Path to database file (JSON)
     * @return true if successful
     */
    bool Initialize(const std::string& dbPath);

    /**
     * @brief Save database to disk
     * @return true if successful
     */
    bool Save();

    /**
     * @brief Add or update asset entry
     * @param entry Asset entry to store
     */
    void SetAssetEntry(const AssetEntry& entry);

    /**
     * @brief Get asset entry by path
     * @param assetPath Relative asset path
     * @return Asset entry or nullptr if not found
     */
    const AssetEntry* GetAssetEntry(const std::string& assetPath) const;

    /**
     * @brief Check if asset needs reprocessing
     * @param assetPath Relative asset path
     * @return true if asset is dirty or missing from database
     */
    bool IsAssetDirty(const std::string& assetPath);

    /**
     * @brief Mark asset as clean after processing
     * @param assetPath Relative asset path
     */
    void MarkAssetClean(const std::string& assetPath);

    /**
     * @brief Mark asset as dirty (needs reprocessing)
     * @param assetPath Relative asset path
     */
    void MarkAssetDirty(const std::string& assetPath);

    /**
     * @brief Get all dirty assets of a specific type
     * @param assetType Asset type filter (empty for all)
     * @return Vector of dirty asset paths
     */
    std::vector<std::string> GetDirtyAssets(const std::string& assetType = "");

    /**
     * @brief Get all assets of a specific type
     * @param assetType Asset type (texture, model, shader, etc.)
     * @return Vector of asset paths
     */
    std::vector<std::string> GetAssetsByType(const std::string& assetType);

    /**
     * @brief Find assets that depend on the given asset
     * @param assetPath Asset path to find dependents
     * @return Vector of dependent asset paths
     */
    std::vector<std::string> FindDependents(const std::string& assetPath);

    /**
     * @brief Register asset dependency
     * @param assetPath Path of asset with dependency
     * @param dependencyPath Path of asset being depended on
     */
    void AddDependency(const std::string& assetPath, const std::string& dependencyPath);

    /**
     * @brief Check integrity of all tracked assets
     * @return Vector of asset paths that failed integrity check
     */
    std::vector<std::string> VerifyIntegrity();

    /**
     * @brief Clear all entries (for fresh build)
     */
    void Clear();

    /**
     * @brief Get total number of tracked assets
     */
    size_t GetAssetCount() const { return m_Entries.size(); }

    /**
     * @brief Get number of dirty assets
     */
    size_t GetDirtyAssetCount() const;

    /**
     * @brief Get database file path
     */
    const std::string& GetDatabasePath() const { return m_DatabasePath; }

private:
    std::string m_DatabasePath;
    std::map<std::string, AssetEntry> m_Entries;  // Key: relative asset path
    nlohmann::json m_RootJson;
};
