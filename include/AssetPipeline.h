#pragma once

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <functional>
#include <filesystem>
#include <thread>
#include <queue>
#include <mutex>
#include <atomic>
#include "AssetHash.h"
#include "AssetDatabase.h"
#include "AssetConverter.h"

/**
 * @brief Asset pipeline with incremental builds and caching
 * 
 * Manages the complete asset build process with incremental rebuilding,
 * format conversion, compression, and caching. Tracks asset hashes to
 * minimize reprocessing and supports multi-threaded asset conversion.
 */
class AssetPipeline {
public:
    /**
     * @brief Pipeline configuration
     */
    struct Config {
        std::string assetSourceDir;        // Directory with source assets
        std::string assetOutputDir;        // Directory for processed assets
        std::string databasePath;          // Path to asset database file
        int maxThreads = 4;                // Number of conversion threads
        bool enableCompression = true;     // Enable asset compression
        bool enableCaching = true;         // Cache processed assets
        bool validateAssets = true;        // Verify asset integrity
        bool incrementalBuild = true;      // Only process changed assets
        bool verbose = false;              // Verbose logging
    };

    /**
     * @brief Asset processing job
     */
    struct ProcessingJob {
        std::string sourcePath;
        std::string outputPath;
        std::string assetType;
        AssetConverter::TextureConversionOptions texOptions;
        AssetConverter::MeshConversionOptions meshOptions;
        bool isHighPriority = false;
        std::function<void(const AssetConverter::ConversionResult&)> onComplete;
    };

    /**
     * @brief Pipeline statistics
     */
    struct Statistics {
        size_t totalAssets = 0;
        size_t processedAssets = 0;
        size_t failedAssets = 0;
        size_t skippedAssets = 0;
        size_t totalInputSize = 0;
        size_t totalOutputSize = 0;
        double totalTimeMs = 0.0;
        
        float GetCompressionRatio() const {
            return totalOutputSize > 0 ? (float)totalInputSize / totalOutputSize : 1.0f;
        }
        
        float GetProgress() const {
            return totalAssets > 0 ? (float)processedAssets / totalAssets : 1.0f;
        }
    };

    /**
     * @brief Initialize pipeline with configuration
     * @param config Pipeline configuration
     * @return true if successful
     */
    bool Initialize(const Config& config);

    /**
     * @brief Shutdown pipeline and stop processing
     */
    void Shutdown();

    /**
     * @brief Scan asset directory and populate database
     * @param assetDir Directory to scan
     * @return true if successful
     */
    bool ScanAssetDirectory(const std::string& assetDir);

    /**
     * @brief Process all assets or only dirty ones (incremental)
     * @param incrementalOnly If true, only process changed assets
     * @return true if all assets processed successfully
     */
    bool ProcessAssets(bool incrementalOnly = true);

    /**
     * @brief Process queued jobs and update callbacks (call once per frame)
     */
    void Update();

    /**
     * @brief Process single asset
     * @param assetPath Relative path to asset
     * @param force Force reprocessing even if not dirty
     * @return true if successful
     */
    bool ProcessAsset(const std::string& assetPath, bool force = false);

    /**
     * @brief Queue asset for asynchronous processing
     * @param job Processing job
     */
    void QueueAssetJob(const ProcessingJob& job);

    /**
     * @brief Wait for all queued jobs to complete
     * @param timeoutMs Timeout in milliseconds (0 = wait indefinitely)
     * @return true if all jobs completed
     */
    bool WaitForCompletion(int timeoutMs = 0);

    /**
     * @brief Check if pipeline is busy processing
     */
    bool IsBusy() const { return m_IsProcessing || !m_JobQueue.empty(); }

    /**
     * @brief Get current statistics
     */
    const Statistics& GetStatistics() const { return m_Statistics; }

    /**
     * @brief Clean up all processed assets
     */
    void Clean();

    /**
     * @brief Clean up specific asset group
     * @param assetType Asset type to clean
     */
    void CleanAssetType(const std::string& assetType);

    /**
     * @brief Force full rebuild (ignores cache)
     */
    void ForceFullRebuild();

    /**
     * @brief Get asset database
     */
    AssetDatabase& GetDatabase() { return m_Database; }
    const AssetDatabase& GetDatabase() const { return m_Database; }

    /**
     * @brief Set progress callback
     * @param callback Function called with (progress 0.0-1.0, description)
     */
    void SetProgressCallback(std::function<void(float, const std::string&)> callback) {
        m_ProgressCallback = callback;
    }

    /**
     * @brief Get list of supported asset types
     */
    static std::vector<std::string> GetSupportedAssetTypes();

    /**
     * @brief Convert config to/from JSON
     */
    static nlohmann::json ConfigToJson(const Config& config);
    static Config ConfigFromJson(const nlohmann::json& j);

    /**
     * @brief Get pipeline instance (singleton)
     */
    static AssetPipeline& GetInstance();

private:
    Config m_Config;
    AssetDatabase m_Database;
    Statistics m_Statistics;
    
    std::function<void(float, const std::string&)> m_ProgressCallback;
    
    // Threading
    std::vector<std::thread> m_WorkerThreads;
    std::queue<ProcessingJob> m_JobQueue;
    std::mutex m_JobQueueMutex;
    std::atomic<bool> m_IsRunning;
    std::atomic<bool> m_IsProcessing;
    std::atomic<size_t> m_ActiveJobs;

    // Private methods
    void WorkerThreadMain();
    void ReportProgress(float progress, const std::string& description);
    std::string GetAssetOutputPath(const std::string& assetPath);
    bool ShouldProcessAsset(const std::string& assetPath) const;
    void UpdateDatabase(const std::string& assetPath, const AssetConverter::ConversionResult& result);
};
