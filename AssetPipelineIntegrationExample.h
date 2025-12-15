#pragma once

/**
 * @brief Example integration of Asset Pipeline into Application
 * 
 * This file demonstrates how to integrate the asset pipeline system
 * into your game engine's Application class.
 * 
 * Add this code to Application.cpp or create a separate AssetPipelineManager
 */

#include "AssetPipeline.h"
#include <iostream>

class AssetPipelineManager {
public:
    /**
     * @brief Initialize asset pipeline during engine startup
     * @param sourceDir Directory containing source assets
     * @param outputDir Directory for processed assets
     * @param forceRebuild Force full rebuild
     * @return true if successful
     */
    static bool Initialize(const std::string& sourceDir = "assets",
                          const std::string& outputDir = "assets/.processed",
                          bool forceRebuild = false) {
        // Configure pipeline
        AssetPipeline::Config config;
        config.assetSourceDir = sourceDir;
        config.assetOutputDir = outputDir;
        config.databasePath = sourceDir + "/.database.json";
        config.maxThreads = std::thread::hardware_concurrency();
        config.enableCompression = !forceRebuild;  // Compress in release, skip in debug
        config.enableCaching = true;
        config.validateAssets = true;
        config.incrementalBuild = !forceRebuild;
        config.verbose = true;

        auto& pipeline = AssetPipeline::GetInstance();

        // Setup progress callback for UI or logging
        pipeline.SetProgressCallback([](float progress, const std::string& description) {
            std::cout << "[" << static_cast<int>(progress * 100) << "%] " << description << std::endl;
        });

        // Initialize
        if (!pipeline.Initialize(config)) {
            std::cerr << "Failed to initialize asset pipeline" << std::endl;
            return false;
        }

        std::cout << "Asset Pipeline initialized with " << config.maxThreads << " threads" << std::endl;

        // Scan assets
        if (!pipeline.ScanAssetDirectory(config.assetSourceDir)) {
            std::cerr << "Failed to scan asset directory" << std::endl;
            return false;
        }

        // Process assets
        std::cout << "Processing assets..." << std::endl;
        if (!pipeline.ProcessAssets(config.incrementalBuild)) {
            std::cerr << "Asset processing encountered issues" << std::endl;
            // Non-fatal - continue but warn
        }

        // Report statistics
        const auto& stats = pipeline.GetStatistics();
        std::cout << "\n=== Asset Pipeline Summary ===" << std::endl;
        std::cout << "Total assets: " << stats.totalAssets << std::endl;
        std::cout << "Processed: " << stats.processedAssets << std::endl;
        std::cout << "Skipped: " << stats.skippedAssets << " (unchanged)" << std::endl;
        std::cout << "Failed: " << stats.failedAssets << std::endl;
        std::cout << "Input size: " << (stats.totalInputSize / 1024.0 / 1024.0) << " MB" << std::endl;
        std::cout << "Output size: " << (stats.totalOutputSize / 1024.0 / 1024.0) << " MB" << std::endl;
        std::cout << "Compression: " << stats.GetCompressionRatio() << "x" << std::endl;
        std::cout << "Time: " << stats.totalTimeMs << " ms" << std::endl;
        std::cout << "==============================\n" << std::endl;

        return true;
    }

    /**
     * @brief Handle runtime asset changes (for development)
     * Called periodically in game loop to detect changes
     */
    static void UpdateAssetChanges() {
        auto& pipeline = AssetPipeline::GetInstance();
        auto& db = pipeline.GetDatabase();

        // In a real implementation, use file watchers:
        // - Monitor source directory with FileWatcher
        // - Mark changed assets as dirty
        // - Call pipeline.ProcessAssets(true) for incremental update

        // For now, this is a stub for future integration
    }

    /**
     * @brief Force rebuild of specific asset type
     * @param assetType Type to rebuild (texture, model, shader, etc.)
     */
    static void RebuildAssetType(const std::string& assetType) {
        auto& pipeline = AssetPipeline::GetInstance();
        auto& db = pipeline.GetDatabase();

        auto assets = db.GetAssetsByType(assetType);
        for (const auto& asset : assets) {
            db.MarkAssetDirty(asset);
        }

        std::cout << "Marked " << assets.size() << " " << assetType << " assets for reprocessing" << std::endl;
        pipeline.ProcessAssets(true);
    }

    /**
     * @brief Verify asset integrity
     * @return Vector of corrupted asset paths
     */
    static std::vector<std::string> VerifyIntegrity() {
        auto& pipeline = AssetPipeline::GetInstance();
        auto& db = pipeline.GetDatabase();

        std::cout << "Verifying asset integrity..." << std::endl;
        auto corrupted = db.VerifyIntegrity();

        if (corrupted.empty()) {
            std::cout << "All assets verified successfully" << std::endl;
        } else {
            std::cout << "Corrupted assets detected:" << std::endl;
            for (const auto& asset : corrupted) {
                std::cout << "  - " << asset << std::endl;
            }
        }

        return corrupted;
    }

    /**
     * @brief Shutdown pipeline
     */
    static void Shutdown() {
        auto& pipeline = AssetPipeline::GetInstance();
        pipeline.Shutdown();
        std::cout << "Asset Pipeline shutdown complete" << std::endl;
    }

    /**
     * @brief Get asset output path
     * @param assetPath Relative path to source asset
     * @return Full path to processed asset
     */
    static std::string GetAssetPath(const std::string& assetPath) {
        auto& pipeline = AssetPipeline::GetInstance();
        const auto& config = pipeline.GetDatabase().GetDatabasePath();
        // Extract base directory from database path
        std::string baseDir = config.substr(0, config.find_last_of("/\\"));
        return baseDir + "/.processed/" + assetPath;
    }

    /**
     * @brief Print database contents
     */
    static void PrintDatabase() {
        auto& pipeline = AssetPipeline::GetInstance();
        auto& db = pipeline.GetDatabase();

        std::cout << "\n=== Asset Database Contents ===" << std::endl;
        std::cout << "Total assets: " << db.GetAssetCount() << std::endl;
        std::cout << "Dirty assets: " << db.GetDirtyAssetCount() << std::endl;

        // Print by type
        for (const auto& type : AssetPipeline::GetSupportedAssetTypes()) {
            auto assets = db.GetAssetsByType(type);
            if (!assets.empty()) {
                std::cout << "\n" << type << " (" << assets.size() << "):" << std::endl;
                for (const auto& asset : assets) {
                    const auto* entry = db.GetAssetEntry(asset);
                    std::cout << "  - " << asset;
                    if (entry && entry->isDirty) std::cout << " [DIRTY]";
                    std::cout << std::endl;
                }
            }
        }
        std::cout << "==============================\n" << std::endl;
    }
};

/**
 * @brief Usage in Application::Initialize()
 */
inline void ExampleApplicationInitialize() {
    // In Application.cpp, during engine initialization:

    // Initialize asset pipeline
    if (!AssetPipelineManager::Initialize(
            "assets",
            "assets/.processed",
            false)) {  // false = don't force rebuild
        std::cerr << "Failed to initialize assets" << std::endl;
        // Handle error or exit
    }

    // Verify assets
    auto corrupted = AssetPipelineManager::VerifyIntegrity();
    if (!corrupted.empty()) {
        // Handle corrupted assets (restore from backup, etc.)
    }

    // Print database for debugging
    // AssetPipelineManager::PrintDatabase();

    // Continue with rest of engine initialization...
}

/**
 * @brief Usage in Application::Shutdown()
 */
inline void ExampleApplicationShutdown() {
    // Before engine shutdown:
    AssetPipelineManager::Shutdown();
}

/**
 * @brief Usage in Application::Update()
 */
inline void ExampleApplicationUpdate() {
    // In main game loop (optional - for live asset reloading):
    
    // Check for asset changes and reload
    // AssetPipelineManager::UpdateAssetChanges();

    // In development only:
    // if (ImGui::Button("Rebuild Textures")) {
    //     AssetPipelineManager::RebuildAssetType("texture");
    // }
}

/**
 * @brief Usage in command-line arguments
 */
inline int ExampleCommandLineUsage(int argc, char** argv) {
    // Support command-line arguments:
    // --process-assets [--force] [--compress]

    bool forceRebuild = false;
    bool compress = false;
    bool processAssets = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--process-assets") processAssets = true;
        if (arg == "--force") forceRebuild = true;
        if (arg == "--compress") compress = true;
    }

    if (processAssets) {
        return AssetPipelineManager::Initialize("assets", "assets/.processed", forceRebuild) ? 0 : 1;
    }

    return 0;  // Continue normal execution
}
