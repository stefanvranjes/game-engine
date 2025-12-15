#include "AssetPipeline.h"
#include <iostream>
#include <algorithm>

// Singleton instance
static AssetPipeline* g_Instance = nullptr;

AssetPipeline& AssetPipeline::GetInstance() {
    if (!g_Instance) {
        g_Instance = new AssetPipeline();
    }
    return *g_Instance;
}

bool AssetPipeline::Initialize(const Config& config) {
    m_Config = config;
    m_IsRunning = true;
    m_IsProcessing = false;
    m_ActiveJobs = 0;

    // Initialize database
    if (!m_Database.Initialize(config.databasePath)) {
        std::cerr << "Failed to initialize asset database" << std::endl;
        return false;
    }

    // Create output directory
    std::filesystem::create_directories(config.assetOutputDir);

    // Setup asset converter progress callback
    AssetConverter::SetProgressCallback([this](float progress, const std::string& desc) {
        ReportProgress(progress, desc);
    });

    // Start worker threads
    for (int i = 0; i < config.maxThreads; ++i) {
        m_WorkerThreads.emplace_back([this]() { WorkerThreadMain(); });
    }

    if (config.verbose) {
        std::cout << "Asset Pipeline initialized: " << config.maxThreads << " threads" << std::endl;
    }

    return true;
}

void AssetPipeline::Shutdown() {
    m_IsRunning = false;

    // Wait for all worker threads to finish
    for (auto& thread : m_WorkerThreads) {
        if (thread.joinable()) {
            thread.join();
        }
    }

    m_WorkerThreads.clear();

    // Save database
    m_Database.Save();

    if (m_Config.verbose) {
        std::cout << "Asset Pipeline shutdown complete" << std::endl;
    }
}

bool AssetPipeline::ScanAssetDirectory(const std::string& assetDir) {
    if (!std::filesystem::exists(assetDir)) {
        std::cerr << "Asset directory not found: " << assetDir << std::endl;
        return false;
    }

    ReportProgress(0.0f, "Scanning asset directory...");

    size_t totalAssets = 0;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(assetDir)) {
        if (entry.is_regular_file()) {
            std::string filePath = entry.path().string();
            std::string assetType = AssetConverter::DetectAssetType(filePath);

            if (AssetConverter::IsAssetTypeSupported(assetType)) {
                // Get relative path
                std::string relativePath = std::filesystem::relative(filePath, assetDir).string();

                // Check if already in database
                const auto* existing = m_Database.GetAssetEntry(relativePath);
                if (!existing) {
                    // New asset
                    AssetDatabase::AssetEntry dbEntry;
                    dbEntry.path = relativePath;
                    dbEntry.type = assetType;
                    dbEntry.isDirty = true;
                    m_Database.SetAssetEntry(dbEntry);
                }

                totalAssets++;
            }
        }
    }

    m_Statistics.totalAssets = totalAssets;
    ReportProgress(1.0f, "Scanned " + std::to_string(totalAssets) + " assets");

    return true;
}

bool AssetPipeline::ProcessAssets(bool incrementalOnly) {
    m_IsProcessing = true;
    m_Statistics.processedAssets = 0;
    m_Statistics.failedAssets = 0;
    m_Statistics.skippedAssets = 0;
    m_Statistics.totalInputSize = 0;
    m_Statistics.totalOutputSize = 0;

    std::vector<std::string> assetsToProcess;

    if (incrementalOnly) {
        assetsToProcess = m_Database.GetDirtyAssets();
    } else {
        for (size_t i = 0; i < m_Database.GetAssetCount(); ++i) {
            // Get all assets - requires iterator support in database
            // For now, use dirty assets as fallback
        }
        assetsToProcess = m_Database.GetDirtyAssets();
    }

    if (m_Config.verbose) {
        std::cout << "Processing " << assetsToProcess.size() << " assets..." << std::endl;
    }

    for (const auto& assetPath : assetsToProcess) {
        if (!m_IsRunning) break;

        const auto* entry = m_Database.GetAssetEntry(assetPath);
        if (!entry) continue;

        std::string fullSourcePath = std::filesystem::path(m_Config.assetSourceDir) / assetPath;
        std::string outputPath = GetAssetOutputPath(assetPath);

        ProcessingJob job;
        job.sourcePath = fullSourcePath;
        job.outputPath = outputPath;
        job.assetType = entry->type;

        // Queue job
        QueueAssetJob(job);
    }

    // Wait for completion
    return WaitForCompletion();
}

bool AssetPipeline::ProcessAsset(const std::string& assetPath, bool force) {
    const auto* entry = m_Database.GetAssetEntry(assetPath);
    if (!entry && !force) {
        std::cerr << "Asset not found in database: " << assetPath << std::endl;
        return false;
    }

    if (!force && !entry->isDirty) {
        m_Statistics.skippedAssets++;
        return true;  // Already processed
    }

    std::string fullSourcePath = std::filesystem::path(m_Config.assetSourceDir) / assetPath;
    std::string outputPath = GetAssetOutputPath(assetPath);

    if (!std::filesystem::exists(fullSourcePath)) {
        std::cerr << "Source asset not found: " << fullSourcePath << std::endl;
        m_Statistics.failedAssets++;
        return false;
    }

    // Determine asset type
    std::string assetType = AssetConverter::DetectAssetType(fullSourcePath);

    // Perform conversion
    AssetConverter::ConversionResult result;

    if (assetType == "texture") {
        result = AssetConverter::ConvertTexture(fullSourcePath, outputPath);
    } else if (assetType == "model") {
        result = AssetConverter::ConvertMesh(fullSourcePath, outputPath);
    } else if (assetType == "material") {
        result = AssetConverter::ConvertMaterial(fullSourcePath, outputPath);
    } else {
        // Copy unsupported types as-is
        try {
            std::filesystem::copy_file(fullSourcePath, outputPath, std::filesystem::copy_options::overwrite_existing);
            result.success = true;
            result.outputPath = outputPath;
        } catch (const std::exception& e) {
            result.success = false;
            result.errorMessage = e.what();
        }
    }

    if (result.success) {
        m_Statistics.processedAssets++;
        m_Statistics.totalInputSize += result.inputSize;
        m_Statistics.totalOutputSize += result.outputSize;

        UpdateDatabase(assetPath, result);
        ReportProgress(m_Statistics.GetProgress(),
                      "Processed: " + assetPath + " (" + std::to_string(result.outputSize) + " bytes)");
    } else {
        m_Statistics.failedAssets++;
        ReportProgress(m_Statistics.GetProgress(),
                      "Failed: " + assetPath + " - " + result.errorMessage);
    }

    return result.success;
}

void AssetPipeline::QueueAssetJob(const ProcessingJob& job) {
    {
        std::lock_guard<std::mutex> lock(m_JobQueueMutex);
        m_JobQueue.push(job);
    }
}

bool AssetPipeline::WaitForCompletion(int timeoutMs) {
    auto startTime = std::chrono::high_resolution_clock::now();

    while (true) {
        {
            std::lock_guard<std::mutex> lock(m_JobQueueMutex);
            if (m_JobQueue.empty() && m_ActiveJobs == 0) {
                break;
            }
        }

        if (timeoutMs > 0) {
            auto currentTime = std::chrono::high_resolution_clock::now();
            auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                currentTime - startTime).count();
            if (elapsedMs > timeoutMs) {
                return false;
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    m_IsProcessing = false;
    return true;
}

void AssetPipeline::Clean() {
    if (std::filesystem::exists(m_Config.assetOutputDir)) {
        std::filesystem::remove_all(m_Config.assetOutputDir);
    }
    std::filesystem::create_directories(m_Config.assetOutputDir);
    m_Database.Clear();

    if (m_Config.verbose) {
        std::cout << "Asset cache cleaned" << std::endl;
    }
}

void AssetPipeline::CleanAssetType(const std::string& assetType) {
    auto assets = m_Database.GetAssetsByType(assetType);
    for (const auto& assetPath : assets) {
        std::string outputPath = GetAssetOutputPath(assetPath);
        if (std::filesystem::exists(outputPath)) {
            std::filesystem::remove(outputPath);
        }
    }
}

void AssetPipeline::ForceFullRebuild() {
    for (size_t i = 0; i < m_Database.GetAssetCount(); ++i) {
        // Mark all assets as dirty
        // Requires iterator support
    }
    m_Statistics = Statistics();
}

std::vector<std::string> AssetPipeline::GetSupportedAssetTypes() {
    return {"texture", "model", "shader", "material", "audio", "scene"};
}

nlohmann::json AssetPipeline::ConfigToJson(const Config& config) {
    nlohmann::json j;
    j["assetSourceDir"] = config.assetSourceDir;
    j["assetOutputDir"] = config.assetOutputDir;
    j["databasePath"] = config.databasePath;
    j["maxThreads"] = config.maxThreads;
    j["enableCompression"] = config.enableCompression;
    j["enableCaching"] = config.enableCaching;
    j["validateAssets"] = config.validateAssets;
    j["incrementalBuild"] = config.incrementalBuild;
    j["verbose"] = config.verbose;
    return j;
}

AssetPipeline::Config AssetPipeline::ConfigFromJson(const nlohmann::json& j) {
    Config config;
    config.assetSourceDir = j.value("assetSourceDir", "assets");
    config.assetOutputDir = j.value("assetOutputDir", "assets_processed");
    config.databasePath = j.value("databasePath", "asset_database.json");
    config.maxThreads = j.value("maxThreads", 4);
    config.enableCompression = j.value("enableCompression", true);
    config.enableCaching = j.value("enableCaching", true);
    config.validateAssets = j.value("validateAssets", true);
    config.incrementalBuild = j.value("incrementalBuild", true);
    config.verbose = j.value("verbose", false);
    return config;
}

void AssetPipeline::WorkerThreadMain() {
    while (m_IsRunning) {
        ProcessingJob job;
        bool hasJob = false;

        {
            std::lock_guard<std::mutex> lock(m_JobQueueMutex);
            if (!m_JobQueue.empty()) {
                job = m_JobQueue.front();
                m_JobQueue.pop();
                hasJob = true;
                m_ActiveJobs++;
            }
        }

        if (hasJob) {
            ProcessAsset(job.sourcePath);
            m_ActiveJobs--;
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
}

void AssetPipeline::ReportProgress(float progress, const std::string& description) {
    if (m_ProgressCallback) {
        m_ProgressCallback(progress, description);
    }
}

std::string AssetPipeline::GetAssetOutputPath(const std::string& assetPath) {
    // Replace extension with expected output format
    std::string outputPath = assetPath;
    std::string assetType = AssetConverter::DetectAssetType(assetPath);

    // Keep path structure but potentially change extension
    return std::filesystem::path(m_Config.assetOutputDir) / outputPath;
}

bool AssetPipeline::ShouldProcessAsset(const std::string& assetPath) const {
    const auto* entry = m_Database.GetAssetEntry(assetPath);
    if (!entry) return true;

    return entry->isDirty;
}

void AssetPipeline::UpdateDatabase(const std::string& assetPath, const AssetConverter::ConversionResult& result) {
    auto* entry = const_cast<AssetDatabase::AssetEntry*>(m_Database.GetAssetEntry(assetPath));

    if (entry && result.success) {
        // Compute hash of source file
        entry->sourceHash = AssetHash::ComputeHash(m_Config.assetSourceDir + "/" + assetPath);

        if (std::filesystem::exists(result.outputPath)) {
            entry->processedHash = AssetHash::ComputeHash(result.outputPath);
        }

        entry->processedPath = result.outputPath;
        entry->isDirty = false;

        // Update timestamp
        auto now = std::chrono::system_clock::now();
        auto tt = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::gmtime(&tt), "%Y-%m-%d %H:%M:%S");
        entry->lastProcessedTime = ss.str();

        m_Database.SetAssetEntry(*entry);
    }
}
