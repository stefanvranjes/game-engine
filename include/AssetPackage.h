#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <memory>
#include <map>
#include <functional>

/**
 * @class AssetPackage
 * @brief Binary asset container format for efficient distribution and loading
 * 
 * Format Overview:
 * - Header: Magic ("APKG"), Version, AssetCount
 * - Directory: List of asset headers (path, offset, size, compression)
 * - Data: Compressed asset data blocks
 * 
 * Benefits:
 * - Single file distribution (easier version management)
 * - Optional compression (LZ4 for fast decompression)
 * - Streaming-friendly (indexed access without loading entire package)
 * - Fast enumeration (directory precedes data)
 * - Optional encryption support
 */
class AssetPackage {
public:
    static constexpr uint32_t MAGIC = 0x504B4741; // "APKG" in little-endian
    static constexpr uint32_t VERSION = 1;
    
    enum class CompressionType : uint32_t {
        None = 0,
        LZ4 = 1,
        Deflate = 2
    };
    
    struct AssetHeader {
        std::string path;           // Virtual path (e.g., "/models/player.gltf")
        uint64_t dataOffset = 0;    // Offset in file
        uint64_t compressedSize = 0;
        uint64_t uncompressedSize = 0;
        CompressionType compression = CompressionType::None;
        uint32_t crc32 = 0;         // For integrity checking
    };
    
    /**
     * Load package file
     */
    bool Load(const std::string& path);
    
    /**
     * Load package from memory
     */
    bool LoadFromMemory(const uint8_t* data, size_t size);
    
    /**
     * Create new package (for building)
     */
    static std::unique_ptr<AssetPackage> Create();
    
    /**
     * Add asset to package
     */
    void AddAsset(
        const std::string& virtualPath,
        const uint8_t* data,
        size_t size,
        CompressionType compression = CompressionType::LZ4);
    
    /**
     * Extract asset from loaded package
     */
    std::vector<uint8_t> ExtractAsset(const std::string& path);
    
    /**
     * Check if asset exists in package
     */
    bool HasAsset(const std::string& path) const;
    
    /**
     * List all assets in package
     */
    std::vector<std::string> ListAssets() const;
    
    /**
     * Get asset header
     */
    const AssetHeader* GetAssetHeader(const std::string& path) const;
    
    /**
     * Save package to file
     */
    bool Save(const std::string& path);
    
    /**
     * Get total package size
     */
    size_t GetPackageSize() const { return m_PackageData.size(); }
    
    /**
     * Get number of assets
     */
    size_t GetAssetCount() const { return m_Directory.size(); }

private:
    uint8_t* m_PackageData = nullptr;
    size_t m_PackageSize = 0;
    
    std::map<std::string, AssetHeader> m_Directory;
    bool m_IsLoaded = false;
    
    // For building
    std::vector<uint8_t> m_BuildBuffer;
    std::vector<AssetHeader> m_BuildDirectory;
};

/**
 * @class AssetPackageProvider
 * @brief VirtualFileSystem provider for AssetPackage files
 * 
 * Allows transparent access to packaged assets through VFS:
 * @code
 * VirtualFileSystem vfs;
 * auto pkg = std::make_shared<AssetPackage>();
 * pkg->Load("data.pak");
 * vfs.Mount("/pak0", std::make_shared<AssetPackageProvider>(pkg));
 * 
 * // Now files in package are accessible:
 * auto data = vfs.ReadFile("/pak0/models/player.gltf", size);
 * @endcode
 */
class AssetPackageProvider : public IFileSystemProvider {
public:
    explicit AssetPackageProvider(std::shared_ptr<AssetPackage> package);
    
    bool ReadFile(const std::string& path, uint8_t*& outData, size_t& outSize) override;
    void ReadFileAsync(
        const std::string& path,
        std::function<void(uint8_t*, size_t, bool)> callback) override;
    
    bool FileExists(const std::string& path) override;
    std::vector<std::string> ListDirectory(const std::string& path) override;
    size_t GetFileSize(const std::string& path) override;
    uint64_t GetModificationTime(const std::string& path) override { return 0; }
    
    void SetMountPoint(const std::string& virtualPath) override;
    std::string GetMountPoint() const override { return m_MountPoint; }

private:
    std::shared_ptr<AssetPackage> m_Package;
    std::string m_MountPoint;
};

/**
 * @class AssetPackageBuilder
 * @brief Utility for building asset packages from directories
 * 
 * Usage:
 * @code
 * AssetPackageBuilder builder;
 * builder.AddDirectory("./assets", "/*");
 * builder.SetCompression(AssetPackage::CompressionType::LZ4);
 * builder.Build("./assets/game.pak");
 * @endcode
 */
class AssetPackageBuilder {
public:
    /**
     * Add directory contents to package
     * @param sourceDir Physical directory to include
     * @param virtualBase Virtual path prefix (e.g., "/assets")
     * @param pattern File pattern ("*" for all, "*.png" for specific)
     */
    void AddDirectory(
        const std::string& sourceDir,
        const std::string& virtualBase = "/",
        const std::string& pattern = "*");
    
    /**
     * Add single file
     */
    void AddFile(const std::string& sourcePath, const std::string& virtualPath);
    
    /**
     * Set compression for new assets
     */
    void SetCompression(AssetPackage::CompressionType compression) {
        m_Compression = compression;
    }
    
    /**
     * Build package file
     */
    bool Build(const std::string& outputPath);
    
    /**
     * Get statistics
     */
    struct BuildStats {
        size_t fileCount = 0;
        size_t uncompressedSize = 0;
        size_t compressedSize = 0;
        float compressionRatio = 0.0f;
    };
    BuildStats GetStats() const { return m_Stats; }

private:
    std::unique_ptr<AssetPackage> m_Package;
    AssetPackage::CompressionType m_Compression = AssetPackage::CompressionType::LZ4;
    BuildStats m_Stats;
};
