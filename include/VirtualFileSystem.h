#pragma once

#include <string>
#include <vector>
#include <memory>
#include <map>
#include <cstdint>
#include <functional>

/**
 * @class IFileSystemProvider
 * @brief Abstract interface for different file system implementations
 * 
 * Allows pluggable backends: physical filesystem, zip archives, memory, etc.
 */
class IFileSystemProvider {
public:
    virtual ~IFileSystemProvider() = default;
    
    /**
     * Read entire file into memory
     * @param path Virtual file path
     * @param outData Output buffer (caller owns memory)
     * @param outSize Output size in bytes
     * @return true if successful
     */
    virtual bool ReadFile(const std::string& path, uint8_t*& outData, size_t& outSize) = 0;
    
    /**
     * Read file asynchronously (non-blocking)
     * @param path Virtual file path
     * @param callback Called with (data, size, success) when complete
     */
    virtual void ReadFileAsync(
        const std::string& path,
        std::function<void(uint8_t*, size_t, bool)> callback) = 0;
    
    /**
     * Check if file exists
     */
    virtual bool FileExists(const std::string& path) = 0;
    
    /**
     * List directory contents
     */
    virtual std::vector<std::string> ListDirectory(const std::string& path) = 0;
    
    /**
     * Get file size in bytes
     */
    virtual size_t GetFileSize(const std::string& path) = 0;
    
    /**
     * Get last modification time (Unix timestamp)
     */
    virtual uint64_t GetModificationTime(const std::string& path) = 0;
    
    /**
     * Mount this provider at virtual path
     * @param virtualPath e.g., "/assets" or "/pak0"
     */
    virtual void SetMountPoint(const std::string& virtualPath) = 0;
    virtual std::string GetMountPoint() const = 0;
};

/**
 * @class PhysicalFileSystemProvider
 * @brief Maps physical directories to virtual paths
 */
class PhysicalFileSystemProvider : public IFileSystemProvider {
public:
    explicit PhysicalFileSystemProvider(const std::string& physicalRoot);
    
    bool ReadFile(const std::string& path, uint8_t*& outData, size_t& outSize) override;
    void ReadFileAsync(
        const std::string& path,
        std::function<void(uint8_t*, size_t, bool)> callback) override;
    
    bool FileExists(const std::string& path) override;
    std::vector<std::string> ListDirectory(const std::string& path) override;
    size_t GetFileSize(const std::string& path) override;
    uint64_t GetModificationTime(const std::string& path) override;
    
    void SetMountPoint(const std::string& virtualPath) override;
    std::string GetMountPoint() const override { return m_MountPoint; }

private:
    std::string m_PhysicalRoot;
    std::string m_MountPoint;
    
    std::string MapVirtualToPhysical(const std::string& virtualPath) const;
};

/**
 * @class MemoryFileSystemProvider
 * @brief In-memory file system for testing and embedded resources
 */
class MemoryFileSystemProvider : public IFileSystemProvider {
public:
    struct MemoryFile {
        uint8_t* data = nullptr;
        size_t size = 0;
        uint64_t modTime = 0;
    };
    
    bool ReadFile(const std::string& path, uint8_t*& outData, size_t& outSize) override;
    void ReadFileAsync(
        const std::string& path,
        std::function<void(uint8_t*, size_t, bool)> callback) override;
    
    bool FileExists(const std::string& path) override;
    std::vector<std::string> ListDirectory(const std::string& path) override;
    size_t GetFileSize(const std::string& path) override;
    uint64_t GetModificationTime(const std::string& path) override;
    
    void SetMountPoint(const std::string& virtualPath) override;
    std::string GetMountPoint() const override { return m_MountPoint; }
    
    /**
     * Add file to memory filesystem
     */
    void AddFile(const std::string& path, uint8_t* data, size_t size);
    void RemoveFile(const std::string& path);
    void Clear();

private:
    std::map<std::string, MemoryFile> m_Files;
    std::string m_MountPoint;
};

/**
 * @class VirtualFileSystem
 * @brief Unified interface for multiple file system backends
 * 
 * Manages mounting of different providers and provides transparent access
 * to files regardless of their actual storage location.
 * 
 * Example:
 * @code
 * VirtualFileSystem vfs;
 * vfs.Mount("/assets", std::make_shared<PhysicalFileSystemProvider>("./assets"));
 * vfs.Mount("/pak0", std::make_shared<ZipFileSystemProvider>("data.pak"));
 * 
 * auto data = vfs.ReadFile("/assets/textures/brick.png", size);
 * auto model = vfs.ReadFile("/pak0/models/player.gltf", size);
 * @endcode
 */
class VirtualFileSystem {
public:
    VirtualFileSystem();
    ~VirtualFileSystem();
    
    /**
     * Mount a file system provider at a virtual path
     * @param virtualPath Mount point (e.g., "/assets", "/pak0")
     * @param provider File system implementation
     */
    void Mount(const std::string& virtualPath, std::shared_ptr<IFileSystemProvider> provider);
    
    /**
     * Unmount a file system provider
     */
    void Unmount(const std::string& virtualPath);
    
    /**
     * Read entire file into memory
     */
    std::vector<uint8_t> ReadFile(const std::string& path);
    
    /**
     * Read file asynchronously
     */
    void ReadFileAsync(
        const std::string& path,
        std::function<void(const std::vector<uint8_t>&, bool)> callback);
    
    /**
     * Check if file exists in any mounted provider
     */
    bool FileExists(const std::string& path);
    
    /**
     * List directory contents (searches all mounted providers)
     */
    std::vector<std::string> ListDirectory(const std::string& path);
    
    /**
     * Get file size
     */
    size_t GetFileSize(const std::string& path);
    
    /**
     * Get modification time
     */
    uint64_t GetModificationTime(const std::string& path);
    
    /**
     * Get list of mounted file systems
     */
    std::vector<std::string> GetMountPoints() const;
    
    /**
     * Process pending async reads (call once per frame)
     */
    void Update();

private:
    struct MountedProvider {
        std::shared_ptr<IFileSystemProvider> provider;
        std::string virtualPath;
        size_t priority; // Higher priority = checked first
    };
    
    std::vector<MountedProvider> m_Providers;
    size_t m_PriorityCounter = 0;
    
    // Async request tracking
    struct AsyncRequest {
        std::string path;
        std::function<void(const std::vector<uint8_t>&, bool)> callback;
    };
    std::vector<AsyncRequest> m_PendingRequests;
    
    IFileSystemProvider* FindProvider(const std::string& path);
    std::string NormalizePath(const std::string& path) const;
};

// Global VFS instance (optional convenience)
VirtualFileSystem& GetVFS();
