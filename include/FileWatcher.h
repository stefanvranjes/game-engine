#pragma once

#include <string>
#include <unordered_map>
#include <functional>
#include <chrono>
#include <vector>
#include <mutex>

/**
 * @class FileWatcher
 * @brief Monitors files and directories for changes and triggers callbacks
 * 
 * Supports watching individual files or entire directories for modifications.
 * Uses file modification timestamps to detect changes.
 */
class FileWatcher {
public:
    using ChangeCallback = std::function<void(const std::string& path)>;
    using ChangeHandle = size_t;

    FileWatcher();
    ~FileWatcher();

    /**
     * Watch a single file for changes
     * @param path File path to watch
     * @param callback Function to call when file changes
     * @return Handle to unwatch later
     */
    ChangeHandle WatchFile(const std::string& path, ChangeCallback callback);

    /**
     * Watch a directory recursively for changes (shader, texture files)
     * @param directory Directory path to watch
     * @param extension File extension to filter (e.g., ".glsl", ".png"), empty = all files
     * @param callback Function to call when a file in directory changes
     * @return Handle to unwatch later
     */
    ChangeHandle WatchDirectory(const std::string& directory, const std::string& extension, ChangeCallback callback);

    /**
     * Stop watching a file or directory
     * @param handle Watch handle returned from WatchFile or WatchDirectory
     */
    void Unwatch(ChangeHandle handle);

    /**
     * Poll for file changes and trigger callbacks (call once per frame)
     * @param pollIntervalMs Minimum time between filesystem checks (default 100ms)
     */
    void Update(int pollIntervalMs = 100);

    /**
     * Clear all watches
     */
    void Clear();

    /**
     * Get all watched files
     */
    std::vector<std::string> GetWatchedFiles() const;

private:
    struct FileWatch {
        std::string path;
        long long lastModified;
        ChangeCallback callback;
        bool isDirectory;
        std::string extensionFilter;
    };

    std::unordered_map<ChangeHandle, FileWatch> m_Watches;
    std::chrono::steady_clock::time_point m_LastPollTime;
    mutable std::mutex m_WatchesMutex;
    ChangeHandle m_NextHandle = 1;

    long long GetFileModifiedTime(const std::string& path) const;
    bool FileExists(const std::string& path) const;
    std::vector<std::string> GetFilesInDirectory(const std::string& directory, const std::string& extension) const;
    bool HasExtension(const std::string& filename, const std::string& extension) const;
};
