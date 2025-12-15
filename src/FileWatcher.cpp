#include "FileWatcher.h"
#include <filesystem>
#include <iostream>
#include <sys/stat.h>

#ifdef _WIN32
    #include <windows.h>
#else
    #include <unistd.h>
#endif

namespace fs = std::filesystem;

FileWatcher::FileWatcher() {
    m_LastPollTime = std::chrono::steady_clock::now();
}

FileWatcher::~FileWatcher() {
    Clear();
}

FileWatcher::ChangeHandle FileWatcher::WatchFile(const std::string& path, ChangeCallback callback) {
    if (!FileExists(path)) {
        std::cerr << "FileWatcher: File not found - " << path << std::endl;
        return 0;
    }

    std::lock_guard<std::mutex> lock(m_WatchesMutex);
    ChangeHandle handle = m_NextHandle++;
    
    FileWatch watch;
    watch.path = path;
    watch.lastModified = GetFileModifiedTime(path);
    watch.callback = callback;
    watch.isDirectory = false;
    watch.extensionFilter = "";
    
    m_Watches[handle] = watch;
    return handle;
}

FileWatcher::ChangeHandle FileWatcher::WatchDirectory(const std::string& directory, const std::string& extension, ChangeCallback callback) {
    if (!fs::exists(directory) || !fs::is_directory(directory)) {
        std::cerr << "FileWatcher: Directory not found - " << directory << std::endl;
        return 0;
    }

    std::lock_guard<std::mutex> lock(m_WatchesMutex);
    ChangeHandle handle = m_NextHandle++;
    
    FileWatch watch;
    watch.path = directory;
    watch.lastModified = 0;
    watch.callback = callback;
    watch.isDirectory = true;
    watch.extensionFilter = extension;
    
    m_Watches[handle] = watch;
    return handle;
}

void FileWatcher::Unwatch(ChangeHandle handle) {
    std::lock_guard<std::mutex> lock(m_WatchesMutex);
    m_Watches.erase(handle);
}

void FileWatcher::Update(int pollIntervalMs) {
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - m_LastPollTime);
    
    if (elapsed.count() < pollIntervalMs) {
        return;
    }
    
    m_LastPollTime = now;

    std::lock_guard<std::mutex> lock(m_WatchesMutex);
    
    for (auto& [handle, watch] : m_Watches) {
        if (!watch.isDirectory) {
            // Monitor single file
            if (FileExists(watch.path)) {
                long long currentTime = GetFileModifiedTime(watch.path);
                if (currentTime > watch.lastModified) {
                    watch.lastModified = currentTime;
                    if (watch.callback) {
                        watch.callback(watch.path);
                    }
                }
            }
        } else {
            // Monitor directory
            auto files = GetFilesInDirectory(watch.path, watch.extensionFilter);
            for (const auto& filePath : files) {
                long long currentTime = GetFileModifiedTime(filePath);
                
                // Check if this file was modified since last poll
                // We store the newest modification time per directory
                if (currentTime > watch.lastModified) {
                    watch.lastModified = currentTime;
                    if (watch.callback) {
                        watch.callback(filePath);
                    }
                }
            }
        }
    }
}

void FileWatcher::Clear() {
    std::lock_guard<std::mutex> lock(m_WatchesMutex);
    m_Watches.clear();
}

std::vector<std::string> FileWatcher::GetWatchedFiles() const {
    std::vector<std::string> files;
    std::lock_guard<std::mutex> lock(m_WatchesMutex);
    
    for (const auto& [handle, watch] : m_Watches) {
        if (!watch.isDirectory) {
            files.push_back(watch.path);
        }
    }
    
    return files;
}

long long FileWatcher::GetFileModifiedTime(const std::string& path) const {
    try {
        auto lastWriteTime = fs::last_write_time(path);
        auto duration = lastWriteTime.time_since_epoch();
        return std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
    } catch (const std::exception& e) {
        std::cerr << "FileWatcher: Error getting file time for " << path << ": " << e.what() << std::endl;
        return 0;
    }
}

bool FileWatcher::FileExists(const std::string& path) const {
    return fs::exists(path);
}

std::vector<std::string> FileWatcher::GetFilesInDirectory(const std::string& directory, const std::string& extension) const {
    std::vector<std::string> files;
    
    try {
        for (const auto& entry : fs::recursive_directory_iterator(directory)) {
            if (entry.is_regular_file()) {
                if (extension.empty() || HasExtension(entry.path().string(), extension)) {
                    files.push_back(entry.path().string());
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "FileWatcher: Error reading directory " << directory << ": " << e.what() << std::endl;
    }
    
    return files;
}

bool FileWatcher::HasExtension(const std::string& filename, const std::string& extension) const {
    if (extension.empty()) return true;
    
    // Ensure extension starts with a dot
    std::string ext = extension;
    if (ext[0] != '.') {
        ext = "." + ext;
    }
    
    size_t pos = filename.rfind(ext);
    return (pos != std::string::npos) && (pos + ext.length() == filename.length());
}
