#include "VirtualFileSystem.h"
#include <iostream>
#include <algorithm>
#include <thread>
#include <filesystem>
#include <cstring>
#include <fstream>
#include <ctime>

// ============================================================================
// PhysicalFileSystemProvider Implementation
// ============================================================================

PhysicalFileSystemProvider::PhysicalFileSystemProvider(const std::string& physicalRoot)
    : m_PhysicalRoot(physicalRoot) {
    // Normalize path
    if (!physicalRoot.empty() && physicalRoot.back() != '/') {
        m_PhysicalRoot += '/';
    }
}

std::string PhysicalFileSystemProvider::MapVirtualToPhysical(const std::string& virtualPath) const {
    std::string relativePath = virtualPath;
    
    // Remove mount point prefix
    if (!m_MountPoint.empty() && virtualPath.find(m_MountPoint) == 0) {
        relativePath = virtualPath.substr(m_MountPoint.length());
    }
    
    // Remove leading slash
    if (!relativePath.empty() && relativePath[0] == '/') {
        relativePath = relativePath.substr(1);
    }
    
    return m_PhysicalRoot + relativePath;
}

bool PhysicalFileSystemProvider::ReadFile(const std::string& path, uint8_t*& outData, size_t& outSize) {
    std::string physicalPath = MapVirtualToPhysical(path);
    
    try {
        std::ifstream file(physicalPath, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            return false;
        }
        
        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);
        
        outData = new uint8_t[size];
        if (!file.read(reinterpret_cast<char*>(outData), size)) {
            delete[] outData;
            return false;
        }
        
        outSize = size;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error reading file: " << physicalPath << " - " << e.what() << std::endl;
        return false;
    }
}

void PhysicalFileSystemProvider::ReadFileAsync(
    const std::string& path,
    std::function<void(uint8_t*, size_t, bool)> callback) {
    
    std::thread([this, path, callback]() {
        uint8_t* data = nullptr;
        size_t size = 0;
        bool success = ReadFile(path, data, size);
        callback(data, size, success);
    }).detach();
}

bool PhysicalFileSystemProvider::FileExists(const std::string& path) {
    std::string physicalPath = MapVirtualToPhysical(path);
    return std::filesystem::exists(physicalPath);
}

std::vector<std::string> PhysicalFileSystemProvider::ListDirectory(const std::string& path) {
    std::vector<std::string> result;
    std::string physicalPath = MapVirtualToPhysical(path);
    
    try {
        if (std::filesystem::is_directory(physicalPath)) {
            for (const auto& entry : std::filesystem::directory_iterator(physicalPath)) {
                result.push_back(entry.path().filename().string());
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error listing directory: " << physicalPath << " - " << e.what() << std::endl;
    }
    
    return result;
}

size_t PhysicalFileSystemProvider::GetFileSize(const std::string& path) {
    std::string physicalPath = MapVirtualToPhysical(path);
    try {
        return std::filesystem::file_size(physicalPath);
    } catch (...) {
        return 0;
    }
}

uint64_t PhysicalFileSystemProvider::GetModificationTime(const std::string& path) {
    std::string physicalPath = MapVirtualToPhysical(path);
    try {
        auto lastWrite = std::filesystem::last_write_time(physicalPath);
        auto duration = lastWrite.time_since_epoch();
        return std::chrono::duration_cast<std::chrono::seconds>(duration).count();
    } catch (...) {
        return 0;
    }
}

void PhysicalFileSystemProvider::SetMountPoint(const std::string& virtualPath) {
    m_MountPoint = virtualPath;
}

// ============================================================================
// MemoryFileSystemProvider Implementation
// ============================================================================

bool MemoryFileSystemProvider::ReadFile(const std::string& path, uint8_t*& outData, size_t& outSize) {
    auto it = m_Files.find(path);
    if (it == m_Files.end()) {
        return false;
    }
    
    outSize = it->second.size;
    outData = new uint8_t[outSize];
    std::memcpy(outData, it->second.data, outSize);
    return true;
}

void MemoryFileSystemProvider::ReadFileAsync(
    const std::string& path,
    std::function<void(uint8_t*, size_t, bool)> callback) {
    
    auto it = m_Files.find(path);
    if (it == m_Files.end()) {
        callback(nullptr, 0, false);
        return;
    }
    
    uint8_t* data = new uint8_t[it->second.size];
    std::memcpy(data, it->second.data, it->second.size);
    callback(data, it->second.size, true);
}

bool MemoryFileSystemProvider::FileExists(const std::string& path) {
    return m_Files.find(path) != m_Files.end();
}

std::vector<std::string> MemoryFileSystemProvider::ListDirectory(const std::string& path) {
    std::vector<std::string> result;
    std::string prefix = path;
    if (!prefix.empty() && prefix.back() != '/') {
        prefix += '/';
    }
    
    for (const auto& pair : m_Files) {
        if (pair.first.find(prefix) == 0) {
            result.push_back(pair.first.substr(prefix.length()));
        }
    }
    
    return result;
}

size_t MemoryFileSystemProvider::GetFileSize(const std::string& path) {
    auto it = m_Files.find(path);
    return it != m_Files.end() ? it->second.size : 0;
}

void MemoryFileSystemProvider::SetMountPoint(const std::string& virtualPath) {
    m_MountPoint = virtualPath;
}

void MemoryFileSystemProvider::AddFile(const std::string& path, uint8_t* data, size_t size) {
    MemoryFile file;
    file.data = new uint8_t[size];
    file.size = size;
    file.modTime = std::time(nullptr);
    std::memcpy(file.data, data, size);
    
    m_Files[path] = file;
}

void MemoryFileSystemProvider::RemoveFile(const std::string& path) {
    auto it = m_Files.find(path);
    if (it != m_Files.end()) {
        delete[] it->second.data;
        m_Files.erase(it);
    }
}

void MemoryFileSystemProvider::Clear() {
    for (auto& pair : m_Files) {
        delete[] pair.second.data;
    }
    m_Files.clear();
}

// ============================================================================
// VirtualFileSystem Implementation
// ============================================================================

static VirtualFileSystem* g_GlobalVFS = nullptr;

VirtualFileSystem::VirtualFileSystem() = default;

VirtualFileSystem::~VirtualFileSystem() {
    for (auto& provider : m_Providers) {
        provider.provider->SetMountPoint("");
    }
    m_Providers.clear();
}

void VirtualFileSystem::Mount(const std::string& virtualPath, std::shared_ptr<IFileSystemProvider> provider) {
    if (!provider) return;
    
    provider->SetMountPoint(virtualPath);
    
    MountedProvider mounted{provider, virtualPath, m_PriorityCounter++};
    m_Providers.push_back(mounted);
    
    // Sort by priority (higher = first)
    std::sort(m_Providers.begin(), m_Providers.end(),
        [](const MountedProvider& a, const MountedProvider& b) {
            return a.priority > b.priority;
        });
}

void VirtualFileSystem::Unmount(const std::string& virtualPath) {
    m_Providers.erase(
        std::remove_if(m_Providers.begin(), m_Providers.end(),
            [&virtualPath](const MountedProvider& p) {
                return p.virtualPath == virtualPath;
            }),
        m_Providers.end());
}

std::string VirtualFileSystem::NormalizePath(const std::string& path) const {
    std::string normalized = path;
    
    // Convert backslashes to forward slashes
    for (auto& c : normalized) {
        if (c == '\\') c = '/';
    }
    
    // Ensure leading slash
    if (normalized.empty() || normalized[0] != '/') {
        normalized = '/' + normalized;
    }
    
    return normalized;
}

IFileSystemProvider* VirtualFileSystem::FindProvider(const std::string& path) {
    std::string normalized = NormalizePath(path);
    
    // Try to find longest matching mount point
    IFileSystemProvider* bestMatch = nullptr;
    size_t bestMatchLength = 0;
    
    for (auto& mounted : m_Providers) {
        if (mounted.virtualPath.length() > bestMatchLength &&
            normalized.find(mounted.virtualPath) == 0) {
            bestMatch = mounted.provider.get();
            bestMatchLength = mounted.virtualPath.length();
        }
    }
    
    return bestMatch;
}

std::vector<uint8_t> VirtualFileSystem::ReadFile(const std::string& path) {
    auto provider = FindProvider(path);
    if (!provider) {
        std::cerr << "VFS: No provider found for path: " << path << std::endl;
        return std::vector<uint8_t>();
    }
    
    uint8_t* data = nullptr;
    size_t size = 0;
    
    if (provider->ReadFile(path, data, size)) {
        std::vector<uint8_t> result(data, data + size);
        delete[] data;
        return result;
    }
    
    std::cerr << "VFS: Failed to read file: " << path << std::endl;
    return std::vector<uint8_t>();
}

void VirtualFileSystem::ReadFileAsync(
    const std::string& path,
    std::function<void(const std::vector<uint8_t>&, bool)> callback) {
    
    auto provider = FindProvider(path);
    if (!provider) {
        std::cerr << "VFS: No provider found for path: " << path << std::endl;
        callback(std::vector<uint8_t>(), false);
        return;
    }
    
    provider->ReadFileAsync(path, [callback](uint8_t* data, size_t size, bool success) {
        if (success) {
            std::vector<uint8_t> result(data, data + size);
            callback(result, true);
        } else {
            callback(std::vector<uint8_t>(), false);
        }
        delete[] data;
    });
}

bool VirtualFileSystem::FileExists(const std::string& path) {
    auto provider = FindProvider(path);
    return provider && provider->FileExists(path);
}

std::vector<std::string> VirtualFileSystem::ListDirectory(const std::string& path) {
    auto provider = FindProvider(path);
    if (!provider) return {};
    
    return provider->ListDirectory(path);
}

size_t VirtualFileSystem::GetFileSize(const std::string& path) {
    auto provider = FindProvider(path);
    return provider ? provider->GetFileSize(path) : 0;
}

uint64_t VirtualFileSystem::GetModificationTime(const std::string& path) {
    auto provider = FindProvider(path);
    return provider ? provider->GetModificationTime(path) : 0;
}

std::vector<std::string> VirtualFileSystem::GetMountPoints() const {
    std::vector<std::string> result;
    for (const auto& mounted : m_Providers) {
        result.push_back(mounted.virtualPath);
    }
    return result;
}

void VirtualFileSystem::Update() {
    // Process async callbacks if any
    // This is a placeholder for future async handling
}

// Global VFS instance
VirtualFileSystem& GetVFS() {
    if (!g_GlobalVFS) {
        g_GlobalVFS = new VirtualFileSystem();
    }
    return *g_GlobalVFS;
}
