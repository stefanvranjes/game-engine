#include "AssetPackage.h"
#include "VirtualFileSystem.h"
#include <iostream>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <algorithm>

// ============================================================================
// AssetPackage Implementation
// ============================================================================

bool AssetPackage::Load(const std::string& path) {
    try {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            std::cerr << "Failed to open package file: " << path << std::endl;
            return false;
        }
        
        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);
        
        m_PackageData = new uint8_t[size];
        if (!file.read(reinterpret_cast<char*>(m_PackageData), size)) {
            delete[] m_PackageData;
            m_PackageData = nullptr;
            return false;
        }
        
        m_PackageSize = size;
        file.close();
        
        return LoadFromMemory(m_PackageData, m_PackageSize);
    } catch (const std::exception& e) {
        std::cerr << "Error loading package: " << e.what() << std::endl;
        return false;
    }
}

bool AssetPackage::LoadFromMemory(const uint8_t* data, size_t size) {
    if (size < 16) { // Minimum header size
        std::cerr << "Package file too small" << std::endl;
        return false;
    }
    
    size_t offset = 0;
    
    // Read header
    uint32_t magic = *reinterpret_cast<const uint32_t*>(data + offset);
    offset += sizeof(uint32_t);
    
    if (magic != MAGIC) {
        std::cerr << "Invalid package magic number" << std::endl;
        return false;
    }
    
    uint32_t version = *reinterpret_cast<const uint32_t*>(data + offset);
    offset += sizeof(uint32_t);
    
    if (version != VERSION) {
        std::cerr << "Unsupported package version: " << version << std::endl;
        return false;
    }
    
    uint32_t assetCount = *reinterpret_cast<const uint32_t*>(data + offset);
    offset += sizeof(uint32_t);
    
    // Skip reserved bytes
    offset += sizeof(uint32_t);
    
    // Read directory
    m_Directory.clear();
    for (uint32_t i = 0; i < assetCount; ++i) {
        if (offset + sizeof(uint64_t) * 3 + sizeof(uint32_t) * 2 > size) {
            std::cerr << "Corrupted package directory" << std::endl;
            return false;
        }
        
        AssetHeader header;
        
        // Read path length
        uint32_t pathLen = *reinterpret_cast<const uint32_t*>(data + offset);
        offset += sizeof(uint32_t);
        
        if (offset + pathLen > size) {
            std::cerr << "Corrupted package path" << std::endl;
            return false;
        }
        
        header.path.assign(reinterpret_cast<const char*>(data + offset), pathLen);
        offset += pathLen;
        
        // Read sizes and offsets
        header.dataOffset = *reinterpret_cast<const uint64_t*>(data + offset);
        offset += sizeof(uint64_t);
        
        header.compressedSize = *reinterpret_cast<const uint64_t*>(data + offset);
        offset += sizeof(uint64_t);
        
        header.uncompressedSize = *reinterpret_cast<const uint64_t*>(data + offset);
        offset += sizeof(uint64_t);
        
        header.compression = *reinterpret_cast<const CompressionType*>(data + offset);
        offset += sizeof(uint32_t);
        
        header.crc32 = *reinterpret_cast<const uint32_t*>(data + offset);
        offset += sizeof(uint32_t);
        
        m_Directory[header.path] = header;
    }
    
    m_IsLoaded = true;
    std::cout << "Loaded package with " << assetCount << " assets" << std::endl;
    return true;
}

std::unique_ptr<AssetPackage> AssetPackage::Create() {
    return std::make_unique<AssetPackage>();
}

void AssetPackage::AddAsset(
    const std::string& virtualPath,
    const uint8_t* data,
    size_t size,
    CompressionType compression) {
    
    if (!m_IsLoaded && m_BuildDirectory.empty()) {
        // New package being built
    }
    
    AssetHeader header;
    header.path = virtualPath;
    header.uncompressedSize = size;
    header.compression = compression;
    
    // For now, store uncompressed in build buffer
    header.dataOffset = m_BuildBuffer.size();
    header.compressedSize = size;
    
    m_BuildBuffer.insert(m_BuildBuffer.end(), data, data + size);
    m_BuildDirectory.push_back(header);
}

std::vector<uint8_t> AssetPackage::ExtractAsset(const std::string& path) {
    if (!m_IsLoaded || !m_PackageData) {
        std::cerr << "Package not loaded" << std::endl;
        return {};
    }
    
    auto it = m_Directory.find(path);
    if (it == m_Directory.end()) {
        std::cerr << "Asset not found in package: " << path << std::endl;
        return {};
    }
    
    const AssetHeader& header = it->second;
    
    if (header.dataOffset + header.compressedSize > m_PackageSize) {
        std::cerr << "Asset data out of bounds" << std::endl;
        return {};
    }
    
    const uint8_t* assetData = m_PackageData + header.dataOffset;
    
    if (header.compression == CompressionType::None) {
        return std::vector<uint8_t>(assetData, assetData + header.compressedSize);
    } else {
        // Decompression would happen here (LZ4, deflate, etc.)
        // For now, just return as-is
        std::cerr << "Compressed assets not yet supported" << std::endl;
        return std::vector<uint8_t>(assetData, assetData + header.compressedSize);
    }
}

bool AssetPackage::HasAsset(const std::string& path) const {
    return m_Directory.find(path) != m_Directory.end();
}

std::vector<std::string> AssetPackage::ListAssets() const {
    std::vector<std::string> result;
    for (const auto& pair : m_Directory) {
        result.push_back(pair.first);
    }
    return result;
}

const AssetPackage::AssetHeader* AssetPackage::GetAssetHeader(const std::string& path) const {
    auto it = m_Directory.find(path);
    return it != m_Directory.end() ? &it->second : nullptr;
}

bool AssetPackage::Save(const std::string& path) {
    try {
        std::ofstream file(path, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Failed to create package file: " << path << std::endl;
            return false;
        }
        
        // Write header
        file.write(reinterpret_cast<const char*>(&MAGIC), sizeof(uint32_t));
        
        uint32_t version = VERSION;
        file.write(reinterpret_cast<const char*>(&version), sizeof(uint32_t));
        
        uint32_t assetCount = static_cast<uint32_t>(m_BuildDirectory.size());
        file.write(reinterpret_cast<const char*>(&assetCount), sizeof(uint32_t));
        
        uint32_t reserved = 0;
        file.write(reinterpret_cast<const char*>(&reserved), sizeof(uint32_t));
        
        // Write directory
        for (const auto& header : m_BuildDirectory) {
            uint32_t pathLen = static_cast<uint32_t>(header.path.length());
            file.write(reinterpret_cast<const char*>(&pathLen), sizeof(uint32_t));
            file.write(header.path.c_str(), pathLen);
            
            file.write(reinterpret_cast<const char*>(&header.dataOffset), sizeof(uint64_t));
            file.write(reinterpret_cast<const char*>(&header.compressedSize), sizeof(uint64_t));
            file.write(reinterpret_cast<const char*>(&header.uncompressedSize), sizeof(uint64_t));
            
            uint32_t compression = static_cast<uint32_t>(header.compression);
            file.write(reinterpret_cast<const char*>(&compression), sizeof(uint32_t));
            
            file.write(reinterpret_cast<const char*>(&header.crc32), sizeof(uint32_t));
        }
        
        // Write asset data
        file.write(reinterpret_cast<const char*>(m_BuildBuffer.data()), m_BuildBuffer.size());
        
        file.close();
        std::cout << "Saved package with " << assetCount << " assets to: " << path << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error saving package: " << e.what() << std::endl;
        return false;
    }
}

// ============================================================================
// AssetPackageProvider Implementation
// ============================================================================

AssetPackageProvider::AssetPackageProvider(std::shared_ptr<AssetPackage> package)
    : m_Package(package) {
}

bool AssetPackageProvider::ReadFile(const std::string& path, uint8_t*& outData, size_t& outSize) {
    if (!m_Package) return false;
    
    auto data = m_Package->ExtractAsset(path);
    if (data.empty()) return false;
    
    outSize = data.size();
    outData = new uint8_t[outSize];
    std::memcpy(outData, data.data(), outSize);
    return true;
}

void AssetPackageProvider::ReadFileAsync(
    const std::string& path,
    std::function<void(uint8_t*, size_t, bool)> callback) {
    
    uint8_t* data = nullptr;
    size_t size = 0;
    bool success = ReadFile(path, data, size);
    callback(data, size, success);
}

bool AssetPackageProvider::FileExists(const std::string& path) {
    return m_Package && m_Package->HasAsset(path);
}

std::vector<std::string> AssetPackageProvider::ListDirectory(const std::string& path) {
    if (!m_Package) return {};
    
    auto assets = m_Package->ListAssets();
    std::vector<std::string> result;
    
    std::string prefix = path;
    if (!prefix.empty() && prefix.back() != '/') {
        prefix += '/';
    }
    
    for (const auto& asset : assets) {
        if (asset.find(prefix) == 0) {
            result.push_back(asset.substr(prefix.length()));
        }
    }
    
    return result;
}

size_t AssetPackageProvider::GetFileSize(const std::string& path) {
    if (!m_Package) return 0;
    
    auto header = m_Package->GetAssetHeader(path);
    return header ? header->uncompressedSize : 0;
}

void AssetPackageProvider::SetMountPoint(const std::string& virtualPath) {
    m_MountPoint = virtualPath;
}

// ============================================================================
// AssetPackageBuilder Implementation
// ============================================================================

void AssetPackageBuilder::AddDirectory(
    const std::string& sourceDir,
    const std::string& virtualBase,
    const std::string& pattern) {
    
    if (!m_Package) {
        m_Package = AssetPackage::Create();
    }
    
    try {
        for (const auto& entry : std::filesystem::recursive_directory_iterator(sourceDir)) {
            if (entry.is_regular_file()) {
                // Check pattern match (simplified - just check extension for *.ext patterns)
                bool matches = (pattern == "*");
                if (!matches && pattern.find('*') != std::string::npos) {
                    std::string ext = entry.path().extension().string();
                    std::string patExt = pattern.substr(pattern.find('*'));
                    matches = ext == patExt;
                }
                
                if (matches) {
                    std::string virtualPath = virtualBase + 
                        entry.path().string().substr(sourceDir.length());
                    
                    // Normalize path separators
                    for (auto& c : virtualPath) {
                        if (c == '\\') c = '/';
                    }
                    
                    AddFile(entry.path().string(), virtualPath);
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error adding directory: " << e.what() << std::endl;
    }
}

void AssetPackageBuilder::AddFile(const std::string& sourcePath, const std::string& virtualPath) {
    try {
        std::ifstream file(sourcePath, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            std::cerr << "Failed to open file: " << sourcePath << std::endl;
            return;
        }
        
        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);
        
        std::vector<uint8_t> data(size);
        if (!file.read(reinterpret_cast<char*>(data.data()), size)) {
            std::cerr << "Failed to read file: " << sourcePath << std::endl;
            return;
        }
        file.close();
        
        m_Package->AddAsset(virtualPath, data.data(), data.size(), m_Compression);
        
        m_Stats.fileCount++;
        m_Stats.uncompressedSize += size;
        m_Stats.compressedSize += size; // Without actual compression
        
    } catch (const std::exception& e) {
        std::cerr << "Error adding file: " << e.what() << std::endl;
    }
}

bool AssetPackageBuilder::Build(const std::string& outputPath) {
    if (!m_Package) {
        std::cerr << "No files added to package" << std::endl;
        return false;
    }
    
    if (m_Stats.uncompressedSize > 0) {
        m_Stats.compressionRatio = 
            static_cast<float>(m_Stats.compressedSize) / m_Stats.uncompressedSize;
    }
    
    return m_Package->Save(outputPath);
}
