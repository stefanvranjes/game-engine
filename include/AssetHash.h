#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <filesystem>

/**
 * @brief Asset hashing and integrity verification utilities
 * 
 * Provides efficient hashing for detecting asset changes and supporting
 * incremental builds. Uses SHA256 for cryptographic integrity checking
 * and xxHash for fast content hashing.
 */
class AssetHash {
public:
    /**
     * @brief Hash type for asset content
     */
    struct Hash {
        std::string sha256;      // Full SHA256 for integrity
        uint64_t xxHash64;       // Fast xxHash for quick comparisons
        std::string timestamp;   // Last modification time
        size_t fileSize;         // Asset size in bytes
        
        bool operator==(const Hash& other) const {
            return sha256 == other.sha256 && fileSize == other.fileSize;
        }
        
        bool operator!=(const Hash& other) const {
            return !(*this == other);
        }
    };

    /**
     * @brief Compute full hash for an asset file
     * @param filepath Path to asset file
     * @return Hash structure or empty on error
     */
    static Hash ComputeHash(const std::string& filepath);

    /**
     * @brief Compute hash from raw data
     * @param data Pointer to data buffer
     * @param size Size of data in bytes
     * @return Hash structure
     */
    static Hash ComputeHashFromData(const unsigned char* data, size_t size);

    /**
     * @brief Quick hash using xxHash64 (for fast comparisons)
     * @param filepath Path to asset file
     * @return 64-bit xxHash value
     */
    static uint64_t ComputeQuickHash(const std::string& filepath);

    /**
     * @brief Quick hash from raw data
     * @param data Pointer to data buffer
     * @param size Size of data in bytes
     * @return 64-bit xxHash value
     */
    static uint64_t ComputeQuickHashFromData(const unsigned char* data, size_t size);

    /**
     * @brief Check if file has been modified since last hash
     * @param filepath Path to asset file
     * @param previousHash Previously computed hash
     * @return true if file content has changed
     */
    static bool HasFileChanged(const std::string& filepath, const Hash& previousHash);

    /**
     * @brief Compute incremental hash for file portions (for streaming large files)
     * @param filepath Path to asset file
     * @param chunkSize Size of each chunk to hash
     * @return Vector of chunk hashes
     */
    static std::vector<uint64_t> ComputeChunkedHash(const std::string& filepath, size_t chunkSize = 1024 * 1024);

    /**
     * @brief Verify file integrity against stored hash
     * @param filepath Path to asset file
     * @param storedHash Previously computed hash
     * @return true if file matches stored hash
     */
    static bool VerifyIntegrity(const std::string& filepath, const Hash& storedHash);

    /**
     * @brief Get human-readable hash string
     * @param hash Hash to stringify
     * @return Formatted hash string (SHA256:xxHash64)
     */
    static std::string ToString(const Hash& hash);

    /**
     * @brief Parse hash string back to Hash structure
     * @param str Hash string from ToString
     * @return Parsed Hash structure
     */
    static Hash FromString(const std::string& str);
};
