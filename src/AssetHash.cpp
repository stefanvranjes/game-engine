#include "AssetHash.h"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cstring>
#include <chrono>

// Simple SHA256 implementation (portable)
namespace {
    // SHA256 constants and functions
    static const uint32_t k[] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
    };

    #define RIGHTROTATE(word, bits) (((word) >> (bits)) | ((word) << (32 - (bits))))

    std::string ComputeSHA256(const unsigned char* data, size_t size) {
        uint32_t h0 = 0x6a09e667;
        uint32_t h1 = 0xbb67ae85;
        uint32_t h2 = 0x3c6ef372;
        uint32_t h3 = 0xa54ff53a;
        uint32_t h4 = 0x510e527f;
        uint32_t h5 = 0x9b05688c;
        uint32_t h6 = 0x1f83d9ab;
        uint32_t h7 = 0x5be0cd19;

        // Pre-processing
        std::vector<unsigned char> msg(data, data + size);
        uint64_t originalLength = size * 8;
        msg.push_back(0x80);
        while ((msg.size() % 64) != 56) {
            msg.push_back(0x00);
        }

        for (int i = 7; i >= 0; i--) {
            msg.push_back((originalLength >> (i * 8)) & 0xff);
        }

        // Process message in 512-bit chunks
        for (size_t offset = 0; offset < msg.size(); offset += 64) {
            uint32_t w[64];

            for (int i = 0; i < 16; i++) {
                w[i] = ((uint32_t)msg[offset + i * 4] << 24) |
                       ((uint32_t)msg[offset + i * 4 + 1] << 16) |
                       ((uint32_t)msg[offset + i * 4 + 2] << 8) |
                       ((uint32_t)msg[offset + i * 4 + 3]);
            }

            for (int i = 16; i < 64; i++) {
                uint32_t s0 = RIGHTROTATE(w[i - 15], 7) ^ RIGHTROTATE(w[i - 15], 18) ^ (w[i - 15] >> 3);
                uint32_t s1 = RIGHTROTATE(w[i - 2], 17) ^ RIGHTROTATE(w[i - 2], 19) ^ (w[i - 2] >> 10);
                w[i] = w[i - 16] + s0 + w[i - 7] + s1;
            }

            uint32_t a = h0, b = h1, c = h2, d = h3, e = h4, f = h5, g = h6, h = h7;

            for (int i = 0; i < 64; i++) {
                uint32_t S1 = RIGHTROTATE(e, 6) ^ RIGHTROTATE(e, 11) ^ RIGHTROTATE(e, 25);
                uint32_t ch = (e & f) ^ ((~e) & g);
                uint32_t temp1 = h + S1 + ch + k[i] + w[i];
                uint32_t S0 = RIGHTROTATE(a, 2) ^ RIGHTROTATE(a, 13) ^ RIGHTROTATE(a, 22);
                uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
                uint32_t temp2 = S0 + maj;

                h = g;
                g = f;
                f = e;
                e = d + temp1;
                d = c;
                c = b;
                b = a;
                a = temp1 + temp2;
            }

            h0 += a;
            h1 += b;
            h2 += c;
            h3 += d;
            h4 += e;
            h5 += f;
            h6 += g;
            h7 += h;
        }

        std::stringstream ss;
        ss << std::hex << std::setfill('0');
        ss << std::setw(8) << h0 << std::setw(8) << h1 << std::setw(8) << h2 << std::setw(8) << h3;
        ss << std::setw(8) << h4 << std::setw(8) << h5 << std::setw(8) << h6 << std::setw(8) << h7;
        return ss.str();
    }

    // Fast xxHash64 implementation
    const uint64_t PRIME64_1 = 11400714785074694791ULL;
    const uint64_t PRIME64_2 = 14029467366386228027ULL;
    const uint64_t PRIME64_3 = 1609587929392839161ULL;
    const uint64_t PRIME64_4 = 9650029242287828579ULL;
    const uint64_t PRIME64_5 = 2870177450012600261ULL;

    uint64_t ComputeXXHash64(const unsigned char* data, size_t size) {
        uint64_t h64;
        const unsigned char* p = data;
        const unsigned char* bEnd = data + size;

        if (size >= 32) {
            const unsigned char* limit = bEnd - 32;
            uint64_t v1 = PRIME64_1 + PRIME64_2;
            uint64_t v2 = PRIME64_2;
            uint64_t v3 = 0;
            uint64_t v4 = 0 - PRIME64_1;

            do {
                auto read64 = [](const unsigned char* ptr) -> uint64_t {
                    return ((uint64_t)ptr[0]) |
                           (((uint64_t)ptr[1]) << 8) |
                           (((uint64_t)ptr[2]) << 16) |
                           (((uint64_t)ptr[3]) << 24) |
                           (((uint64_t)ptr[4]) << 32) |
                           (((uint64_t)ptr[5]) << 40) |
                           (((uint64_t)ptr[6]) << 48) |
                           (((uint64_t)ptr[7]) << 56);
                };

                v1 += read64(p) * PRIME64_2;
                v1 = RIGHTROTATE(v1, 31) * PRIME64_1;
                p += 8;

                v2 += read64(p) * PRIME64_2;
                v2 = RIGHTROTATE(v2, 31) * PRIME64_1;
                p += 8;

                v3 += read64(p) * PRIME64_2;
                v3 = RIGHTROTATE(v3, 31) * PRIME64_1;
                p += 8;

                v4 += read64(p) * PRIME64_2;
                v4 = RIGHTROTATE(v4, 31) * PRIME64_1;
                p += 8;
            } while (p <= limit);

            h64 = RIGHTROTATE(v1, 1) + RIGHTROTATE(v2, 7) + RIGHTROTATE(v3, 12) + RIGHTROTATE(v4, 18);
        } else {
            h64 = PRIME64_5;
        }

        h64 += size;

        while (p + 8 <= bEnd) {
            auto read64 = [](const unsigned char* ptr) -> uint64_t {
                return ((uint64_t)ptr[0]) | (((uint64_t)ptr[1]) << 8) |
                       (((uint64_t)ptr[2]) << 16) | (((uint64_t)ptr[3]) << 24) |
                       (((uint64_t)ptr[4]) << 32) | (((uint64_t)ptr[5]) << 40) |
                       (((uint64_t)ptr[6]) << 48) | (((uint64_t)ptr[7]) << 56);
            };
            h64 ^= read64(p) * PRIME64_2;
            h64 = RIGHTROTATE(h64, 31) * PRIME64_1;
            p += 8;
        }

        while (p < bEnd) {
            h64 ^= (*p) * PRIME64_5;
            h64 = RIGHTROTATE(h64, 11) * PRIME64_1;
            p++;
        }

        h64 ^= h64 >> 33;
        h64 *= PRIME64_2;
        h64 ^= h64 >> 29;
        return h64;
    }
}

AssetHash::Hash AssetHash::ComputeHash(const std::string& filepath) {
    Hash hash;
    std::ifstream file(filepath, std::ios::binary);

    if (!file.is_open()) {
        return hash;  // Return empty hash on error
    }

    // Read entire file into memory
    file.seekg(0, std::ios::end);
    size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<unsigned char> buffer(fileSize);
    file.read(reinterpret_cast<char*>(buffer.data()), fileSize);
    file.close();

    hash = ComputeHashFromData(buffer.data(), buffer.size());

    // Get modification time
    auto lastWriteTime = std::filesystem::last_write_time(filepath);
    auto sctp = std::chrono::time_point_cast<std::chrono::system_clock::duration>(
        lastWriteTime - std::filesystem::file_time_type::clock::now() + std::chrono::system_clock::now()
    );
    auto tt = std::chrono::system_clock::to_time_t(sctp);
    std::stringstream ss;
    ss << std::put_time(std::gmtime(&tt), "%Y-%m-%d %H:%M:%S");
    hash.timestamp = ss.str();
    hash.fileSize = fileSize;

    return hash;
}

AssetHash::Hash AssetHash::ComputeHashFromData(const unsigned char* data, size_t size) {
    Hash hash;
    hash.sha256 = ComputeSHA256(data, size);
    hash.xxHash64 = ComputeXXHash64(data, size);
    hash.fileSize = size;
    return hash;
}

uint64_t AssetHash::ComputeQuickHash(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        return 0;
    }

    file.seekg(0, std::ios::end);
    size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<unsigned char> buffer(fileSize);
    file.read(reinterpret_cast<char*>(buffer.data()), fileSize);
    file.close();

    return ComputeXXHash64(buffer.data(), buffer.size());
}

uint64_t AssetHash::ComputeQuickHashFromData(const unsigned char* data, size_t size) {
    return ComputeXXHash64(data, size);
}

bool AssetHash::HasFileChanged(const std::string& filepath, const Hash& previousHash) {
    if (!std::filesystem::exists(filepath)) {
        return true;
    }

    Hash currentHash = ComputeHash(filepath);
    return currentHash != previousHash;
}

std::vector<uint64_t> AssetHash::ComputeChunkedHash(const std::string& filepath, size_t chunkSize) {
    std::vector<uint64_t> chunkHashes;
    std::ifstream file(filepath, std::ios::binary);

    if (!file.is_open()) {
        return chunkHashes;
    }

    std::vector<unsigned char> chunk(chunkSize);
    while (file.read(reinterpret_cast<char*>(chunk.data()), chunkSize) || file.gcount() > 0) {
        size_t bytesRead = file.gcount();
        chunkHashes.push_back(ComputeXXHash64(chunk.data(), bytesRead));
    }

    file.close();
    return chunkHashes;
}

bool AssetHash::VerifyIntegrity(const std::string& filepath, const Hash& storedHash) {
    return HasFileChanged(filepath, storedHash) == false;  // File hasn't changed = integrity OK
}

std::string AssetHash::ToString(const Hash& hash) {
    std::stringstream ss;
    ss << "SHA256:" << hash.sha256 << "|xxHash:" << std::hex << hash.xxHash64
       << "|Size:" << std::dec << hash.fileSize;
    return ss.str();
}

AssetHash::Hash AssetHash::FromString(const std::string& str) {
    Hash hash;
    // Simple parsing - assumes format from ToString()
    size_t pos = 0;

    if (str.find("SHA256:") != std::string::npos) {
        pos = str.find("SHA256:") + 7;
        size_t endPos = str.find("|", pos);
        hash.sha256 = str.substr(pos, endPos - pos);

        pos = str.find("xxHash:", pos) + 7;
        endPos = str.find("|", pos);
        hash.xxHash64 = std::stoull(str.substr(pos, endPos - pos), nullptr, 16);

        pos = str.find("Size:", pos) + 5;
        hash.fileSize = std::stoull(str.substr(pos));
    }

    return hash;
}
