#include "StateSerializer.h"
#ifdef USE_PHYSX
#include "PhysXSoftBody.h"
#endif
#include <cstring>
#include <algorithm>
#include <iostream>

#ifdef HAS_LZ4
    #include <lz4.h>
    #include <lz4hc.h>
#endif

struct StateSerializer::Impl {
    SerializationStats stats;
    
#ifdef HAS_LZ4
    // Production LZ4 compression
    std::vector<uint8_t> CompressLZ4(const std::vector<uint8_t>& data) {
        if (data.empty()) return {};
        
        int maxCompressedSize = LZ4_compressBound(static_cast<int>(data.size()));
        std::vector<uint8_t> compressed(maxCompressedSize + 8); // +8 for size header
        
        // Write uncompressed size header
        uint32_t uncompressedSize = static_cast<uint32_t>(data.size());
        std::memcpy(compressed.data(), &uncompressedSize, 4);
        
        // Write magic number to identify LZ4 compression
        uint32_t magic = 0x184D2204; // LZ4 magic number
        std::memcpy(compressed.data() + 4, &magic, 4);
        
        // Compress using LZ4 HC for better ratio (slower but acceptable for network)
        int compressedSize = LZ4_compress_HC(
            reinterpret_cast<const char*>(data.data()),
            reinterpret_cast<char*>(compressed.data() + 8),
            static_cast<int>(data.size()),
            maxCompressedSize,
            LZ4HC_CLEVEL_DEFAULT
        );
        
        if (compressedSize <= 0) {
            std::cerr << "LZ4 compression failed" << std::endl;
            return data; // Return uncompressed on failure
        }
        
        compressed.resize(compressedSize + 8);
        return compressed;
    }
    
    std::vector<uint8_t> DecompressLZ4(const std::vector<uint8_t>& data) {
        if (data.size() < 8) return {};
        
        // Read uncompressed size
        uint32_t uncompressedSize;
        std::memcpy(&uncompressedSize, data.data(), 4);
        
        // Verify magic number
        uint32_t magic;
        std::memcpy(&magic, data.data() + 4, 4);
        if (magic != 0x184D2204) {
            std::cerr << "Invalid LZ4 magic number" << std::endl;
            return {};
        }
        
        std::vector<uint8_t> decompressed(uncompressedSize);
        
        int result = LZ4_decompress_safe(
            reinterpret_cast<const char*>(data.data() + 8),
            reinterpret_cast<char*>(decompressed.data()),
            static_cast<int>(data.size() - 8),
            static_cast<int>(uncompressedSize)
        );
        
        if (result < 0) {
            std::cerr << "LZ4 decompression failed" << std::endl;
            return {};
        }
        
        return decompressed;
    }
#else
    // Simple RLE compression (placeholder for LZ4)
    std::vector<uint8_t> CompressRLE(const std::vector<uint8_t>& data) {
        std::vector<uint8_t> compressed;
        
        if (data.empty()) return compressed;
        
        size_t i = 0;
        while (i < data.size()) {
            uint8_t value = data[i];
            size_t count = 1;
            
            while (i + count < data.size() && data[i + count] == value && count < 255) {
                count++;
            }
            
            compressed.push_back(static_cast<uint8_t>(count));
            compressed.push_back(value);
            i += count;
        }
        
        return compressed;
    }
    
    std::vector<uint8_t> DecompressRLE(const std::vector<uint8_t>& data) {
        std::vector<uint8_t> decompressed;
        
        for (size_t i = 0; i + 1 < data.size(); i += 2) {
            uint8_t count = data[i];
            uint8_t value = data[i + 1];
            
            for (uint8_t j = 0; j < count; ++j) {
                decompressed.push_back(value);
            }
        }
        
        return decompressed;
    }
#endif
};

StateSerializer::StateSerializer() : m_Impl(std::make_unique<Impl>()) {}

StateSerializer::~StateSerializer() = default;

#ifdef USE_PHYSX
std::vector<uint8_t> StateSerializer::SerializeSoftBody(const PhysXSoftBody* softBody,
                                                        const SerializationOptions& options) {
    if (!softBody) {
        return {};
    }
    
    std::vector<uint8_t> buffer;
    
    // Header: magic number + version
    WriteUInt32(buffer, 0x53425354);  // "SBST" = SoftBody STate
    WriteUInt32(buffer, 1);           // Version 1
    
    // Vertex count
    uint32_t vertexCount = static_cast<uint32_t>(softBody->GetVertexCount());
    WriteUInt32(buffer, vertexCount);
    
    // Tetrahedron count
    uint32_t tetraCount = static_cast<uint32_t>(softBody->GetTetrahedronCount());
    WriteUInt32(buffer, tetraCount);
    
    // Frame number
    WriteUInt64(buffer, 0);  // TODO: Get actual frame number
    
    // Flags
    uint8_t flags = 0;
    if (softBody->HasTearing()) flags |= 0x01;
    if (softBody->HasPlasticity()) flags |= 0x02;
    if (softBody->HasSelfCollision()) flags |= 0x04;
    buffer.push_back(flags);
    
    // Vertex positions
    const float* positions = softBody->GetVertexPositions();
    if (options.quantizePositions) {
        // Quantize to 16-bit
        for (uint32_t i = 0; i < vertexCount * 3; ++i) {
            uint16_t quantized = QuantizeFloat(positions[i], options.positionPrecision);
            buffer.push_back(quantized & 0xFF);
            buffer.push_back((quantized >> 8) & 0xFF);
        }
    } else {
        // Full precision
        for (uint32_t i = 0; i < vertexCount * 3; ++i) {
            WriteFloat(buffer, positions[i]);
        }
    }
    
    // Vertex velocities
    const float* velocities = softBody->GetVertexVelocities();
    if (options.quantizeVelocities) {
        // Quantize velocities
        for (uint32_t i = 0; i < vertexCount * 3; ++i) {
            uint16_t quantized = QuantizeFloat(velocities[i], options.velocityPrecision);
            buffer.push_back(quantized & 0xFF);
            buffer.push_back((quantized >> 8) & 0xFF);
        }
    } else {
        // Full precision
        for (uint32_t i = 0; i < vertexCount * 3; ++i) {
            WriteFloat(buffer, velocities[i]);
        }
    }
    
    // Compress if requested
    std::vector<uint8_t> result;
    if (options.compressData) {
        result = Compress(buffer);
        m_Impl->stats.totalBytesUncompressed += buffer.size();
        m_Impl->stats.totalBytesCompressed += result.size();
    } else {
        result = buffer;
        m_Impl->stats.totalBytesUncompressed += buffer.size();
        m_Impl->stats.totalBytesCompressed += buffer.size();
    }
    
    m_Impl->stats.totalSerializations++;
    m_Impl->stats.fullSerializations++;
    
    if (m_Impl->stats.totalBytesUncompressed > 0) {
        m_Impl->stats.avgCompressionRatio = 
            static_cast<float>(m_Impl->stats.totalBytesCompressed) / 
            m_Impl->stats.totalBytesUncompressed;
    }
    
    return result;
}

bool StateSerializer::DeserializeSoftBody(PhysXSoftBody* softBody, 
                                         const std::vector<uint8_t>& data) {
    if (!softBody || data.empty()) {
        return false;
    }
    
    // Check for compression and decompress if needed
    std::vector<uint8_t> buffer = data;
    
    if (data.size() >= 8) {
        uint32_t magic;
        std::memcpy(&magic, data.data() + 4, 4);
        if (magic == 0x184D2204) { // LZ4 magic
            buffer = Decompress(data);
            if (buffer.empty()) {
                std::cerr << "Failed to decompress data" << std::endl;
                return false;
            }
        }
    }
    
    const uint8_t* ptr = buffer.data();
    
    // Read header
    uint32_t magic = ReadUInt32(ptr);
    if (magic != 0x53425354) {
        std::cerr << "Invalid magic number in serialized data: 0x" 
                  << std::hex << magic << std::dec << std::endl;
        return false;
    }
    
    uint32_t version = ReadUInt32(ptr);
    if (version != 1) {
        std::cerr << "Unsupported version: " << version << std::endl;
        return false;
    }
    
    // Read counts
    uint32_t vertexCount = ReadUInt32(ptr);
    uint32_t tetraCount = ReadUInt32(ptr);
    uint64_t frameNumber = ReadUInt64(ptr);
    
    // Validate vertex count
    if (vertexCount != softBody->GetVertexCount()) {
        std::cerr << "Vertex count mismatch: expected " << softBody->GetVertexCount()
                  << ", got " << vertexCount << std::endl;
        return false;
    }
    
    // Read flags
    uint8_t flags = *ptr++;
    bool hasTearing = (flags & 0x01) != 0;
    bool hasPlasticity = (flags & 0x02) != 0;
    bool hasSelfCollision = (flags & 0x04) != 0;
    
    // Check if we have enough data
    size_t expectedSize = (ptr - buffer.data()) + (vertexCount * 3 * 4 * 2); // positions + velocities
    if (buffer.size() < expectedSize) {
        std::cerr << "Insufficient data: expected " << expectedSize 
                  << ", got " << buffer.size() << std::endl;
        return false;
    }
    
    // Read positions
    std::vector<float> positions(vertexCount * 3);
    for (uint32_t i = 0; i < vertexCount * 3; ++i) {
        positions[i] = ReadFloat(ptr);
    }
    
    // Read velocities
    std::vector<float> velocities(vertexCount * 3);
    for (uint32_t i = 0; i < vertexCount * 3; ++i) {
        velocities[i] = ReadFloat(ptr);
    }
    
    // Apply to soft body
    softBody->SetVertexPositions(positions.data(), vertexCount);
    softBody->SetVertexVelocities(velocities.data(), vertexCount);
    
    m_Impl->stats.totalDeserializations++;
    
    return true;
}

std::vector<uint8_t> StateSerializer::SerializeDelta(const PhysXSoftBody* softBody,
                                                     const SoftBodyState& previousState,
                                                     const SerializationOptions& options) {
    if (!softBody) {
        return {};
    }
    
    std::vector<uint8_t> buffer;
    
    // Header: magic number for delta + version
    WriteUInt32(buffer, 0x53424454);  // "SBDT" = SoftBody DelTa
    WriteUInt32(buffer, 1);           // Version 1
    
    // Frame number
    WriteUInt64(buffer, 0);  // TODO: Get actual frame number
    
    // Get current positions
    uint32_t vertexCount = static_cast<uint32_t>(softBody->GetVertexCount());
    const float* currentPos = softBody->GetVertexPositions();
    
    // Calculate deltas
    std::vector<float> currentPositions(currentPos, currentPos + vertexCount * 3);
    std::vector<float> deltaPositions = EncodeDelta(currentPositions, previousState.positions);
    
    // Write delta positions
    for (float delta : deltaPositions) {
        WriteFloat(buffer, delta);
    }
    
    // Get current velocities
    const float* currentVel = softBody->GetVertexVelocities();
    std::vector<float> currentVelocities(currentVel, currentVel + vertexCount * 3);
    std::vector<float> deltaVelocities = EncodeDelta(currentVelocities, previousState.velocities);
    
    // Write delta velocities
    for (float delta : deltaVelocities) {
        WriteFloat(buffer, delta);
    }
    
    // Compress if requested
    std::vector<uint8_t> result;
    if (options.compressData) {
        result = Compress(buffer);
        m_Impl->stats.totalBytesUncompressed += buffer.size();
        m_Impl->stats.totalBytesCompressed += result.size();
    } else {
        result = buffer;
    }
    
    m_Impl->stats.totalSerializations++;
    m_Impl->stats.deltaSerializations++;
    
    return result;
}

bool StateSerializer::ApplyDelta(PhysXSoftBody* softBody, 
                                const std::vector<uint8_t>& delta) {
    if (!softBody || delta.empty()) {
        return false;
    }
    
    // Decompress if needed
    std::vector<uint8_t> buffer = delta;
    
    // Check for compression
    if (buffer.size() >= 8) {
        uint32_t magic;
        std::memcpy(&magic, buffer.data() + 4, 4);
        if (magic == 0x184D2204) {
            buffer = Decompress(delta);
            if (buffer.empty()) {
                std::cerr << "Failed to decompress delta" << std::endl;
                return false;
            }
        }
    }
    
    const uint8_t* ptr = buffer.data();
    
    // Read header
    uint32_t magic = ReadUInt32(ptr);
    if (magic != 0x53424454) { // "SBDT"
        std::cerr << "Invalid delta magic number" << std::endl;
        return false;
    }
    
    uint32_t version = ReadUInt32(ptr);
    if (version != 1) {
        std::cerr << "Unsupported delta version: " << version << std::endl;
        return false;
    }
    
    uint64_t frameNumber = ReadUInt64(ptr);
    
    // Get current state
    uint32_t vertexCount = static_cast<uint32_t>(softBody->GetVertexCount());
    const float* currentPos = softBody->GetVertexPositions();
    const float* currentVel = softBody->GetVertexVelocities();
    
    std::vector<float> currentPositions(currentPos, currentPos + vertexCount * 3);
    std::vector<float> currentVelocities(currentVel, currentVel + vertexCount * 3);
    
    // Read delta positions
    std::vector<float> deltaPositions(vertexCount * 3);
    for (uint32_t i = 0; i < vertexCount * 3; ++i) {
        deltaPositions[i] = ReadFloat(ptr);
    }
    
    // Read delta velocities
    std::vector<float> deltaVelocities(vertexCount * 3);
    for (uint32_t i = 0; i < vertexCount * 3; ++i) {
        deltaVelocities[i] = ReadFloat(ptr);
    }
    
    // Apply deltas
    std::vector<float> newPositions = DecodeDelta(deltaPositions, currentPositions);
    std::vector<float> newVelocities = DecodeDelta(deltaVelocities, currentVelocities);
    
    // Update soft body
    softBody->SetVertexPositions(newPositions.data(), vertexCount);
    softBody->SetVertexVelocities(newVelocities.data(), vertexCount);
    
    m_Impl->stats.totalDeserializations++;
    
    return true;
}

StateSerializer::SoftBodyState StateSerializer::CaptureState(const PhysXSoftBody* softBody) {
    SoftBodyState state;
    
    if (!softBody) {
        return state;
    }
    
    state.vertexCount = static_cast<uint32_t>(softBody->GetVertexCount());
    state.tetrahedronCount = static_cast<uint32_t>(softBody->GetTetrahedronCount());
    state.hasTearing = softBody->HasTearing();
    state.hasPlasticity = softBody->HasPlasticity();
    state.hasSelfCollision = softBody->HasSelfCollision();
    
    // Copy positions
    const float* positions = softBody->GetVertexPositions();
    state.positions.assign(positions, positions + state.vertexCount * 3);
    
    // Copy velocities
    const float* velocities = softBody->GetVertexVelocities();
    state.velocities.assign(velocities, velocities + state.vertexCount * 3);
    
    return state;
}

void StateSerializer::RestoreState(PhysXSoftBody* softBody, const SoftBodyState& state) {
    if (!softBody) {
        return;
    }
    
    softBody->SetVertexPositions(state.positions.data(), state.vertexCount);
    softBody->SetVertexVelocities(state.velocities.data(), state.vertexCount);
}
#endif

std::vector<uint8_t> StateSerializer::Compress(const std::vector<uint8_t>& data) {
#ifdef HAS_LZ4
    return m_Impl->CompressLZ4(data);
#else
    // Fallback to RLE
    return m_Impl->CompressRLE(data);
#endif
}

std::vector<uint8_t> StateSerializer::Decompress(const std::vector<uint8_t>& data) {
#ifdef HAS_LZ4
    // Check for LZ4 magic number
    if (data.size() >= 8) {
        uint32_t magic;
        std::memcpy(&magic, data.data() + 4, 4);
        if (magic == 0x184D2204) {
            return m_Impl->DecompressLZ4(data);
        }
    }
    // Fall through to RLE if not LZ4
#endif
    return m_Impl->DecompressRLE(data);
}

StateSerializer::SerializationStats StateSerializer::GetStatistics() const {
    return m_Impl->stats;
}

void StateSerializer::ResetStatistics() {
    m_Impl->stats = SerializationStats();
}

// Helper methods

void StateSerializer::WriteFloat(std::vector<uint8_t>& buffer, float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(float));
    buffer.push_back(bits & 0xFF);
    buffer.push_back((bits >> 8) & 0xFF);
    buffer.push_back((bits >> 16) & 0xFF);
    buffer.push_back((bits >> 24) & 0xFF);
}

float StateSerializer::ReadFloat(const uint8_t*& ptr) {
    uint32_t bits = ptr[0] | (ptr[1] << 8) | (ptr[2] << 16) | (ptr[3] << 24);
    ptr += 4;
    float value;
    std::memcpy(&value, &bits, sizeof(float));
    return value;
}

void StateSerializer::WriteUInt32(std::vector<uint8_t>& buffer, uint32_t value) {
    buffer.push_back(value & 0xFF);
    buffer.push_back((value >> 8) & 0xFF);
    buffer.push_back((value >> 16) & 0xFF);
    buffer.push_back((value >> 24) & 0xFF);
}

uint32_t StateSerializer::ReadUInt32(const uint8_t*& ptr) {
    uint32_t value = ptr[0] | (ptr[1] << 8) | (ptr[2] << 16) | (ptr[3] << 24);
    ptr += 4;
    return value;
}

void StateSerializer::WriteUInt64(std::vector<uint8_t>& buffer, uint64_t value) {
    for (int i = 0; i < 8; ++i) {
        buffer.push_back((value >> (i * 8)) & 0xFF);
    }
}

uint64_t StateSerializer::ReadUInt64(const uint8_t*& ptr) {
    uint64_t value = 0;
    for (int i = 0; i < 8; ++i) {
        value |= static_cast<uint64_t>(ptr[i]) << (i * 8);
    }
    ptr += 8;
    return value;
}

uint16_t StateSerializer::QuantizeFloat(float value, float precision) {
    int32_t quantized = static_cast<int32_t>(value / precision);
    // Clamp to 16-bit range
    quantized = std::max(-32768, std::min(32767, quantized));
    return static_cast<uint16_t>(quantized + 32768);
}

float StateSerializer::DequantizeFloat(uint16_t quantized, float precision) {
    int32_t value = static_cast<int32_t>(quantized) - 32768;
    return value * precision;
}

std::vector<float> StateSerializer::EncodeDelta(const std::vector<float>& current,
                                               const std::vector<float>& previous) {
    std::vector<float> delta(current.size());
    for (size_t i = 0; i < current.size(); ++i) {
        delta[i] = current[i] - previous[i];
    }
    return delta;
}

std::vector<float> StateSerializer::DecodeDelta(const std::vector<float>& delta,
                                               const std::vector<float>& previous) {
    std::vector<float> current(delta.size());
    for (size_t i = 0; i < delta.size(); ++i) {
        current[i] = previous[i] + delta[i];
    }
    return current;
}
