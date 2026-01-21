#pragma once

#include <vector>
#include <cstdint>
#include <memory>

class PhysXSoftBody;

/**
 * @brief Serializes and deserializes soft body state for network transfer
 * 
 * Provides efficient serialization with delta encoding and compression
 * for minimal network bandwidth usage.
 */
class StateSerializer {
public:
    /**
     * @brief Soft body state snapshot
     */
    struct SoftBodyState {
        // Vertex data
        std::vector<float> positions;      // x,y,z per vertex
        std::vector<float> velocities;     // x,y,z per vertex
        
        // Simulation state
        uint32_t vertexCount;
        uint32_t tetrahedronCount;
        float timeStep;
        uint64_t frameNumber;
        
        // Flags
        bool hasTearing;
        bool hasPlasticity;
        bool hasSelfCollision;
        
        SoftBodyState() : vertexCount(0), tetrahedronCount(0), 
                         timeStep(0.0f), frameNumber(0),
                         hasTearing(false), hasPlasticity(false), 
                         hasSelfCollision(false) {}
    };
    
    /**
     * @brief Serialization options
     */
    struct SerializationOptions {
        bool compressData = true;           // Use LZ4 compression
        bool deltaEncoding = true;          // Use delta encoding for positions
        bool quantizePositions = false;     // Quantize to 16-bit (lossy)
        bool quantizeVelocities = false;    // Quantize velocities (lossy)
        float positionPrecision = 0.001f;   // Precision for quantization (mm)
        float velocityPrecision = 0.01f;    // Precision for velocity quantization
    };
    
    StateSerializer();
    ~StateSerializer();
    
    // Full serialization
    
    /**
     * @brief Serialize complete soft body state
     * @param softBody Soft body to serialize
     * @param options Serialization options
     * @return Serialized data
     */
    std::vector<uint8_t> SerializeSoftBody(const PhysXSoftBody* softBody,
                                          const SerializationOptions& options = {});
    
    /**
     * @brief Deserialize complete soft body state
     * @param softBody Target soft body
     * @param data Serialized data
     * @param options Serialization options
     * @return True if successful
     */
    bool DeserializeSoftBody(PhysXSoftBody* softBody, 
                            const std::vector<uint8_t>& data,
                            const SerializationOptions& options = {});
    
    // Delta serialization
    
    /**
     * @brief Serialize delta (changes since previous state)
     * @param softBody Current soft body state
     * @param previousState Previous state snapshot
     * @param options Serialization options
     * @return Serialized delta data
     */
    std::vector<uint8_t> SerializeDelta(const PhysXSoftBody* softBody,
                                        const SoftBodyState& previousState,
                                        const SerializationOptions& options = {});
    
    /**
     * @brief Apply delta to soft body
     * @param softBody Target soft body
     * @param delta Serialized delta data
     * @param options Serialization options
     * @return True if successful
     */
    bool ApplyDelta(PhysXSoftBody* softBody, 
                   const std::vector<uint8_t>& delta,
                   const SerializationOptions& options = {});
    
    // State management
    
    /**
     * @brief Capture current soft body state
     * @param softBody Soft body to capture
     * @return State snapshot
     */
    SoftBodyState CaptureState(const PhysXSoftBody* softBody);
    
    /**
     * @brief Restore soft body from state
     * @param softBody Target soft body
     * @param state State to restore
     */
    void RestoreState(PhysXSoftBody* softBody, const SoftBodyState& state);
    
    // Compression
    
    /**
     * @brief Compress data using LZ4
     * @param data Input data
     * @return Compressed data
     */
    std::vector<uint8_t> Compress(const std::vector<uint8_t>& data);
    
    /**
     * @brief Decompress LZ4 data
     * @param data Compressed data
     * @return Decompressed data
     */
    std::vector<uint8_t> Decompress(const std::vector<uint8_t>& data);
    
    // Statistics
    
    /**
     * @brief Serialization statistics
     */
    struct SerializationStats {
        size_t totalSerializations;
        size_t totalDeserializations;
        size_t totalBytesUncompressed;
        size_t totalBytesCompressed;
        float avgCompressionRatio;
        size_t deltaSerializations;
        size_t fullSerializations;
    };
    
    /**
     * @brief Get serialization statistics
     * @return Statistics
     */
    SerializationStats GetStatistics() const;
    
    /**
     * @brief Reset statistics
     */
    void ResetStatistics();

private:
    struct Impl;
    std::unique_ptr<Impl> m_Impl;
    
    // Internal helpers
    void WriteFloat(std::vector<uint8_t>& buffer, float value);
    float ReadFloat(const uint8_t*& ptr);
    void WriteUInt32(std::vector<uint8_t>& buffer, uint32_t value);
    uint32_t ReadUInt32(const uint8_t*& ptr);
    void WriteUInt64(std::vector<uint8_t>& buffer, uint64_t value);
    uint64_t ReadUInt64(const uint8_t*& ptr);
    
    // Quantization helpers
    uint16_t QuantizeFloat(float value, float precision);
    float DequantizeFloat(uint16_t quantized, float precision);
    
    // Delta encoding helpers
    std::vector<float> EncodeDelta(const std::vector<float>& current,
                                   const std::vector<float>& previous);
    std::vector<float> DecodeDelta(const std::vector<float>& delta,
                                   const std::vector<float>& previous);
};
