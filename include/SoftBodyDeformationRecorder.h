#pragma once

#include "Math/Vec3.h"
#include <vector>
#include <string>
#include <mutex>
#include <nlohmann/json.hpp>

/**
 * @brief Recording state for deformation recorder
 */
enum class RecordingState {
    Stopped,
    Recording,
    Paused
};

/**
 * @brief Playback state for deformation recorder
 */
enum class PlaybackState {
    Stopped,
    Playing,
    Paused
};

/**
 * @brief Compression mode for keyframe storage
 */
enum class CompressionMode {
    None,           // No compression, full precision
    DeltaEncoding,  // Store differences from previous frame
    Quantized       // Reduce precision to 16-bit
};

/**
 * @brief Loop mode for playback
 */
enum class LoopMode {
    None,      // Stop at end
    Loop,      // Restart from beginning
    PingPong   // Reverse direction at end
};

/**
 * @brief Interpolation mode for playback
 */
enum class InterpolationMode {
    Linear,  // Fast, simple linear interpolation
    Cubic    // Smooth Catmull-Rom spline interpolation
};

/**
 * @brief Quantized position (16-bit)
 */
struct QuantizedVec3_16 {
    int16_t x, y, z;
};

/**
 * @brief Quantized position (8-bit)
 */
struct QuantizedVec3_8 {
    int8_t x, y, z;
};

/**
 * @brief Compression metadata
 */
struct CompressionMetadata {
    Vec3 boundsMin;              // Bounding box minimum
    Vec3 boundsMax;              // Bounding box maximum
    Vec3 quantizationScale;      // Scale factor for quantization
    int referenceFrameInterval;  // Frames between reference frames
    
    CompressionMetadata() 
        : boundsMin(0, 0, 0)
        , boundsMax(0, 0, 0)
        , quantizationScale(1, 1, 1)
        , referenceFrameInterval(30) {}
};

/**
 * @brief Sparse vertex data (index + position)
 */
struct SparseVertex {
    int index;      // Vertex index
    Vec3 position;  // Vertex position
    
    SparseVertex() : index(0), position(0, 0, 0) {}
    SparseVertex(int idx, const Vec3& pos) : index(idx), position(pos) {}
};

/**
 * @brief Single keyframe storing deformation state
 */
struct DeformationKeyframe {
    float timestamp;                    // Time in seconds
    bool isReferenceFrame;              // True if this is a reference frame for delta encoding
    bool isSparse;                      // True if using sparse storage (changed vertices only)
    
    // Full precision data (for reference frames or uncompressed)
    std::vector<Vec3> positions;        // Vertex positions
    std::vector<Vec3> velocities;       // Optional: vertex velocities
    
    // Sparse storage (changed vertices only)
    std::vector<SparseVertex> sparsePositions;  // Changed vertices with indices
    std::vector<int> changedIndices;            // Quick lookup of changed indices
    
    // Compressed data (for delta/quantized frames)
    std::vector<QuantizedVec3_16> quantizedPositions16;  // 16-bit quantized positions
    std::vector<QuantizedVec3_8> quantizedPositions8;    // 8-bit quantized positions
    std::vector<Vec3> deltaPositions;                    // Delta from reference frame
    
    bool hasVelocities;                 // Whether velocities are stored
    
    DeformationKeyframe() : timestamp(0.0f), isReferenceFrame(true), isSparse(false), hasVelocities(false) {}
};

/**
 * @brief Keyframe chunk for streaming large recordings
 */
struct KeyframeChunk {
    int startIndex;                              // First keyframe index in chunk
    int count;                                   // Number of keyframes in chunk
    float startTime;                             // Start timestamp
    float endTime;                               // End timestamp
    size_t fileOffset;                           // Offset in file
    size_t compressedSize;                       // Size in bytes
    mutable bool isLoaded;                       // Whether chunk is in memory
    mutable std::vector<DeformationKeyframe> keyframes;  // Actual data when loaded
    mutable int lastAccessFrame;                 // For LRU eviction
    
    KeyframeChunk() 
        : startIndex(0), count(0), startTime(0.0f), endTime(0.0f)
        , fileOffset(0), compressedSize(0), isLoaded(false), lastAccessFrame(0) {}
};

/**
 * @brief Deformation recorder for PhysX soft bodies
 * 
 * Records and plays back soft body deformations with configurable
 * compression, sample rate, and memory management.
 */
class SoftBodyDeformationRecorder {
public:
    SoftBodyDeformationRecorder();
    ~SoftBodyDeformationRecorder();
    
    // Recording controls
    /**
     * @brief Start recording deformation
     * @param vertexCount Number of vertices in the soft body
     * @param recordVelocities Whether to record velocities (increases size)
     */
    void StartRecording(int vertexCount, bool recordVelocities = false);
    
    /**
     * @brief Stop recording
     */
    void StopRecording();
    
    /**
     * @brief Pause recording
     */
    void PauseRecording();
    
    /**
     * @brief Resume recording after pause
     */
    void ResumeRecording();
    
    /**
     * @brief Record a frame
     * @param positions Vertex positions
     * @param velocities Vertex velocities (can be nullptr if not recording velocities)
     * @param timestamp Current time in seconds
     */
    void RecordFrame(const Vec3* positions, const Vec3* velocities, float timestamp);
    
    /**
     * @brief Check if currently recording
     */
    bool IsRecording() const { return m_RecordingState == RecordingState::Recording; }
    
    /**
     * @brief Get recording state
     */
    RecordingState GetRecordingState() const { return m_RecordingState; }
    
    // Playback controls
    /**
     * @brief Start playback from beginning
     */
    void StartPlayback();
    
    /**
     * @brief Stop playback
     */
    void StopPlayback();
    
    /**
     * @brief Pause playback
     */
    void PausePlayback();
    
    /**
     * @brief Resume playback after pause
     */
    void ResumePlayback();
    
    /**
     * @brief Seek to specific time
     * @param time Time in seconds
     */
    void SeekPlayback(float time);
    
    /**
     * @brief Update playback and get interpolated positions
     * @param deltaTime Time since last update
     * @param outPositions Output buffer for interpolated positions
     * @return True if playback is active, false if stopped
     */
    bool UpdatePlayback(float deltaTime, Vec3* outPositions);
    
    /**
     * @brief Check if currently playing back
     */
    bool IsPlayingBack() const { return m_PlaybackState == PlaybackState::Playing; }
    
    /**
     * @brief Get playback state
     */
    PlaybackState GetPlaybackState() const { return m_PlaybackState; }
    
    // Configuration
    /**
     * @brief Set sample rate for recording
     * @param samplesPerSecond Samples per second (default: 60)
     */
    void SetSampleRate(float samplesPerSecond);
    
    /**
     * @brief Get sample rate
     */
    float GetSampleRate() const { return m_SampleRate; }
    
    /**
     * @brief Set compression mode
     * @param mode Compression mode
     */
    void SetCompressionMode(CompressionMode mode);
    
    /**
     * @brief Get compression mode
     */
    CompressionMode GetCompressionMode() const { return m_CompressionMode; }
    
    /**
     * @brief Set maximum number of keyframes
     * @param maxKeyframes Maximum keyframes (0 = unlimited)
     */
    void SetMaxKeyframes(int maxKeyframes);
    
    /**
     * @brief Get maximum keyframes
     */
    int GetMaxKeyframes() const { return m_MaxKeyframes; }
    
    /**
     * @brief Set playback speed multiplier
     * @param speed Speed multiplier (1.0 = normal, 2.0 = double speed, etc.)
     */
    void SetPlaybackSpeed(float speed);
    
    /**
     * @brief Get playback speed
     */
    float GetPlaybackSpeed() const { return m_PlaybackSpeed; }
    
    /**
     * @brief Set loop mode
     * @param mode Loop mode
     */
    void SetLoopMode(LoopMode mode);
    
    /**
     * @brief Get loop mode
     */
    LoopMode GetLoopMode() const { return m_LoopMode; }
    
    /**
     * @brief Set interpolation mode for playback
     * @param mode Interpolation mode (Linear or Cubic)
     */
    void SetInterpolationMode(InterpolationMode mode);
    
    /**
     * @brief Get interpolation mode
     */
    InterpolationMode GetInterpolationMode() const { return m_InterpolationMode; }
    
    /**
     * @brief Enable/disable partial recording (only changed vertices)
     * @param enabled True to only record changed vertices
     */
    void SetPartialRecording(bool enabled);
    
    /**
     * @brief Check if partial recording is enabled
     */
    bool IsPartialRecording() const { return m_PartialRecording; }
    
    /**
     * @brief Set change detection threshold
     * @param threshold Minimum distance to consider vertex changed (default: 0.001)
     */
    void SetChangeThreshold(float threshold);
    
    /**
     * @brief Get change threshold
     */
    float GetChangeThreshold() const { return m_ChangeThreshold; }
    
    // Query
    /**
     * @brief Get number of recorded keyframes
     */
    int GetKeyframeCount() const { return static_cast<int>(m_Keyframes.size()); }
    
    /**
     * @brief Get total duration of recording
     */
    float GetDuration() const;
    
    /**
     * @brief Get current playback time
     */
    float GetPlaybackTime() const { return m_PlaybackTime; }
    
    /**
     * @brief Get vertex count
     */
    int GetVertexCount() const { return m_VertexCount; }
    
    /**
     * @brief Check if recording has velocities
     */
    bool HasVelocities() const { return m_RecordVelocities; }
    
    /**
     * @brief Clear all recorded keyframes
     */
    void Clear();
    
    // Compression
    /**
     * @brief Set reference frame interval for delta encoding
     * @param interval Frames between reference frames (0 = every frame is reference)
     */
    void SetReferenceFrameInterval(int interval);
    
    /**
     * @brief Get reference frame interval
     */
    int GetReferenceFrameInterval() const { return m_ReferenceFrameInterval; }
    
    /**
     * @brief Compression statistics
     */
    struct CompressionStats {
        size_t uncompressedSize;    // Estimated uncompressed size in bytes
        size_t compressedSize;      // Actual compressed size in bytes
        float compressionRatio;     // Ratio (compressed / uncompressed)
        float averageError;         // Average quantization error (if applicable)
        int referenceFrameCount;    // Number of reference frames
        int deltaFrameCount;        // Number of delta frames
    };
    
    /**
     * @brief Get compression statistics
     */
    CompressionStats GetCompressionStats() const;
    
    /**
     * @brief Partial recording statistics
     */
    struct PartialRecordingStats {
        float averageChangedRatio;      // Average % of vertices changed per frame
        size_t totalVerticesStored;     // Total vertices stored across all frames
        size_t totalVerticesPossible;   // Total if all vertices stored
        float compressionRatio;         // Ratio of actual/possible
        int sparseFrameCount;           // Number of sparse frames
        int denseFrameCount;            // Number of dense frames
    };
    
    /**
     * @brief Get partial recording statistics
     */
    PartialRecordingStats GetPartialRecordingStats() const;
    
    // Streaming
    /**
     * @brief Enable/disable streaming mode for large recordings
     * @param enabled True to use streaming
     * @param memoryBudgetMB Maximum memory for keyframe cache in MB (default: 100)
     */
    void SetStreamingMode(bool enabled, int memoryBudgetMB = 100);
    
    /**
     * @brief Check if streaming mode is enabled
     */
    bool IsStreamingMode() const { return m_StreamingMode; }
    
    /**
     * @brief Save recording with streaming (for large files)
     * @param filename Output file path
     * @param progressCallback Optional callback(current, total) for progress
     * @return True if successful
     */
    bool SaveStreaming(const std::string& filename, 
                      std::function<void(int, int)> progressCallback = nullptr);
    
    /**
     * @brief Load recording with streaming (for large files)
     * @param filename Input file path
     * @return True if successful
     */
    bool LoadStreaming(const std::string& filename);
    
    /**
     * @brief Streaming statistics
     */
    struct StreamingStats {
        size_t totalChunks;         // Total number of chunks
        size_t loadedChunks;        // Currently loaded chunks
        size_t cacheHits;           // Number of cache hits
        size_t cacheMisses;         // Number of cache misses
        size_t memoryUsage;         // Current memory usage in bytes
        size_t memoryBudget;        // Memory budget in bytes
        float cacheHitRatio;        // Hit ratio (hits / (hits + misses))
    };
    
    /**
     * @brief Get streaming statistics
     */
    StreamingStats GetStreamingStats() const;
    
    // Serialization
    /**
     * @brief Serialize to JSON
     */
    nlohmann::json ToJson() const;
    
    /**
     * @brief Deserialize from JSON
     */
    bool FromJson(const nlohmann::json& json);
    
    /**
     * @brief Save to file
     * @param filename Output file path
     * @param binary Use binary format (more efficient)
     * @return True if successful
     */
    bool SaveToFile(const std::string& filename, bool binary = false) const;
    
    /**
     * @brief Load from file
     * @param filename Input file path
     * @return True if successful
     */
    bool LoadFromFile(const std::string& filename);
    
private:
    // Recording state
    RecordingState m_RecordingState;
    int m_VertexCount;
    bool m_RecordVelocities;
    float m_LastRecordTime;
    float m_SampleInterval;  // 1.0 / m_SampleRate
    
    // Playback state
    PlaybackState m_PlaybackState;
    float m_PlaybackTime;
    float m_PlaybackSpeed;
    LoopMode m_LoopMode;
    bool m_PlaybackReversed;  // For ping-pong mode
    
    // Configuration
    float m_SampleRate;
    CompressionMode m_CompressionMode;
    int m_MaxKeyframes;
    InterpolationMode m_InterpolationMode;
    bool m_PartialRecording;
    float m_ChangeThreshold;
    
    // Data
    std::vector<DeformationKeyframe> m_Keyframes;
    mutable std::mutex m_Mutex;  // Thread safety
    
    // Compression
    CompressionMetadata m_CompressionMetadata;
    int m_ReferenceFrameInterval;
    int m_FramesSinceReference;
    std::vector<Vec3> m_LastReferencePositions;
    
    // Partial recording
    std::vector<Vec3> m_LastFramePositions;
    
    // Streaming
    bool m_StreamingMode;
    size_t m_MemoryBudgetBytes;
    int m_ChunkSize;  // Keyframes per chunk
    std::vector<KeyframeChunk> m_Chunks;
    std::string m_StreamingFilename;
    mutable size_t m_CacheHits;
    mutable size_t m_CacheMisses;
    mutable int m_CurrentPlaybackFrame;
    
    // Helper methods
    void CullOldestKeyframe();
    void InterpolatePositions(float time, Vec3* outPositions) const;
    void InterpolateLinear(float time, Vec3* outPositions) const;
    void InterpolateCubic(float time, Vec3* outPositions) const;
    Vec3 CatmullRom(const Vec3& p0, const Vec3& p1, const Vec3& p2, const Vec3& p3, float t) const;
    int FindKeyframeIndex(float time) const;
    void CompressKeyframe(DeformationKeyframe& keyframe);
    void DecompressKeyframe(DeformationKeyframe& keyframe) const;
    
    // Compression helpers
    void CalculateBounds();
    void UpdateBounds(const Vec3* positions, int count);
    QuantizedVec3_16 QuantizePosition16(const Vec3& pos) const;
    QuantizedVec3_8 QuantizePosition8(const Vec3& pos) const;
    Vec3 DequantizePosition16(const QuantizedVec3_16& qpos) const;
    Vec3 DequantizePosition8(const QuantizedVec3_8& qpos) const;
    void ApplyDeltaEncoding(DeformationKeyframe& keyframe);
    void ApplyQuantization(DeformationKeyframe& keyframe);
    void ReconstructFromDelta(int keyframeIndex, Vec3* outPositions) const;
    
    // Partial recording helpers
    void DetectChangedVertices(const Vec3* positions, std::vector<int>& changedIndices);
    void ReconstructFromSparse(int keyframeIndex, Vec3* outPositions) const;
    
    // Streaming helpers
    void LoadChunk(int chunkIndex) const;
    void UnloadChunk(int chunkIndex) const;
    void EvictLRUChunk() const;
    int FindChunkForTime(float time) const;
    int FindChunkForIndex(int keyframeIndex) const;
    size_t GetCacheMemoryUsage() const;
    size_t EstimateChunkSize(const KeyframeChunk& chunk) const;
    
    // Binary serialization helpers
    bool SaveBinary(const std::string& filename) const;
    bool LoadBinary(const std::string& filename);
};
