#include "SoftBodyDeformationRecorder.h"
#include <fstream>
#include <algorithm>
#include <cstring>

SoftBodyDeformationRecorder::SoftBodyDeformationRecorder()
    : m_RecordingState(RecordingState::Stopped)
    , m_VertexCount(0)
    , m_RecordVelocities(false)
    , m_LastRecordTime(0.0f)
    , m_SampleInterval(1.0f / 60.0f)
    , m_PlaybackState(PlaybackState::Stopped)
    , m_PlaybackTime(0.0f)
    , m_PlaybackSpeed(1.0f)
    , m_LoopMode(LoopMode::None)
    , m_PlaybackReversed(false)
    , m_SampleRate(60.0f)
    , m_CompressionMode(CompressionMode::None)
    , m_MaxKeyframes(1000)
    , m_ReferenceFrameInterval(30)
    , m_FramesSinceReference(0)
    , m_InterpolationMode(InterpolationMode::Linear)
    , m_PartialRecording(false)
    , m_ChangeThreshold(0.001f)
    , m_StreamingMode(false)
    , m_MemoryBudgetBytes(100 * 1024 * 1024)  // 100 MB default
    , m_ChunkSize(100)
    , m_CacheHits(0)
    , m_CacheMisses(0)
    , m_CurrentPlaybackFrame(0)
{
}

SoftBodyDeformationRecorder::~SoftBodyDeformationRecorder()
{
    Clear();
}

void SoftBodyDeformationRecorder::StartRecording(int vertexCount, bool recordVelocities)
{
    std::lock_guard<std::mutex> lock(m_Mutex);
    
    m_VertexCount = vertexCount;
    m_RecordVelocities = recordVelocities;
    m_RecordingState = RecordingState::Recording;
    m_LastRecordTime = 0.0f;
    
    // Clear existing keyframes
    m_Keyframes.clear();
}

void SoftBodyDeformationRecorder::StopRecording()
{
    std::lock_guard<std::mutex> lock(m_Mutex);
    m_RecordingState = RecordingState::Stopped;
}

void SoftBodyDeformationRecorder::PauseRecording()
{
    std::lock_guard<std::mutex> lock(m_Mutex);
    if (m_RecordingState == RecordingState::Recording) {
        m_RecordingState = RecordingState::Paused;
    }
}

void SoftBodyDeformationRecorder::ResumeRecording()
{
    std::lock_guard<std::mutex> lock(m_Mutex);
    if (m_RecordingState == RecordingState::Paused) {
        m_RecordingState = RecordingState::Recording;
    }
}

void SoftBodyDeformationRecorder::RecordFrame(const Vec3* positions, const Vec3* velocities, float timestamp)
{
    std::lock_guard<std::mutex> lock(m_Mutex);
    
    if (m_RecordingState != RecordingState::Recording) {
        return;
    }
    
    // Check if enough time has passed since last recording
    if (!m_Keyframes.empty() && (timestamp - m_LastRecordTime) < m_SampleInterval) {
        return;
    }
    
    // Create new keyframe
    DeformationKeyframe keyframe;
    keyframe.timestamp = timestamp;
    keyframe.hasVelocities = m_RecordVelocities && velocities != nullptr;
    keyframe.isSparse = m_PartialRecording;
    
    if (m_PartialRecording) {
        // Partial recording - detect and store only changed vertices
        std::vector<int> changedIndices;
        DetectChangedVertices(positions, changedIndices);
        
        // Store sparse positions
        keyframe.sparsePositions.reserve(changedIndices.size());
        for (int idx : changedIndices) {
            keyframe.sparsePositions.emplace_back(idx, positions[idx]);
        }
        keyframe.changedIndices = changedIndices;
        
        // Note: Velocities not supported in sparse mode for simplicity
    } else {
        // Full recording - store all vertices
        keyframe.positions.resize(m_VertexCount);
        std::memcpy(keyframe.positions.data(), positions, m_VertexCount * sizeof(Vec3));
        
        // Copy velocities if needed
        if (keyframe.hasVelocities) {
            keyframe.velocities.resize(m_VertexCount);
            std::memcpy(keyframe.velocities.data(), velocities, m_VertexCount * sizeof(Vec3));
        }
    }
    
    // Apply compression if enabled (only for non-sparse frames)
    if (!m_PartialRecording && m_CompressionMode != CompressionMode::None) {
        CompressKeyframe(keyframe);
    }
    
    // Add keyframe
    m_Keyframes.push_back(std::move(keyframe));
    m_LastRecordTime = timestamp;
    
    // Check memory limit
    if (m_MaxKeyframes > 0 && static_cast<int>(m_Keyframes.size()) > m_MaxKeyframes) {
        CullOldestKeyframe();
    }
}

void SoftBodyDeformationRecorder::StartPlayback()
{
    std::lock_guard<std::mutex> lock(m_Mutex);
    
    if (m_Keyframes.empty()) {
        return;
    }
    
    m_PlaybackState = PlaybackState::Playing;
    m_PlaybackTime = 0.0f;
    m_PlaybackReversed = false;
}

void SoftBodyDeformationRecorder::StopPlayback()
{
    std::lock_guard<std::mutex> lock(m_Mutex);
    m_PlaybackState = PlaybackState::Stopped;
    m_PlaybackTime = 0.0f;
}

void SoftBodyDeformationRecorder::PausePlayback()
{
    std::lock_guard<std::mutex> lock(m_Mutex);
    if (m_PlaybackState == PlaybackState::Playing) {
        m_PlaybackState = PlaybackState::Paused;
    }
}

void SoftBodyDeformationRecorder::ResumePlayback()
{
    std::lock_guard<std::mutex> lock(m_Mutex);
    if (m_PlaybackState == PlaybackState::Paused) {
        m_PlaybackState = PlaybackState::Playing;
    }
}

void SoftBodyDeformationRecorder::SeekPlayback(float time)
{
    std::lock_guard<std::mutex> lock(m_Mutex);
    
    float duration = GetDuration();
    m_PlaybackTime = std::max(0.0f, std::min(time, duration));
}

bool SoftBodyDeformationRecorder::UpdatePlayback(float deltaTime, Vec3* outPositions)
{
    std::lock_guard<std::mutex> lock(m_Mutex);
    
    if (m_PlaybackState != PlaybackState::Playing || m_Keyframes.empty()) {
        return false;
    }
    
    // Update playback time
    float timeStep = deltaTime * m_PlaybackSpeed;
    if (m_PlaybackReversed) {
        timeStep = -timeStep;
    }
    
    m_PlaybackTime += timeStep;
    
    // Handle loop modes
    float duration = GetDuration();
    if (m_PlaybackTime >= duration) {
        switch (m_LoopMode) {
            case LoopMode::None:
                m_PlaybackTime = duration;
                m_PlaybackState = PlaybackState::Stopped;
                break;
                
            case LoopMode::Loop:
                m_PlaybackTime = 0.0f;
                break;
                
            case LoopMode::PingPong:
                m_PlaybackTime = duration;
                m_PlaybackReversed = true;
                break;
        }
    } else if (m_PlaybackTime < 0.0f) {
        // Only happens in ping-pong mode
        m_PlaybackTime = 0.0f;
        m_PlaybackReversed = false;
    }
    
    // Interpolate positions
    InterpolatePositions(m_PlaybackTime, outPositions);
    
    return m_PlaybackState == PlaybackState::Playing;
}

void SoftBodyDeformationRecorder::SetSampleRate(float samplesPerSecond)
{
    std::lock_guard<std::mutex> lock(m_Mutex);
    m_SampleRate = std::max(1.0f, samplesPerSecond);
    m_SampleInterval = 1.0f / m_SampleRate;
}

void SoftBodyDeformationRecorder::SetCompressionMode(CompressionMode mode)
{
    std::lock_guard<std::mutex> lock(m_Mutex);
    m_CompressionMode = mode;
}

void SoftBodyDeformationRecorder::SetMaxKeyframes(int maxKeyframes)
{
    std::lock_guard<std::mutex> lock(m_Mutex);
    m_MaxKeyframes = maxKeyframes;
}

void SoftBodyDeformationRecorder::SetPlaybackSpeed(float speed)
{
    std::lock_guard<std::mutex> lock(m_Mutex);
    m_PlaybackSpeed = std::max(0.1f, std::min(speed, 10.0f));
}

void SoftBodyDeformationRecorder::SetLoopMode(LoopMode mode)
{
    std::lock_guard<std::mutex> lock(m_Mutex);
    m_LoopMode = mode;
}

void SoftBodyDeformationRecorder::SetInterpolationMode(InterpolationMode mode)
{
    std::lock_guard<std::mutex> lock(m_Mutex);
    m_InterpolationMode = mode;
}

void SoftBodyDeformationRecorder::SetPartialRecording(bool enabled)
{
    std::lock_guard<std::mutex> lock(m_Mutex);
    m_PartialRecording = enabled;
    if (!enabled) {
        m_LastFramePositions.clear();
    }
}

void SoftBodyDeformationRecorder::SetChangeThreshold(float threshold)
{
    std::lock_guard<std::mutex> lock(m_Mutex);
    m_ChangeThreshold = std::max(0.0f, threshold);
}

float SoftBodyDeformationRecorder::GetDuration() const
{
    if (m_Keyframes.empty()) {
        return 0.0f;
    }
    return m_Keyframes.back().timestamp;
}

void SoftBodyDeformationRecorder::Clear()
{
    std::lock_guard<std::mutex> lock(m_Mutex);
    m_Keyframes.clear();
    m_RecordingState = RecordingState::Stopped;
    m_PlaybackState = PlaybackState::Stopped;
}

void SoftBodyDeformationRecorder::CullOldestKeyframe()
{
    // Remove oldest keyframe (first in vector)
    if (!m_Keyframes.empty()) {
        m_Keyframes.erase(m_Keyframes.begin());
    }
}

void SoftBodyDeformationRecorder::InterpolatePositions(float time, Vec3* outPositions) const
{
    if (m_InterpolationMode == InterpolationMode::Cubic) {
        InterpolateCubic(time, outPositions);
    } else {
        InterpolateLinear(time, outPositions);
    }
}

void SoftBodyDeformationRecorder::InterpolateLinear(float time, Vec3* outPositions) const
{
    if (m_Keyframes.empty()) {
        return;
    }
    
    // Find surrounding keyframes
    int index = FindKeyframeIndex(time);
    
    if (index < 0) {
        // Before first keyframe, use first keyframe
        ReconstructFromDelta(0, outPositions);
        return;
    }
    
    if (index >= static_cast<int>(m_Keyframes.size()) - 1) {
        // After last keyframe, use last keyframe
        ReconstructFromDelta(static_cast<int>(m_Keyframes.size()) - 1, outPositions);
        return;
    }
    
    // Get positions for both keyframes (handling compression)
    std::vector<Vec3> positions1(m_VertexCount);
    std::vector<Vec3> positions2(m_VertexCount);
    
    ReconstructFromDelta(index, positions1.data());
    ReconstructFromDelta(index + 1, positions2.data());
    
    // Interpolate between keyframes
    const DeformationKeyframe& k1 = m_Keyframes[index];
    const DeformationKeyframe& k2 = m_Keyframes[index + 1];
    
    float t = (time - k1.timestamp) / (k2.timestamp - k1.timestamp);
    t = std::max(0.0f, std::min(t, 1.0f));
    
    for (int i = 0; i < m_VertexCount; ++i) {
        outPositions[i].x = positions1[i].x + (positions2[i].x - positions1[i].x) * t;
        outPositions[i].y = positions1[i].y + (positions2[i].y - positions1[i].y) * t;
        outPositions[i].z = positions1[i].z + (positions2[i].z - positions1[i].z) * t;
    }
}

void SoftBodyDeformationRecorder::InterpolateCubic(float time, Vec3* outPositions) const
{
    if (m_Keyframes.size() < 2) {
        // Not enough keyframes for cubic, use first keyframe
        if (!m_Keyframes.empty()) {
            ReconstructFromDelta(0, outPositions);
        }
        return;
    }
    
    int index = FindKeyframeIndex(time);
    
    if (index < 0) {
        // Before first keyframe
        ReconstructFromDelta(0, outPositions);
        return;
    }
    
    if (index >= static_cast<int>(m_Keyframes.size()) - 1) {
        // After last keyframe
        ReconstructFromDelta(static_cast<int>(m_Keyframes.size()) - 1, outPositions);
        return;
    }
    
    // Get 4 keyframe indices for Catmull-Rom spline
    // P0 (before), P1 (start), P2 (end), P3 (after)
    int i0 = std::max(0, index - 1);
    int i1 = index;
    int i2 = std::min(index + 1, static_cast<int>(m_Keyframes.size()) - 1);
    int i3 = std::min(index + 2, static_cast<int>(m_Keyframes.size()) - 1);
    
    // Reconstruct positions for all 4 keyframes
    std::vector<Vec3> pos0(m_VertexCount);
    std::vector<Vec3> pos1(m_VertexCount);
    std::vector<Vec3> pos2(m_VertexCount);
    std::vector<Vec3> pos3(m_VertexCount);
    
    ReconstructFromDelta(i0, pos0.data());
    ReconstructFromDelta(i1, pos1.data());
    ReconstructFromDelta(i2, pos2.data());
    ReconstructFromDelta(i3, pos3.data());
    
    // Calculate t between i1 and i2
    const DeformationKeyframe& k1 = m_Keyframes[i1];
    const DeformationKeyframe& k2 = m_Keyframes[i2];
    
    float t = (time - k1.timestamp) / (k2.timestamp - k1.timestamp);
    t = std::max(0.0f, std::min(t, 1.0f));
    
    // Apply Catmull-Rom spline to each vertex
    for (int v = 0; v < m_VertexCount; ++v) {
        outPositions[v] = CatmullRom(pos0[v], pos1[v], pos2[v], pos3[v], t);
    }
}

Vec3 SoftBodyDeformationRecorder::CatmullRom(
    const Vec3& p0, const Vec3& p1, const Vec3& p2, const Vec3& p3, float t) const
{
    float t2 = t * t;
    float t3 = t2 * t;
    
    Vec3 result;
    
    // Catmull-Rom spline formula
    result.x = 0.5f * (
        (2.0f * p1.x) +
        (-p0.x + p2.x) * t +
        (2.0f * p0.x - 5.0f * p1.x + 4.0f * p2.x - p3.x) * t2 +
        (-p0.x + 3.0f * p1.x - 3.0f * p2.x + p3.x) * t3
    );
    
    result.y = 0.5f * (
        (2.0f * p1.y) +
        (-p0.y + p2.y) * t +
        (2.0f * p0.y - 5.0f * p1.y + 4.0f * p2.y - p3.y) * t2 +
        (-p0.y + 3.0f * p1.y - 3.0f * p2.y + p3.y) * t3
    );
    
    result.z = 0.5f * (
        (2.0f * p1.z) +
        (-p0.z + p2.z) * t +
        (2.0f * p0.z - 5.0f * p1.z + 4.0f * p2.z - p3.z) * t2 +
        (-p0.z + 3.0f * p1.z - 3.0f * p2.z + p3.z) * t3
    );
    
    return result;
}


int SoftBodyDeformationRecorder::FindKeyframeIndex(float time) const
{
    // Binary search for keyframe index
    int left = 0;
    int right = static_cast<int>(m_Keyframes.size()) - 1;
    
    if (time < m_Keyframes[0].timestamp) {
        return -1;
    }
    
    while (left <= right) {
        int mid = left + (right - left) / 2;
        
        if (m_Keyframes[mid].timestamp <= time) {
            if (mid == static_cast<int>(m_Keyframes.size()) - 1 || m_Keyframes[mid + 1].timestamp > time) {
                return mid;
            }
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    
    return static_cast<int>(m_Keyframes.size()) - 1;
}

void SoftBodyDeformationRecorder::CompressKeyframe(DeformationKeyframe& keyframe)
{
    // Update bounds with current frame
    UpdateBounds(keyframe.positions.data(), static_cast<int>(keyframe.positions.size()));
    
    // Apply compression based on mode
    switch (m_CompressionMode) {
        case CompressionMode::DeltaEncoding:
            ApplyDeltaEncoding(keyframe);
            break;
            
        case CompressionMode::Quantized:
            ApplyQuantization(keyframe);
            break;
            
        case CompressionMode::None:
        default:
            break;
    }
}

void SoftBodyDeformationRecorder::DecompressKeyframe(DeformationKeyframe& keyframe) const
{
    // Decompression is handled during playback interpolation
}

void SoftBodyDeformationRecorder::SetReferenceFrameInterval(int interval)
{
    std::lock_guard<std::mutex> lock(m_Mutex);
    m_ReferenceFrameInterval = std::max(0, interval);
}

SoftBodyDeformationRecorder::CompressionStats SoftBodyDeformationRecorder::GetCompressionStats() const
{
    std::lock_guard<std::mutex> lock(m_Mutex);
    
    CompressionStats stats;
    stats.uncompressedSize = 0;
    stats.compressedSize = 0;
    stats.averageError = 0.0f;
    stats.referenceFrameCount = 0;
    stats.deltaFrameCount = 0;
    
    for (const auto& keyframe : m_Keyframes) {
        // Calculate uncompressed size
        stats.uncompressedSize += sizeof(float); // timestamp
        stats.uncompressedSize += m_VertexCount * sizeof(Vec3); // positions
        if (keyframe.hasVelocities) {
            stats.uncompressedSize += m_VertexCount * sizeof(Vec3); // velocities
        }
        
        // Calculate compressed size
        stats.compressedSize += sizeof(float); // timestamp
        
        if (!keyframe.quantizedPositions16.empty()) {
            stats.compressedSize += keyframe.quantizedPositions16.size() * sizeof(QuantizedVec3_16);
        } else if (!keyframe.quantizedPositions8.empty()) {
            stats.compressedSize += keyframe.quantizedPositions8.size() * sizeof(QuantizedVec3_8);
        } else if (!keyframe.deltaPositions.empty()) {
            stats.compressedSize += keyframe.deltaPositions.size() * sizeof(Vec3);
            stats.deltaFrameCount++;
        } else {
            stats.compressedSize += keyframe.positions.size() * sizeof(Vec3);
        }
        
        if (keyframe.isReferenceFrame) {
            stats.referenceFrameCount++;
        }
        
        if (keyframe.hasVelocities && !keyframe.velocities.empty()) {
            stats.compressedSize += keyframe.velocities.size() * sizeof(Vec3);
        }
    }
    
    if (stats.uncompressedSize > 0) {
        stats.compressionRatio = static_cast<float>(stats.compressedSize) / static_cast<float>(stats.uncompressedSize);
    } else {
        stats.compressionRatio = 1.0f;
    }
    
    return stats;
}

// Compression helper implementations

void SoftBodyDeformationRecorder::CalculateBounds()
{
    if (m_Keyframes.empty()) {
        return;
    }
    
    Vec3 min(FLT_MAX, FLT_MAX, FLT_MAX);
    Vec3 max(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    
    for (const auto& keyframe : m_Keyframes) {
        for (const auto& pos : keyframe.positions) {
            min.x = std::min(min.x, pos.x);
            min.y = std::min(min.y, pos.y);
            min.z = std::min(min.z, pos.z);
            max.x = std::max(max.x, pos.x);
            max.y = std::max(max.y, pos.y);
            max.z = std::max(max.z, pos.z);
        }
    }
    
    m_CompressionMetadata.boundsMin = min;
    m_CompressionMetadata.boundsMax = max;
}

void SoftBodyDeformationRecorder::UpdateBounds(const Vec3* positions, int count)
{
    for (int i = 0; i < count; ++i) {
        const Vec3& pos = positions[i];
        m_CompressionMetadata.boundsMin.x = std::min(m_CompressionMetadata.boundsMin.x, pos.x);
        m_CompressionMetadata.boundsMin.y = std::min(m_CompressionMetadata.boundsMin.y, pos.y);
        m_CompressionMetadata.boundsMin.z = std::min(m_CompressionMetadata.boundsMin.z, pos.z);
        m_CompressionMetadata.boundsMax.x = std::max(m_CompressionMetadata.boundsMax.x, pos.x);
        m_CompressionMetadata.boundsMax.y = std::max(m_CompressionMetadata.boundsMax.y, pos.y);
        m_CompressionMetadata.boundsMax.z = std::max(m_CompressionMetadata.boundsMax.z, pos.z);
    }
}

QuantizedVec3_16 SoftBodyDeformationRecorder::QuantizePosition16(const Vec3& pos) const
{
    QuantizedVec3_16 qpos;
    
    Vec3 range = m_CompressionMetadata.boundsMax - m_CompressionMetadata.boundsMin;
    Vec3 normalized;
    
    // Avoid division by zero
    normalized.x = (range.x > 0.0001f) ? (pos.x - m_CompressionMetadata.boundsMin.x) / range.x : 0.0f;
    normalized.y = (range.y > 0.0001f) ? (pos.y - m_CompressionMetadata.boundsMin.y) / range.y : 0.0f;
    normalized.z = (range.z > 0.0001f) ? (pos.z - m_CompressionMetadata.boundsMin.z) / range.z : 0.0f;
    
    // Map [0, 1] to [-32768, 32767]
    qpos.x = static_cast<int16_t>(normalized.x * 65535.0f - 32768.0f);
    qpos.y = static_cast<int16_t>(normalized.y * 65535.0f - 32768.0f);
    qpos.z = static_cast<int16_t>(normalized.z * 65535.0f - 32768.0f);
    
    return qpos;
}

QuantizedVec3_8 SoftBodyDeformationRecorder::QuantizePosition8(const Vec3& pos) const
{
    QuantizedVec3_8 qpos;
    
    Vec3 range = m_CompressionMetadata.boundsMax - m_CompressionMetadata.boundsMin;
    Vec3 normalized;
    
    normalized.x = (range.x > 0.0001f) ? (pos.x - m_CompressionMetadata.boundsMin.x) / range.x : 0.0f;
    normalized.y = (range.y > 0.0001f) ? (pos.y - m_CompressionMetadata.boundsMin.y) / range.y : 0.0f;
    normalized.z = (range.z > 0.0001f) ? (pos.z - m_CompressionMetadata.boundsMin.z) / range.z : 0.0f;
    
    // Map [0, 1] to [-128, 127]
    qpos.x = static_cast<int8_t>(normalized.x * 255.0f - 128.0f);
    qpos.y = static_cast<int8_t>(normalized.y * 255.0f - 128.0f);
    qpos.z = static_cast<int8_t>(normalized.z * 255.0f - 128.0f);
    
    return qpos;
}

Vec3 SoftBodyDeformationRecorder::DequantizePosition16(const QuantizedVec3_16& qpos) const
{
    Vec3 pos;
    
    // Map [-32768, 32767] to [0, 1]
    pos.x = (qpos.x + 32768.0f) / 65535.0f;
    pos.y = (qpos.y + 32768.0f) / 65535.0f;
    pos.z = (qpos.z + 32768.0f) / 65535.0f;
    
    // Scale to original range
    Vec3 range = m_CompressionMetadata.boundsMax - m_CompressionMetadata.boundsMin;
    pos.x = pos.x * range.x + m_CompressionMetadata.boundsMin.x;
    pos.y = pos.y * range.y + m_CompressionMetadata.boundsMin.y;
    pos.z = pos.z * range.z + m_CompressionMetadata.boundsMin.z;
    
    return pos;
}

Vec3 SoftBodyDeformationRecorder::DequantizePosition8(const QuantizedVec3_8& qpos) const
{
    Vec3 pos;
    
    // Map [-128, 127] to [0, 1]
    pos.x = (qpos.x + 128.0f) / 255.0f;
    pos.y = (qpos.y + 128.0f) / 255.0f;
    pos.z = (qpos.z + 128.0f) / 255.0f;
    
    // Scale to original range
    Vec3 range = m_CompressionMetadata.boundsMax - m_CompressionMetadata.boundsMin;
    pos.x = pos.x * range.x + m_CompressionMetadata.boundsMin.x;
    pos.y = pos.y * range.y + m_CompressionMetadata.boundsMin.y;
    pos.z = pos.z * range.z + m_CompressionMetadata.boundsMin.z;
    
    return pos;
}

void SoftBodyDeformationRecorder::ApplyDeltaEncoding(DeformationKeyframe& keyframe)
{
    if (m_ReferenceFrameInterval == 0 || 
        m_FramesSinceReference >= m_ReferenceFrameInterval || 
        m_LastReferencePositions.empty()) {
        // This is a reference frame
        keyframe.isReferenceFrame = true;
        m_LastReferencePositions = keyframe.positions;
        m_FramesSinceReference = 0;
    } else {
        // Delta frame
        keyframe.isReferenceFrame = false;
        keyframe.deltaPositions.resize(keyframe.positions.size());
        
        for (size_t i = 0; i < keyframe.positions.size(); ++i) {
            keyframe.deltaPositions[i].x = keyframe.positions[i].x - m_LastReferencePositions[i].x;
            keyframe.deltaPositions[i].y = keyframe.positions[i].y - m_LastReferencePositions[i].y;
            keyframe.deltaPositions[i].z = keyframe.positions[i].z - m_LastReferencePositions[i].z;
        }
        
        // Clear full positions to save memory
        keyframe.positions.clear();
        m_FramesSinceReference++;
    }
}

void SoftBodyDeformationRecorder::ApplyQuantization(DeformationKeyframe& keyframe)
{
    keyframe.isReferenceFrame = true; // All quantized frames are reference frames
    keyframe.quantizedPositions16.resize(keyframe.positions.size());
    
    for (size_t i = 0; i < keyframe.positions.size(); ++i) {
        keyframe.quantizedPositions16[i] = QuantizePosition16(keyframe.positions[i]);
    }
    
    // Clear full positions to save memory
    keyframe.positions.clear();
}

void SoftBodyDeformationRecorder::ReconstructFromDelta(int keyframeIndex, Vec3* outPositions) const
{
    if (keyframeIndex < 0 || keyframeIndex >= static_cast<int>(m_Keyframes.size())) {
        return;
    }
    
    const DeformationKeyframe& keyframe = m_Keyframes[keyframeIndex];
    
    if (keyframe.isReferenceFrame) {
        // Reference frame - copy positions directly
        if (!keyframe.positions.empty()) {
            std::memcpy(outPositions, keyframe.positions.data(), keyframe.positions.size() * sizeof(Vec3));
        } else if (!keyframe.quantizedPositions16.empty()) {
            // Dequantize
            for (size_t i = 0; i < keyframe.quantizedPositions16.size(); ++i) {
                outPositions[i] = DequantizePosition16(keyframe.quantizedPositions16[i]);
            }
        }
    } else {
        // Delta frame - find reference and apply deltas
        int refIndex = keyframeIndex - 1;
        while (refIndex >= 0 && !m_Keyframes[refIndex].isReferenceFrame) {
            refIndex--;
        }
        
        if (refIndex >= 0) {
            // Get reference positions
            ReconstructFromDelta(refIndex, outPositions);
            
            // Apply deltas
            for (size_t i = 0; i < keyframe.deltaPositions.size(); ++i) {
                outPositions[i].x += keyframe.deltaPositions[i].x;
                outPositions[i].y += keyframe.deltaPositions[i].y;
                outPositions[i].z += keyframe.deltaPositions[i].z;
            }
        }
    }
}


nlohmann::json SoftBodyDeformationRecorder::ToJson() const
{
    std::lock_guard<std::mutex> lock(m_Mutex);
    
    nlohmann::json j;
    j["version"] = "1.0";
    j["vertexCount"] = m_VertexCount;
    j["sampleRate"] = m_SampleRate;
    j["compressionMode"] = static_cast<int>(m_CompressionMode);
    j["hasVelocities"] = m_RecordVelocities;
    
    nlohmann::json keyframesJson = nlohmann::json::array();
    for (const auto& keyframe : m_Keyframes) {
        nlohmann::json kf;
        kf["timestamp"] = keyframe.timestamp;
        
        nlohmann::json positions = nlohmann::json::array();
        for (const auto& pos : keyframe.positions) {
            positions.push_back({pos.x, pos.y, pos.z});
        }
        kf["positions"] = positions;
        
        if (keyframe.hasVelocities) {
            nlohmann::json velocities = nlohmann::json::array();
            for (const auto& vel : keyframe.velocities) {
                velocities.push_back({vel.x, vel.y, vel.z});
            }
            kf["velocities"] = velocities;
        }
        
        keyframesJson.push_back(kf);
    }
    j["keyframes"] = keyframesJson;
    
    return j;
}

bool SoftBodyDeformationRecorder::FromJson(const nlohmann::json& json)
{
    std::lock_guard<std::mutex> lock(m_Mutex);
    
    try {
        m_VertexCount = json["vertexCount"];
        m_SampleRate = json["sampleRate"];
        m_SampleInterval = 1.0f / m_SampleRate;
        m_CompressionMode = static_cast<CompressionMode>(json["compressionMode"].get<int>());
        m_RecordVelocities = json["hasVelocities"];
        
        m_Keyframes.clear();
        
        for (const auto& kfJson : json["keyframes"]) {
            DeformationKeyframe keyframe;
            keyframe.timestamp = kfJson["timestamp"];
            
            for (const auto& posJson : kfJson["positions"]) {
                Vec3 pos;
                pos.x = posJson[0];
                pos.y = posJson[1];
                pos.z = posJson[2];
                keyframe.positions.push_back(pos);
            }
            
            if (kfJson.contains("velocities")) {
                keyframe.hasVelocities = true;
                for (const auto& velJson : kfJson["velocities"]) {
                    Vec3 vel;
                    vel.x = velJson[0];
                    vel.y = velJson[1];
                    vel.z = velJson[2];
                    keyframe.velocities.push_back(vel);
                }
            } else {
                keyframe.hasVelocities = false;
            }
            
            m_Keyframes.push_back(std::move(keyframe));
        }
        
        return true;
    } catch (const std::exception& e) {
        return false;
    }
}

bool SoftBodyDeformationRecorder::SaveToFile(const std::string& filename, bool binary) const
{
    if (binary) {
        return SaveBinary(filename);
    }
    
    try {
        nlohmann::json j = ToJson();
        std::ofstream file(filename);
        if (!file.is_open()) {
            return false;
        }
        file << j.dump(2);
        file.close();
        return true;
    } catch (const std::exception& e) {
        return false;
    }
}

bool SoftBodyDeformationRecorder::LoadFromFile(const std::string& filename)
{
    // Try binary first
    if (LoadBinary(filename)) {
        return true;
    }
    
    // Try JSON
    try {
        std::ifstream file(filename);
        if (!file.is_open()) {
            return false;
        }
        
        nlohmann::json j;
        file >> j;
        file.close();
        
        return FromJson(j);
    } catch (const std::exception& e) {
        return false;
    }
}

bool SoftBodyDeformationRecorder::SaveBinary(const std::string& filename) const
{
    std::lock_guard<std::mutex> lock(m_Mutex);
    
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    
    // Write header
    char magic[4] = {'S', 'B', 'D', 'R'};  // SoftBody Deformation Recording
    file.write(magic, 4);
    
    int version = 1;
    file.write(reinterpret_cast<const char*>(&version), sizeof(int));
    file.write(reinterpret_cast<const char*>(&m_VertexCount), sizeof(int));
    file.write(reinterpret_cast<const char*>(&m_SampleRate), sizeof(float));
    
    int compressionMode = static_cast<int>(m_CompressionMode);
    file.write(reinterpret_cast<const char*>(&compressionMode), sizeof(int));
    
    int hasVelocities = m_RecordVelocities ? 1 : 0;
    file.write(reinterpret_cast<const char*>(&hasVelocities), sizeof(int));
    
    int keyframeCount = static_cast<int>(m_Keyframes.size());
    file.write(reinterpret_cast<const char*>(&keyframeCount), sizeof(int));
    
    // Write keyframes
    for (const auto& keyframe : m_Keyframes) {
        file.write(reinterpret_cast<const char*>(&keyframe.timestamp), sizeof(float));
        file.write(reinterpret_cast<const char*>(keyframe.positions.data()), 
                   m_VertexCount * sizeof(Vec3));
        
        if (keyframe.hasVelocities) {
            file.write(reinterpret_cast<const char*>(keyframe.velocities.data()), 
                       m_VertexCount * sizeof(Vec3));
        }
    }
    
    file.close();
    return true;
}

bool SoftBodyDeformationRecorder::LoadBinary(const std::string& filename)
{
    std::lock_guard<std::mutex> lock(m_Mutex);
    
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    
    // Read header
    char magic[4];
    file.read(magic, 4);
    if (magic[0] != 'S' || magic[1] != 'B' || magic[2] != 'D' || magic[3] != 'R') {
        file.close();
        return false;  // Not a binary recording file
    }
    
    int version;
    file.read(reinterpret_cast<char*>(&version), sizeof(int));
    if (version != 1) {
        file.close();
        return false;  // Unsupported version
    }
    
    file.read(reinterpret_cast<char*>(&m_VertexCount), sizeof(int));
    file.read(reinterpret_cast<char*>(&m_SampleRate), sizeof(float));
    m_SampleInterval = 1.0f / m_SampleRate;
    
    int compressionMode;
    file.read(reinterpret_cast<char*>(&compressionMode), sizeof(int));
    m_CompressionMode = static_cast<CompressionMode>(compressionMode);
    
    int hasVelocities;
    file.read(reinterpret_cast<char*>(&hasVelocities), sizeof(int));
    m_RecordVelocities = hasVelocities != 0;
    
    int keyframeCount;
    file.read(reinterpret_cast<char*>(&keyframeCount), sizeof(int));
    
    m_Keyframes.clear();
    m_Keyframes.reserve(keyframeCount);
    
    // Read keyframes
    for (int i = 0; i < keyframeCount; ++i) {
        DeformationKeyframe keyframe;
        
        file.read(reinterpret_cast<char*>(&keyframe.timestamp), sizeof(float));
        
        keyframe.positions.resize(m_VertexCount);
        file.read(reinterpret_cast<char*>(keyframe.positions.data()), 
                  m_VertexCount * sizeof(Vec3));
        
        if (m_RecordVelocities) {
            keyframe.hasVelocities = true;
            keyframe.velocities.resize(m_VertexCount);
            file.read(reinterpret_cast<char*>(keyframe.velocities.data()), 
                      m_VertexCount * sizeof(Vec3));
        } else {
            keyframe.hasVelocities = false;
        }
        
        m_Keyframes.push_back(std::move(keyframe));
    }
    
    file.close();
    return true;
}

// Partial recording helper implementations

void SoftBodyDeformationRecorder::DetectChangedVertices(const Vec3* positions, std::vector<int>& changedIndices)
{
    changedIndices.clear();
    
    if (m_LastFramePositions.empty()) {
        // First frame - all vertices considered changed
        changedIndices.reserve(m_VertexCount);
        for (int i = 0; i < m_VertexCount; ++i) {
            changedIndices.push_back(i);
        }
        
        // Store current positions for next frame
        m_LastFramePositions.assign(positions, positions + m_VertexCount);
    } else {
        // Compare with last frame
        float thresholdSq = m_ChangeThreshold * m_ChangeThreshold;
        
        for (int i = 0; i < m_VertexCount; ++i) {
            const Vec3& current = positions[i];
            const Vec3& previous = m_LastFramePositions[i];
            
            float dx = current.x - previous.x;
            float dy = current.y - previous.y;
            float dz = current.z - previous.z;
            float distSq = dx*dx + dy*dy + dz*dz;
            
            if (distSq > thresholdSq) {
                changedIndices.push_back(i);
            }
        }
        
        // Update last frame positions
        std::memcpy(m_LastFramePositions.data(), positions, m_VertexCount * sizeof(Vec3));
    }
}

void SoftBodyDeformationRecorder::ReconstructFromSparse(int keyframeIndex, Vec3* outPositions) const
{
    if (keyframeIndex < 0 || keyframeIndex >= static_cast<int>(m_Keyframes.size())) {
        return;
    }
    
    const DeformationKeyframe& keyframe = m_Keyframes[keyframeIndex];
    
    if (!keyframe.isSparse) {
        // Dense storage - use normal reconstruction
        ReconstructFromDelta(keyframeIndex, outPositions);
        return;
    }
    
    // Find last dense/reference frame
    int refIndex = keyframeIndex - 1;
    while (refIndex >= 0 && m_Keyframes[refIndex].isSparse) {
        refIndex--;
    }
    
    if (refIndex >= 0) {
        // Start with reference frame
        ReconstructFromDelta(refIndex, outPositions);
    } else {
        // No reference - initialize to zero
        std::memset(outPositions, 0, m_VertexCount * sizeof(Vec3));
    }
    
    // Apply sparse changes from all frames since reference
    for (int i = refIndex + 1; i <= keyframeIndex; ++i) {
        const DeformationKeyframe& frame = m_Keyframes[i];
        if (frame.isSparse) {
            // Apply sparse positions
            for (const SparseVertex& sv : frame.sparsePositions) {
                if (sv.index >= 0 && sv.index < m_VertexCount) {
                    outPositions[sv.index] = sv.position;
                }
            }
        }
    }
}

SoftBodyDeformationRecorder::PartialRecordingStats SoftBodyDeformationRecorder::GetPartialRecordingStats() const
{
    std::lock_guard<std::mutex> lock(m_Mutex);
    
    PartialRecordingStats stats;
    stats.totalVerticesStored = 0;
    stats.totalVerticesPossible = m_Keyframes.size() * m_VertexCount;
    stats.sparseFrameCount = 0;
    stats.denseFrameCount = 0;
    stats.averageChangedRatio = 0.0f;
    
    if (m_Keyframes.empty()) {
        stats.compressionRatio = 1.0f;
        return stats;
    }
    
    for (const auto& keyframe : m_Keyframes) {
        if (keyframe.isSparse) {
            stats.totalVerticesStored += keyframe.sparsePositions.size();
            stats.sparseFrameCount++;
        } else {
            stats.totalVerticesStored += m_VertexCount;
            stats.denseFrameCount++;
        }
    }
    
    if (stats.totalVerticesPossible > 0) {
        stats.compressionRatio = static_cast<float>(stats.totalVerticesStored) / 
                                  static_cast<float>(stats.totalVerticesPossible);
        stats.averageChangedRatio = stats.compressionRatio;
    } else {
        stats.compressionRatio = 1.0f;
    }
    
    return stats;
}

// Streaming implementation

void SoftBodyDeformationRecorder::SetStreamingMode(bool enabled, int memoryBudgetMB)
{
    std::lock_guard<std::mutex> lock(m_Mutex);
    m_StreamingMode = enabled;
    m_MemoryBudgetBytes = static_cast<size_t>(memoryBudgetMB) * 1024 * 1024;
}

bool SoftBodyDeformationRecorder::SaveStreaming(const std::string& filename, 
                                                std::function<void(int, int)> progressCallback)
{
    std::lock_guard<std::mutex> lock(m_Mutex);
    
    if (m_Keyframes.empty()) {
        return false;
    }
    
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    
    // Write header
    char magic[4] = {'S', 'B', 'D', 'S'};  // SoftBody Deformation Streaming
    file.write(magic, 4);
    
    int version = 1;
    file.write(reinterpret_cast<const char*>(&version), sizeof(int));
    file.write(reinterpret_cast<const char*>(&m_VertexCount), sizeof(int));
    file.write(reinterpret_cast<const char*>(&m_ChunkSize), sizeof(int));
    
    // Calculate chunk count
    int chunkCount = (static_cast<int>(m_Keyframes.size()) + m_ChunkSize - 1) / m_ChunkSize;
    file.write(reinterpret_cast<const char*>(&chunkCount), sizeof(int));
    
    // Reserve space for chunk index
    size_t indexOffset = file.tellp();
    std::vector<size_t> chunkOffsets(chunkCount);
    std::vector<size_t> chunkSizes(chunkCount);
    std::vector<float> chunkStartTimes(chunkCount);
    std::vector<float> chunkEndTimes(chunkCount);
    
    // Skip index space
    file.seekp(indexOffset + chunkCount * (sizeof(size_t) * 2 + sizeof(float) * 2));
    
    // Write chunks
    for (int i = 0; i < chunkCount; ++i) {
        chunkOffsets[i] = file.tellp();
        
        int startIdx = i * m_ChunkSize;
        int endIdx = std::min(startIdx + m_ChunkSize, static_cast<int>(m_Keyframes.size()));
        int count = endIdx - startIdx;
        
        chunkStartTimes[i] = m_Keyframes[startIdx].timestamp;
        chunkEndTimes[i] = m_Keyframes[endIdx - 1].timestamp;
        
        // Write chunk header
        file.write(reinterpret_cast<const char*>(&count), sizeof(int));
        
        // Write keyframes in chunk
        for (int j = startIdx; j < endIdx; ++j) {
            const DeformationKeyframe& kf = m_Keyframes[j];
            
            // Write keyframe data
            file.write(reinterpret_cast<const char*>(&kf.timestamp), sizeof(float));
            file.write(reinterpret_cast<const char*>(&kf.isReferenceFrame), sizeof(bool));
            file.write(reinterpret_cast<const char*>(&kf.isSparse), sizeof(bool));
            file.write(reinterpret_cast<const char*>(&kf.hasVelocities), sizeof(bool));
            
            // Write positions
            int posCount = static_cast<int>(kf.positions.size());
            file.write(reinterpret_cast<const char*>(&posCount), sizeof(int));
            if (posCount > 0) {
                file.write(reinterpret_cast<const char*>(kf.positions.data()), 
                          posCount * sizeof(Vec3));
            }
            
            // Write sparse positions
            int sparseCount = static_cast<int>(kf.sparsePositions.size());
            file.write(reinterpret_cast<const char*>(&sparseCount), sizeof(int));
            for (const auto& sv : kf.sparsePositions) {
                file.write(reinterpret_cast<const char*>(&sv.index), sizeof(int));
                file.write(reinterpret_cast<const char*>(&sv.position), sizeof(Vec3));
            }
        }
        
        chunkSizes[i] = static_cast<size_t>(file.tellp()) - chunkOffsets[i];
        
        if (progressCallback) {
            progressCallback(i + 1, chunkCount);
        }
    }
    
    // Write chunk index
    file.seekp(indexOffset);
    for (int i = 0; i < chunkCount; ++i) {
        file.write(reinterpret_cast<const char*>(&chunkOffsets[i]), sizeof(size_t));
        file.write(reinterpret_cast<const char*>(&chunkSizes[i]), sizeof(size_t));
        file.write(reinterpret_cast<const char*>(&chunkStartTimes[i]), sizeof(float));
        file.write(reinterpret_cast<const char*>(&chunkEndTimes[i]), sizeof(float));
    }
    
    file.close();
    return true;
}

bool SoftBodyDeformationRecorder::LoadStreaming(const std::string& filename)
{
    std::lock_guard<std::mutex> lock(m_Mutex);
    
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    
    // Read header
    char magic[4];
    file.read(magic, 4);
    if (magic[0] != 'S' || magic[1] != 'B' || magic[2] != 'D' || magic[3] != 'S') {
        file.close();
        return false;
    }
    
    int version;
    file.read(reinterpret_cast<char*>(&version), sizeof(int));
    if (version != 1) {
        file.close();
        return false;
    }
    
    file.read(reinterpret_cast<char*>(&m_VertexCount), sizeof(int));
    file.read(reinterpret_cast<char*>(&m_ChunkSize), sizeof(int));
    
    int chunkCount;
    file.read(reinterpret_cast<char*>(&chunkCount), sizeof(int));
    
    // Read chunk index
    m_Chunks.resize(chunkCount);
    for (int i = 0; i < chunkCount; ++i) {
        file.read(reinterpret_cast<char*>(&m_Chunks[i].fileOffset), sizeof(size_t));
        file.read(reinterpret_cast<char*>(&m_Chunks[i].compressedSize), sizeof(size_t));
        file.read(reinterpret_cast<char*>(&m_Chunks[i].startTime), sizeof(float));
        file.read(reinterpret_cast<char*>(&m_Chunks[i].endTime), sizeof(float));
        
        m_Chunks[i].startIndex = i * m_ChunkSize;
        m_Chunks[i].isLoaded = false;
        m_Chunks[i].lastAccessFrame = 0;
    }
    
    file.close();
    
    m_StreamingFilename = filename;
    m_StreamingMode = true;
    m_CacheHits = 0;
    m_CacheMisses = 0;
    
    // Load first chunk for immediate playback
    LoadChunk(0);
    
    return true;
}

void SoftBodyDeformationRecorder::LoadChunk(int chunkIndex) const
{
    if (chunkIndex < 0 || chunkIndex >= static_cast<int>(m_Chunks.size())) {
        return;
    }
    
    KeyframeChunk& chunk = m_Chunks[chunkIndex];
    if (chunk.isLoaded) {
        m_CacheHits++;
        chunk.lastAccessFrame = m_CurrentPlaybackFrame;
        return;
    }
    
    m_CacheMisses++;
    
    // Check memory budget and evict if needed
    while (GetCacheMemoryUsage() + EstimateChunkSize(chunk) > m_MemoryBudgetBytes) {
        EvictLRUChunk();
    }
    
    // Load from file
    std::ifstream file(m_StreamingFilename, std::ios::binary);
    if (!file.is_open()) {
        return;
    }
    
    file.seekg(chunk.fileOffset);
    
    // Read chunk header
    int count;
    file.read(reinterpret_cast<char*>(&count), sizeof(int));
    chunk.count = count;
    
    // Read keyframes
    chunk.keyframes.resize(count);
    for (int i = 0; i < count; ++i) {
        DeformationKeyframe& kf = chunk.keyframes[i];
        
        file.read(reinterpret_cast<char*>(&kf.timestamp), sizeof(float));
        file.read(reinterpret_cast<char*>(&kf.isReferenceFrame), sizeof(bool));
        file.read(reinterpret_cast<char*>(&kf.isSparse), sizeof(bool));
        file.read(reinterpret_cast<char*>(&kf.hasVelocities), sizeof(bool));
        
        // Read positions
        int posCount;
        file.read(reinterpret_cast<char*>(&posCount), sizeof(int));
        if (posCount > 0) {
            kf.positions.resize(posCount);
            file.read(reinterpret_cast<char*>(kf.positions.data()), 
                     posCount * sizeof(Vec3));
        }
        
        // Read sparse positions
        int sparseCount;
        file.read(reinterpret_cast<char*>(&sparseCount), sizeof(int));
        kf.sparsePositions.resize(sparseCount);
        for (int j = 0; j < sparseCount; ++j) {
            file.read(reinterpret_cast<char*>(&kf.sparsePositions[j].index), sizeof(int));
            file.read(reinterpret_cast<char*>(&kf.sparsePositions[j].position), sizeof(Vec3));
        }
    }
    
    file.close();
    
    chunk.isLoaded = true;
    chunk.lastAccessFrame = m_CurrentPlaybackFrame;
}

void SoftBodyDeformationRecorder::UnloadChunk(int chunkIndex) const
{
    if (chunkIndex < 0 || chunkIndex >= static_cast<int>(m_Chunks.size())) {
        return;
    }
    
    KeyframeChunk& chunk = m_Chunks[chunkIndex];
    chunk.keyframes.clear();
    chunk.isLoaded = false;
}

void SoftBodyDeformationRecorder::EvictLRUChunk() const
{
    int lruIndex = -1;
    int oldestFrame = m_CurrentPlaybackFrame;
    
    for (size_t i = 0; i < m_Chunks.size(); ++i) {
        if (m_Chunks[i].isLoaded) {
            if (m_Chunks[i].lastAccessFrame < oldestFrame) {
                oldestFrame = m_Chunks[i].lastAccessFrame;
                lruIndex = static_cast<int>(i);
            }
        }
    }
    
    if (lruIndex >= 0) {
        UnloadChunk(lruIndex);
    }
}

int SoftBodyDeformationRecorder::FindChunkForTime(float time) const
{
    for (size_t i = 0; i < m_Chunks.size(); ++i) {
        if (time >= m_Chunks[i].startTime && time <= m_Chunks[i].endTime) {
            return static_cast<int>(i);
        }
    }
    return m_Chunks.empty() ? -1 : 0;
}

int SoftBodyDeformationRecorder::FindChunkForIndex(int keyframeIndex) const
{
    return keyframeIndex / m_ChunkSize;
}

size_t SoftBodyDeformationRecorder::GetCacheMemoryUsage() const
{
    size_t total = 0;
    for (const auto& chunk : m_Chunks) {
        if (chunk.isLoaded) {
            total += EstimateChunkSize(chunk);
        }
    }
    return total;
}

size_t SoftBodyDeformationRecorder::EstimateChunkSize(const KeyframeChunk& chunk) const
{
    if (chunk.isLoaded) {
        size_t size = 0;
        for (const auto& kf : chunk.keyframes) {
            size += kf.positions.size() * sizeof(Vec3);
            size += kf.sparsePositions.size() * (sizeof(int) + sizeof(Vec3));
            size += kf.velocities.size() * sizeof(Vec3);
        }
        return size;
    }
    return chunk.compressedSize;  // Estimate
}

SoftBodyDeformationRecorder::StreamingStats SoftBodyDeformationRecorder::GetStreamingStats() const
{
    std::lock_guard<std::mutex> lock(m_Mutex);
    
    StreamingStats stats;
    stats.totalChunks = m_Chunks.size();
    stats.loadedChunks = 0;
    stats.cacheHits = m_CacheHits;
    stats.cacheMisses = m_CacheMisses;
    stats.memoryUsage = GetCacheMemoryUsage();
    stats.memoryBudget = m_MemoryBudgetBytes;
    
    for (const auto& chunk : m_Chunks) {
        if (chunk.isLoaded) {
            stats.loadedChunks++;
        }
    }
    
    size_t totalAccesses = m_CacheHits + m_CacheMisses;
    stats.cacheHitRatio = totalAccesses > 0 ? 
        static_cast<float>(m_CacheHits) / static_cast<float>(totalAccesses) : 0.0f;
    
    return stats;
}
