#include "NetworkTransform.hpp"
#include <cstring>
#include <cmath>

NetworkTransform::NetworkTransform()
    : m_Transform(nullptr)
    , m_LastSentPosition(0, 0, 0)
    , m_LastSentRotation(0, 0, 0)
    , m_LastSentScale(1, 1, 1)
    , m_IsDirty(true)
    , m_InterpolationTime(0.0f) {}

NetworkTransform::NetworkTransform(Transform* transform)
    : m_Transform(transform)
    , m_IsDirty(true)
    , m_InterpolationTime(0.0f) {
    if (transform) {
        m_LastSentPosition = transform->GetPosition();
        m_LastSentRotation = transform->GetRotation();
        m_LastSentScale = transform->GetScale();
    }
}

void NetworkTransform::Update(float deltaTime) {
    if (!m_Transform || m_NetId.isLocal) return;
    
    // Advance interpolation time
    m_InterpolationTime += deltaTime;
    
    // Need at least 2 snapshots to interpolate
    if (m_SnapshotBuffer.size() < 2) return;
    
    // Get the two snapshots to interpolate between
    Snapshot from = m_SnapshotBuffer.front();
    
    // Find the appropriate snapshot pair based on render time
    float renderTime = m_InterpolationTime - m_Settings.interpolationDelay;
    
    while (m_SnapshotBuffer.size() > 2) {
        Snapshot next = m_SnapshotBuffer.front();
        m_SnapshotBuffer.pop();
        Snapshot afterNext = m_SnapshotBuffer.front();
        
        if (afterNext.timestamp > renderTime) {
            from = next;
            break;
        }
    }
    
    if (m_SnapshotBuffer.empty()) return;
    Snapshot to = m_SnapshotBuffer.front();
    
    // Calculate interpolation factor
    float duration = to.timestamp - from.timestamp;
    float t = (duration > 0.0001f) ? (renderTime - from.timestamp) / duration : 1.0f;
    t = std::max(0.0f, std::min(1.0f, t));
    
    // Check for teleport
    Vec3 diff = to.position - from.position;
    float distance = std::sqrt(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z);
    
    if (distance > m_Settings.teleportThreshold) {
        // Teleport instead of interpolate
        m_Transform->SetPosition(to.position);
        m_Transform->SetRotation(to.rotation);
        m_Transform->SetScale(to.scale);
    } else {
        // Smooth interpolation
        if (m_Settings.syncPosition) {
            m_Transform->SetPosition(InterpolateVec3(from.position, to.position, t));
        }
        if (m_Settings.syncRotation) {
            m_Transform->SetRotation(InterpolateVec3(from.rotation, to.rotation, t));
        }
        if (m_Settings.syncScale) {
            m_Transform->SetScale(InterpolateVec3(from.scale, to.scale, t));
        }
    }
}

bool NetworkTransform::IsDirty() const {
    if (!m_Transform) return false;
    
    if (m_Settings.syncPosition && 
        HasChanged(m_Transform->GetPosition(), m_LastSentPosition, m_Settings.snapshotThreshold)) {
        return true;
    }
    if (m_Settings.syncRotation && 
        HasChanged(m_Transform->GetRotation(), m_LastSentRotation, m_Settings.snapshotThreshold)) {
        return true;
    }
    if (m_Settings.syncScale && 
        HasChanged(m_Transform->GetScale(), m_LastSentScale, m_Settings.snapshotThreshold)) {
        return true;
    }
    
    return false;
}

void NetworkTransform::ClearDirty() {
    if (m_Transform) {
        m_LastSentPosition = m_Transform->GetPosition();
        m_LastSentRotation = m_Transform->GetRotation();
        m_LastSentScale = m_Transform->GetScale();
    }
    m_IsDirty = false;
}

void NetworkTransform::Serialize(std::vector<uint8_t>& outData, bool fullState) const {
    if (!m_Transform) return;
    
    // Header: flags byte indicating which components are included
    uint8_t flags = 0;
    if (m_Settings.syncPosition) flags |= 0x01;
    if (m_Settings.syncRotation) flags |= 0x02;
    if (m_Settings.syncScale) flags |= 0x04;
    if (fullState) flags |= 0x80;  // Full state flag
    
    outData.push_back(flags);
    
    // Network ID (4 bytes)
    uint32_t netId = m_NetId.networkId;
    outData.push_back(static_cast<uint8_t>(netId & 0xFF));
    outData.push_back(static_cast<uint8_t>((netId >> 8) & 0xFF));
    outData.push_back(static_cast<uint8_t>((netId >> 16) & 0xFF));
    outData.push_back(static_cast<uint8_t>((netId >> 24) & 0xFF));
    
    auto writeFloat = [&outData](float value) {
        uint32_t bits;
        std::memcpy(&bits, &value, sizeof(float));
        outData.push_back(static_cast<uint8_t>(bits & 0xFF));
        outData.push_back(static_cast<uint8_t>((bits >> 8) & 0xFF));
        outData.push_back(static_cast<uint8_t>((bits >> 16) & 0xFF));
        outData.push_back(static_cast<uint8_t>((bits >> 24) & 0xFF));
    };
    
    auto writeVec3 = [&writeFloat](const Vec3& v) {
        writeFloat(v.x);
        writeFloat(v.y);
        writeFloat(v.z);
    };
    
    Vec3 pos = m_Transform->GetPosition();
    Vec3 rot = m_Transform->GetRotation();
    Vec3 scl = m_Transform->GetScale();
    
    if (m_Settings.syncPosition) writeVec3(pos);
    if (m_Settings.syncRotation) writeVec3(rot);
    if (m_Settings.syncScale) writeVec3(scl);
}

size_t NetworkTransform::Deserialize(const uint8_t* data, size_t length) {
    if (length < 5) return 0;  // Minimum: flags + netId
    
    size_t offset = 0;
    
    uint8_t flags = data[offset++];
    bool hasPosition = (flags & 0x01) != 0;
    bool hasRotation = (flags & 0x02) != 0;
    bool hasScale = (flags & 0x04) != 0;
    
    // Read network ID
    uint32_t netId = 0;
    netId |= static_cast<uint32_t>(data[offset++]);
    netId |= static_cast<uint32_t>(data[offset++]) << 8;
    netId |= static_cast<uint32_t>(data[offset++]) << 16;
    netId |= static_cast<uint32_t>(data[offset++]) << 24;
    
    auto readFloat = [&data, &offset]() -> float {
        uint32_t bits = 0;
        bits |= static_cast<uint32_t>(data[offset++]);
        bits |= static_cast<uint32_t>(data[offset++]) << 8;
        bits |= static_cast<uint32_t>(data[offset++]) << 16;
        bits |= static_cast<uint32_t>(data[offset++]) << 24;
        float value;
        std::memcpy(&value, &bits, sizeof(float));
        return value;
    };
    
    auto readVec3 = [&readFloat]() -> Vec3 {
        float x = readFloat();
        float y = readFloat();
        float z = readFloat();
        return Vec3(x, y, z);
    };
    
    Vec3 position(0, 0, 0), rotation(0, 0, 0), scale(1, 1, 1);
    
    if (hasPosition) position = readVec3();
    if (hasRotation) rotation = readVec3();
    if (hasScale) scale = readVec3();
    
    // For remote entities, push to interpolation buffer
    if (!m_NetId.isLocal) {
        PushSnapshot(position, rotation, scale, m_InterpolationTime);
    }
    
    return offset;
}

void NetworkTransform::PushSnapshot(const Vec3& position, const Vec3& rotation, 
                                     const Vec3& scale, float timestamp) {
    Snapshot snapshot;
    snapshot.position = position;
    snapshot.rotation = rotation;
    snapshot.scale = scale;
    snapshot.timestamp = timestamp;
    
    m_SnapshotBuffer.push(snapshot);
    
    // Limit buffer size
    while (m_SnapshotBuffer.size() > 30) {
        m_SnapshotBuffer.pop();
    }
}

void NetworkTransform::SavePredictionState() {
    if (m_Transform) {
        m_PredictionState.position = m_Transform->GetPosition();
        m_PredictionState.rotation = m_Transform->GetRotation();
        m_PredictionState.scale = m_Transform->GetScale();
    }
}

void NetworkTransform::RestorePredictionState() {
    if (m_Transform) {
        m_Transform->SetPosition(m_PredictionState.position);
        m_Transform->SetRotation(m_PredictionState.rotation);
        m_Transform->SetScale(m_PredictionState.scale);
    }
}

Vec3 NetworkTransform::InterpolateVec3(const Vec3& a, const Vec3& b, float t) const {
    return Vec3(
        a.x + (b.x - a.x) * t,
        a.y + (b.y - a.y) * t,
        a.z + (b.z - a.z) * t
    );
}

bool NetworkTransform::HasChanged(const Vec3& a, const Vec3& b, float threshold) const {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    float dz = a.z - b.z;
    return (dx * dx + dy * dy + dz * dz) > (threshold * threshold);
}
