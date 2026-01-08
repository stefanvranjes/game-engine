#pragma once

#include "NetEntityId.hpp"
#include "../../include/Transform.h"
#include "../../include/Math/Vec3.h"
#include <vector>
#include <cstdint>
#include <queue>

/**
 * @brief Network component for synchronizing Transform data across clients
 * 
 * Handles serialization, interpolation, and delta compression
 * for efficient network synchronization of position/rotation/scale.
 */
class NetworkTransform {
public:
    /**
     * @brief Configuration for sync behavior
     */
    struct Settings {
        float syncRate = 20.0f;              // Updates per second (server->client)
        float interpolationDelay = 0.1f;     // Seconds of buffering for smooth playback
        float snapshotThreshold = 0.001f;    // Min change to trigger update
        float teleportThreshold = 5.0f;      // Distance to teleport instead of interpolate
        
        bool syncPosition = true;
        bool syncRotation = true;
        bool syncScale = false;              // Usually static, disable by default
        
        // Compression settings
        uint8_t positionPrecision = 3;       // Decimal places (1000 = mm precision)
        uint8_t rotationPrecision = 2;       // Decimal places for angles
    };
    
    NetworkTransform();
    explicit NetworkTransform(Transform* transform);
    ~NetworkTransform() = default;
    
    // Core operations
    void SetTransform(Transform* transform) { m_Transform = transform; }
    Transform* GetTransform() const { return m_Transform; }
    
    void SetNetworkId(const NetEntityId& id) { m_NetId = id; }
    const NetEntityId& GetNetworkId() const { return m_NetId; }
    
    /**
     * @brief Update interpolation (call every frame on remote entities)
     */
    void Update(float deltaTime);
    
    /**
     * @brief Check if transform changed enough to require network update
     */
    bool IsDirty() const;
    
    /**
     * @brief Mark as synchronized (reset dirty state)
     */
    void ClearDirty();
    
    /**
     * @brief Serialize current transform state to bytes
     * @param outData Output buffer
     * @param fullState If true, sends all data; if false, only changed values
     */
    void Serialize(std::vector<uint8_t>& outData, bool fullState = false) const;
    
    /**
     * @brief Deserialize and apply received transform state
     * @param data Input buffer
     * @param offset Start position in buffer
     * @return Number of bytes consumed
     */
    size_t Deserialize(const uint8_t* data, size_t length);
    
    /**
     * @brief Add received state to interpolation buffer
     */
    void PushSnapshot(const Vec3& position, const Vec3& rotation, const Vec3& scale, float timestamp);
    
    // Settings
    Settings& GetSettings() { return m_Settings; }
    const Settings& GetSettings() const { return m_Settings; }
    
    // For owner prediction
    void SavePredictionState();
    void RestorePredictionState();
    
private:
    struct Snapshot {
        Vec3 position;
        Vec3 rotation;
        Vec3 scale;
        float timestamp;
    };
    
    Transform* m_Transform = nullptr;
    NetEntityId m_NetId;
    Settings m_Settings;
    
    // State tracking
    Vec3 m_LastSentPosition;
    Vec3 m_LastSentRotation;
    Vec3 m_LastSentScale;
    bool m_IsDirty = true;
    
    // Interpolation buffer for remote entities
    std::queue<Snapshot> m_SnapshotBuffer;
    float m_InterpolationTime = 0.0f;
    
    // Prediction rollback for local entity
    Snapshot m_PredictionState;
    
    // Helpers
    Vec3 InterpolateVec3(const Vec3& a, const Vec3& b, float t) const;
    bool HasChanged(const Vec3& a, const Vec3& b, float threshold) const;
};
