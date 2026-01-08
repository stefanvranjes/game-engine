#pragma once

#include "NetEntityId.hpp"
#include "NetworkTransform.hpp"
#include "Message.hpp"
#include <unordered_map>
#include <vector>
#include <memory>
#include <functional>
#include <cstdint>

// Forward declarations
class GameObject;
class NetworkManager;

/**
 * @brief Manages replication of game entities across the network
 * 
 * The ReplicationManager tracks all networked entities and handles:
 * - Entity registration/unregistration
 * - State snapshot creation and application
 * - Delta compression for bandwidth efficiency
 * - Priority-based updates for relevancy
 */
class ReplicationManager {
public:
    /**
     * @brief Replication settings
     */
    struct Settings {
        float snapshotRate = 20.0f;        // Snapshots per second
        float priorityUpdateRate = 5.0f;   // Priority recalculation rate
        size_t maxEntitiesPerSnapshot = 64; // Bandwidth limiting
        float relevancyDistance = 100.0f;  // Max distance for updates
        bool enableDeltaCompression = true;
    };
    
    /**
     * @brief Entity replication data
     */
    struct ReplicatedEntity {
        uint32_t networkId;
        GameObject* gameObject;
        std::unique_ptr<NetworkTransform> networkTransform;
        float priority = 1.0f;             // Higher = more frequent updates
        float lastUpdateTime = 0.0f;
        bool isDirty = true;
        uint32_t ownerPeerId = 0;          // 0 = server owned
    };
    
    ReplicationManager();
    explicit ReplicationManager(NetworkManager* networkManager);
    ~ReplicationManager();
    
    /**
     * @brief Register a GameObject for network replication
     * @param obj The GameObject to replicate
     * @param networkId Unique network ID (0 = auto-assign)
     * @param ownerPeerId Peer that owns this entity
     * @return Assigned network ID
     */
    uint32_t RegisterEntity(GameObject* obj, uint32_t networkId = 0, uint32_t ownerPeerId = 0);
    
    /**
     * @brief Unregister an entity from replication
     */
    void UnregisterEntity(uint32_t networkId);
    
    /**
     * @brief Get a replicated entity by network ID
     */
    ReplicatedEntity* GetEntity(uint32_t networkId);
    const ReplicatedEntity* GetEntity(uint32_t networkId) const;
    
    /**
     * @brief Update replication (call every frame)
     * @param deltaTime Time since last frame
     */
    void Update(float deltaTime);
    
    /**
     * @brief Create a full state snapshot for a new client
     * @param outData Output buffer
     */
    void CreateFullSnapshot(std::vector<uint8_t>& outData) const;
    
    /**
     * @brief Create a delta snapshot with only changed entities
     * @param outData Output buffer
     * @param targetPeerId Peer to send to (for relevancy calculation)
     */
    void CreateDeltaSnapshot(std::vector<uint8_t>& outData, uint32_t targetPeerId = 0);
    
    /**
     * @brief Apply received snapshot data
     * @param data Input buffer
     * @param length Buffer length
     */
    void ApplySnapshot(const uint8_t* data, size_t length);
    
    /**
     * @brief Process entity spawn message
     */
    void ProcessSpawn(const uint8_t* data, size_t length);
    
    /**
     * @brief Process entity destroy message
     */
    void ProcessDestroy(const uint8_t* data, size_t length);
    
    /**
     * @brief Set callback for entity spawn requests
     */
    using SpawnCallback = std::function<GameObject*(uint32_t networkId, uint32_t prefabId)>;
    void SetSpawnCallback(SpawnCallback callback) { m_SpawnCallback = callback; }
    
    /**
     * @brief Set callback for entity destroy notifications
     */
    using DestroyCallback = std::function<void(uint32_t networkId)>;
    void SetDestroyCallback(DestroyCallback callback) { m_DestroyCallback = callback; }
    
    // Settings
    Settings& GetSettings() { return m_Settings; }
    const Settings& GetSettings() const { return m_Settings; }
    
    // Statistics
    size_t GetEntityCount() const { return m_Entities.size(); }
    size_t GetDirtyEntityCount() const;
    
private:
    NetworkManager* m_NetworkManager = nullptr;
    Settings m_Settings;
    
    std::unordered_map<uint32_t, std::unique_ptr<ReplicatedEntity>> m_Entities;
    uint32_t m_NextNetworkId = 1000;  // Start after reserved IDs
    
    float m_TimeSinceLastSnapshot = 0.0f;
    float m_TimeSinceLastPriorityUpdate = 0.0f;
    
    SpawnCallback m_SpawnCallback;
    DestroyCallback m_DestroyCallback;
    
    // Priority calculation
    void UpdatePriorities();
    float CalculatePriority(const ReplicatedEntity& entity, uint32_t viewerPeerId) const;
    
    // Snapshot helpers
    void WriteEntityToSnapshot(std::vector<uint8_t>& data, const ReplicatedEntity& entity) const;
    void ReadEntityFromSnapshot(const uint8_t* data, size_t& offset, size_t length);
};
