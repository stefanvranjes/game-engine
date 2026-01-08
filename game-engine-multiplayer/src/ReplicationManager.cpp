#include "ReplicationManager.hpp"
#include "NetworkManager.hpp"
#include "../../include/GameObject.h"
#include <algorithm>
#include <cstring>

ReplicationManager::ReplicationManager()
    : m_NetworkManager(nullptr) {}

ReplicationManager::ReplicationManager(NetworkManager* networkManager)
    : m_NetworkManager(networkManager) {}

ReplicationManager::~ReplicationManager() = default;

uint32_t ReplicationManager::RegisterEntity(GameObject* obj, uint32_t networkId, uint32_t ownerPeerId) {
    if (!obj) return 0;
    
    // Auto-assign network ID if not provided
    if (networkId == 0) {
        networkId = m_NextNetworkId++;
    }
    
    auto entity = std::make_unique<ReplicatedEntity>();
    entity->networkId = networkId;
    entity->gameObject = obj;
    entity->ownerPeerId = ownerPeerId;
    entity->priority = 1.0f;
    entity->isDirty = true;
    
    // Create NetworkTransform component
    entity->networkTransform = std::make_unique<NetworkTransform>(&obj->GetTransform());
    
    NetEntityId netId;
    netId.networkId = networkId;
    netId.ownerId = ownerPeerId;
    netId.isLocal = (m_NetworkManager == nullptr); // Local if no network manager
    netId.isSpawned = true;
    entity->networkTransform->SetNetworkId(netId);
    
    m_Entities[networkId] = std::move(entity);
    
    return networkId;
}

void ReplicationManager::UnregisterEntity(uint32_t networkId) {
    m_Entities.erase(networkId);
}

ReplicationManager::ReplicatedEntity* ReplicationManager::GetEntity(uint32_t networkId) {
    auto it = m_Entities.find(networkId);
    return (it != m_Entities.end()) ? it->second.get() : nullptr;
}

const ReplicationManager::ReplicatedEntity* ReplicationManager::GetEntity(uint32_t networkId) const {
    auto it = m_Entities.find(networkId);
    return (it != m_Entities.end()) ? it->second.get() : nullptr;
}

void ReplicationManager::Update(float deltaTime) {
    m_TimeSinceLastSnapshot += deltaTime;
    m_TimeSinceLastPriorityUpdate += deltaTime;
    
    // Update priorities periodically
    if (m_TimeSinceLastPriorityUpdate >= 1.0f / m_Settings.priorityUpdateRate) {
        UpdatePriorities();
        m_TimeSinceLastPriorityUpdate = 0.0f;
    }
    
    // Update all NetworkTransform components
    for (auto& [id, entity] : m_Entities) {
        if (entity->networkTransform) {
            entity->networkTransform->Update(deltaTime);
            
            // Check if entity is dirty
            if (entity->networkTransform->IsDirty()) {
                entity->isDirty = true;
            }
        }
    }
}

void ReplicationManager::CreateFullSnapshot(std::vector<uint8_t>& outData) const {
    // Header: message type + entity count
    outData.push_back(static_cast<uint8_t>(MessageType::Snapshot));
    
    uint32_t entityCount = static_cast<uint32_t>(m_Entities.size());
    outData.push_back(static_cast<uint8_t>(entityCount & 0xFF));
    outData.push_back(static_cast<uint8_t>((entityCount >> 8) & 0xFF));
    
    for (const auto& [id, entity] : m_Entities) {
        WriteEntityToSnapshot(outData, *entity);
    }
}

void ReplicationManager::CreateDeltaSnapshot(std::vector<uint8_t>& outData, uint32_t targetPeerId) {
    // Collect dirty entities sorted by priority
    std::vector<ReplicatedEntity*> dirtyEntities;
    for (auto& [id, entity] : m_Entities) {
        if (entity->isDirty) {
            dirtyEntities.push_back(entity.get());
        }
    }
    
    // Sort by priority (highest first)
    std::sort(dirtyEntities.begin(), dirtyEntities.end(),
        [](const ReplicatedEntity* a, const ReplicatedEntity* b) {
            return a->priority > b->priority;
        });
    
    // Limit entities per snapshot
    size_t count = std::min(dirtyEntities.size(), m_Settings.maxEntitiesPerSnapshot);
    
    // Header
    outData.push_back(static_cast<uint8_t>(MessageType::Delta));
    outData.push_back(static_cast<uint8_t>(count & 0xFF));
    outData.push_back(static_cast<uint8_t>((count >> 8) & 0xFF));
    
    // Write entities
    for (size_t i = 0; i < count; ++i) {
        WriteEntityToSnapshot(outData, *dirtyEntities[i]);
        dirtyEntities[i]->isDirty = false;
        if (dirtyEntities[i]->networkTransform) {
            dirtyEntities[i]->networkTransform->ClearDirty();
        }
    }
}

void ReplicationManager::ApplySnapshot(const uint8_t* data, size_t length) {
    if (length < 3) return;
    
    size_t offset = 0;
    
    // Skip message type (already handled by caller)
    MessageType type = static_cast<MessageType>(data[offset++]);
    (void)type;
    
    uint16_t entityCount = 0;
    entityCount |= static_cast<uint16_t>(data[offset++]);
    entityCount |= static_cast<uint16_t>(data[offset++]) << 8;
    
    for (uint16_t i = 0; i < entityCount && offset < length; ++i) {
        ReadEntityFromSnapshot(data, offset, length);
    }
}

void ReplicationManager::ProcessSpawn(const uint8_t* data, size_t length) {
    if (length < 8) return;
    
    size_t offset = 0;
    
    // Read network ID
    uint32_t networkId = 0;
    networkId |= static_cast<uint32_t>(data[offset++]);
    networkId |= static_cast<uint32_t>(data[offset++]) << 8;
    networkId |= static_cast<uint32_t>(data[offset++]) << 16;
    networkId |= static_cast<uint32_t>(data[offset++]) << 24;
    
    // Read prefab ID
    uint32_t prefabId = 0;
    prefabId |= static_cast<uint32_t>(data[offset++]);
    prefabId |= static_cast<uint32_t>(data[offset++]) << 8;
    prefabId |= static_cast<uint32_t>(data[offset++]) << 16;
    prefabId |= static_cast<uint32_t>(data[offset++]) << 24;
    
    // Call spawn callback if set
    if (m_SpawnCallback) {
        GameObject* obj = m_SpawnCallback(networkId, prefabId);
        if (obj) {
            RegisterEntity(obj, networkId);
        }
    }
}

void ReplicationManager::ProcessDestroy(const uint8_t* data, size_t length) {
    if (length < 4) return;
    
    uint32_t networkId = 0;
    networkId |= static_cast<uint32_t>(data[0]);
    networkId |= static_cast<uint32_t>(data[1]) << 8;
    networkId |= static_cast<uint32_t>(data[2]) << 16;
    networkId |= static_cast<uint32_t>(data[3]) << 24;
    
    if (m_DestroyCallback) {
        m_DestroyCallback(networkId);
    }
    
    UnregisterEntity(networkId);
}

size_t ReplicationManager::GetDirtyEntityCount() const {
    size_t count = 0;
    for (const auto& [id, entity] : m_Entities) {
        if (entity->isDirty) ++count;
    }
    return count;
}

void ReplicationManager::UpdatePriorities() {
    // For now, use simple time-based priority
    // Entities that haven't been updated recently get higher priority
    float currentTime = m_TimeSinceLastSnapshot;
    
    for (auto& [id, entity] : m_Entities) {
        float timeSinceUpdate = currentTime - entity->lastUpdateTime;
        entity->priority = 1.0f + timeSinceUpdate * 0.1f;
        
        // Boost priority for dirty entities
        if (entity->isDirty) {
            entity->priority *= 1.5f;
        }
    }
}

float ReplicationManager::CalculatePriority(const ReplicatedEntity& entity, uint32_t viewerPeerId) const {
    // Base priority
    float priority = entity.priority;
    
    // TODO: Add distance-based priority when viewer position is available
    // TODO: Add visibility-based priority
    
    return priority;
}

void ReplicationManager::WriteEntityToSnapshot(std::vector<uint8_t>& data, 
                                                const ReplicatedEntity& entity) const {
    if (!entity.networkTransform) return;
    
    // Entity header: network ID (4 bytes) + owner ID (4 bytes)
    uint32_t netId = entity.networkId;
    data.push_back(static_cast<uint8_t>(netId & 0xFF));
    data.push_back(static_cast<uint8_t>((netId >> 8) & 0xFF));
    data.push_back(static_cast<uint8_t>((netId >> 16) & 0xFF));
    data.push_back(static_cast<uint8_t>((netId >> 24) & 0xFF));
    
    uint32_t ownerId = entity.ownerPeerId;
    data.push_back(static_cast<uint8_t>(ownerId & 0xFF));
    data.push_back(static_cast<uint8_t>((ownerId >> 8) & 0xFF));
    data.push_back(static_cast<uint8_t>((ownerId >> 16) & 0xFF));
    data.push_back(static_cast<uint8_t>((ownerId >> 24) & 0xFF));
    
    // Transform data
    entity.networkTransform->Serialize(data, false);
}

void ReplicationManager::ReadEntityFromSnapshot(const uint8_t* data, size_t& offset, size_t length) {
    if (offset + 8 > length) return;
    
    // Read network ID
    uint32_t networkId = 0;
    networkId |= static_cast<uint32_t>(data[offset++]);
    networkId |= static_cast<uint32_t>(data[offset++]) << 8;
    networkId |= static_cast<uint32_t>(data[offset++]) << 16;
    networkId |= static_cast<uint32_t>(data[offset++]) << 24;
    
    // Read owner ID
    uint32_t ownerId = 0;
    ownerId |= static_cast<uint32_t>(data[offset++]);
    ownerId |= static_cast<uint32_t>(data[offset++]) << 8;
    ownerId |= static_cast<uint32_t>(data[offset++]) << 16;
    ownerId |= static_cast<uint32_t>(data[offset++]) << 24;
    
    // Find or create entity
    auto entity = GetEntity(networkId);
    if (entity && entity->networkTransform) {
        size_t consumed = entity->networkTransform->Deserialize(data + offset, length - offset);
        offset += consumed;
    }
}
