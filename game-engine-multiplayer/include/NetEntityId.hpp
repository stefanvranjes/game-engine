#pragma once

#include <cstdint>

/**
 * @brief Unique identifier for network-synchronized entities
 * 
 * Each networked entity has a unique ID that persists across
 * all connected clients. The server assigns IDs for spawned entities.
 */
struct NetEntityId {
    uint32_t networkId = 0;     // Unique ID across the network session
    uint32_t ownerId = 0;       // Peer ID that owns/controls this entity
    bool isLocal = false;       // True if this client owns the entity
    bool isSpawned = false;     // True if entity has been replicated
    
    bool IsValid() const { return networkId != 0; }
    bool IsOwner() const { return isLocal && ownerId != 0; }
    
    bool operator==(const NetEntityId& other) const {
        return networkId == other.networkId;
    }
    
    bool operator!=(const NetEntityId& other) const {
        return networkId != other.networkId;
    }
    
    // For use in hash maps
    struct Hash {
        size_t operator()(const NetEntityId& id) const {
            return std::hash<uint32_t>()(id.networkId);
        }
    };
};

// Special network ID values
namespace NetId {
    inline constexpr uint32_t Invalid = 0;
    inline constexpr uint32_t Server = 1;
    inline constexpr uint32_t Broadcast = 0xFFFFFFFF;
}
