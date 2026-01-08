#pragma once

#include "Message.hpp"
#include <string>
#include <vector>
#include <functional>
#include <unordered_map>
#include <any>
#include <cstdint>

/**
 * @brief Target for RPC invocation
 */
enum class RPCTarget : uint8_t {
    Server = 0,     // Send to server only (client -> server)
    Owner = 1,      // Send to entity owner only
    All = 2,        // Send to all connected clients
    Others = 3      // Send to all except the caller
};

/**
 * @brief Delivery mode for RPC
 */
enum class RPCMode : uint8_t {
    Reliable = 0,       // Guaranteed delivery, ordered
    Unreliable = 1,     // Best effort, unordered
    ReliableUnordered = 2  // Guaranteed but may arrive out of order
};

/**
 * @brief RPC invocation metadata
 */
struct RPCInfo {
    uint32_t callerId = 0;      // Peer ID of the caller
    uint32_t entityId = 0;      // Network entity ID (if entity-specific RPC)
    float timestamp = 0.0f;      // When the RPC was invoked
};

/**
 * @brief Remote Procedure Call manager for network gameplay events
 * 
 * Provides a type-safe way to invoke methods across the network.
 * Supports both entity-specific RPCs and global RPCs.
 */
class RPCManager {
public:
    RPCManager() = default;
    ~RPCManager() = default;
    
    /**
     * @brief Register an RPC handler
     * @param methodName Unique name for the RPC method
     * @param handler Function to call when RPC is received
     */
    template<typename... Args>
    void RegisterHandler(const std::string& methodName, 
                         std::function<void(const RPCInfo&, Args...)> handler) {
        m_Handlers[methodName] = [handler](const RPCInfo& info, const std::vector<uint8_t>& data) {
            // Deserialize args and call handler
            // This is a simplified version - full impl would use template magic
            handler(info);
        };
    }
    
    // Simplified registration for common cases
    void RegisterHandler(const std::string& methodName, 
                         std::function<void(const RPCInfo&)> handler);
    
    void RegisterHandler(const std::string& methodName,
                         std::function<void(const RPCInfo&, const std::string&)> handler);
    
    void RegisterHandler(const std::string& methodName,
                         std::function<void(const RPCInfo&, int32_t)> handler);
    
    void RegisterHandler(const std::string& methodName,
                         std::function<void(const RPCInfo&, float, float, float)> handler);
    
    /**
     * @brief Unregister an RPC handler
     */
    void UnregisterHandler(const std::string& methodName);
    
    /**
     * @brief Invoke an RPC (queues for sending)
     * @param methodName Name of the registered RPC
     * @param target Who should receive this RPC
     * @param mode Delivery mode
     * @param entityId Optional entity ID for entity-specific RPCs
     */
    void Invoke(const std::string& methodName, RPCTarget target, 
                RPCMode mode = RPCMode::Reliable, uint32_t entityId = 0);
    
    // Overloads for common parameter types
    void Invoke(const std::string& methodName, RPCTarget target,
                const std::string& param1, RPCMode mode = RPCMode::Reliable);
    
    void Invoke(const std::string& methodName, RPCTarget target,
                int32_t param1, RPCMode mode = RPCMode::Reliable);
    
    void Invoke(const std::string& methodName, RPCTarget target,
                float x, float y, float z, RPCMode mode = RPCMode::Reliable);
    
    /**
     * @brief Process received RPC message
     * @param data Serialized RPC data
     * @param length Data length
     * @param senderId Peer ID of the sender
     */
    void ProcessRPC(const uint8_t* data, size_t length, uint32_t senderId);
    
    /**
     * @brief Get pending RPC messages to send
     */
    std::vector<std::pair<RPCTarget, std::vector<uint8_t>>> GetPendingRPCs();
    
    /**
     * @brief Clear pending RPC queue
     */
    void ClearPending();
    
private:
    using HandlerFunc = std::function<void(const RPCInfo&, const std::vector<uint8_t>&)>;
    std::unordered_map<std::string, HandlerFunc> m_Handlers;
    
    struct PendingRPC {
        std::string methodName;
        RPCTarget target;
        RPCMode mode;
        uint32_t entityId;
        std::vector<uint8_t> params;
    };
    std::vector<PendingRPC> m_PendingRPCs;
    
    // Serialization helpers
    void SerializeRPC(std::vector<uint8_t>& outData, const PendingRPC& rpc) const;
    bool DeserializeRPC(const uint8_t* data, size_t length, 
                        std::string& methodName, uint32_t& entityId,
                        std::vector<uint8_t>& params) const;
};

// ============================================================================
// Macros for easier RPC definition
// ============================================================================

/**
 * @brief Declare an RPC method in a class header
 * Usage: DECLARE_RPC(OnPlayerDamage, float damage, int attackerId);
 */
#define DECLARE_RPC(MethodName, ...) \
    void MethodName##_RPC(__VA_ARGS__); \
    void MethodName(__VA_ARGS__)

/**
 * @brief Register an RPC handler in constructor
 * Usage: REGISTER_RPC(this, OnPlayerDamage);
 */
#define REGISTER_RPC(Manager, MethodName) \
    (Manager).RegisterHandler(#MethodName, [this](const RPCInfo& info) { \
        this->MethodName##_RPC(); \
    })
