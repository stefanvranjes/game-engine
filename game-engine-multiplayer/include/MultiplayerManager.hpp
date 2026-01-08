#pragma once

#include "NetworkManager.hpp"
#include "ReplicationManager.hpp"
#include "RPC.hpp"
#include "ClientPrediction.hpp"
#include <memory>
#include <functional>
#include <vector>
#include <string>

// Forward declarations
class GameObject;

/**
 * @brief High-level multiplayer API for common game networking patterns
 * 
 * Provides an easy-to-use interface for:
 * - Hosting and joining games
 * - Player connection/disconnection events
 * - Entity spawning across the network
 * - State synchronization
 */
class MultiplayerManager {
public:
    /**
     * @brief Connection state
     */
    enum class State {
        Disconnected,
        Connecting,
        Connected,
        Hosting
    };
    
    /**
     * @brief Configuration for multiplayer session
     */
    struct Config {
        uint16_t port = 12345;
        uint32_t maxPlayers = 16;
        float tickRate = 64.0f;        // Server updates per second
        float syncRate = 20.0f;        // State sync rate
        std::string serverAddress = "127.0.0.1";
    };
    
    /**
     * @brief Information about a connected player
     */
    struct PlayerInfo {
        uint32_t playerId;
        std::string name;
        uint32_t ping;
        bool isHost;
        bool isLocal;
    };
    
    MultiplayerManager();
    ~MultiplayerManager();
    
    // ========================================================================
    // Connection Management
    // ========================================================================
    
    /**
     * @brief Host a new game session
     * @param config Server configuration
     * @return True if server started successfully
     */
    bool HostGame(const Config& config = Config());
    
    /**
     * @brief Join an existing game
     * @param address Server address
     * @param port Server port
     * @return True if connection initiated (not yet connected)
     */
    bool JoinGame(const std::string& address, uint16_t port = 12345);
    
    /**
     * @brief Disconnect from current session
     */
    void Disconnect();
    
    /**
     * @brief Get current connection state
     */
    State GetState() const { return m_State; }
    
    /**
     * @brief Check if currently hosting
     */
    bool IsHost() const { return m_State == State::Hosting; }
    
    /**
     * @brief Check if connected (as host or client)
     */
    bool IsConnected() const { 
        return m_State == State::Connected || m_State == State::Hosting; 
    }
    
    // ========================================================================
    // Player Information
    // ========================================================================
    
    /**
     * @brief Get local player's network ID
     */
    uint32_t GetLocalPlayerId() const { return m_LocalPlayerId; }
    
    /**
     * @brief Get list of all connected players
     */
    std::vector<PlayerInfo> GetConnectedPlayers() const;
    
    /**
     * @brief Get player info by ID
     */
    const PlayerInfo* GetPlayer(uint32_t playerId) const;
    
    /**
     * @brief Get number of connected players
     */
    uint32_t GetPlayerCount() const;
    
    /**
     * @brief Get local player's ping to server (ms)
     */
    uint32_t GetPing() const;
    
    // ========================================================================
    // Entity Management
    // ========================================================================
    
    /**
     * @brief Spawn a networked entity (host only)
     * @param prefabId ID of prefab to spawn
     * @param position Initial position
     * @param ownerPlayerId Player who owns this entity (0 = server)
     * @return Network ID of spawned entity, or 0 on failure
     */
    uint32_t SpawnEntity(uint32_t prefabId, const Vec3& position, uint32_t ownerPlayerId = 0);
    
    /**
     * @brief Destroy a networked entity (host/owner only)
     * @param networkId Network ID of entity to destroy
     */
    void DestroyEntity(uint32_t networkId);
    
    /**
     * @brief Register a GameObject for replication
     * @param obj GameObject to replicate
     * @return Assigned network ID
     */
    uint32_t RegisterEntity(GameObject* obj);
    
    /**
     * @brief Get the ReplicationManager for advanced usage
     */
    ReplicationManager& GetReplicationManager() { return m_ReplicationManager; }
    
    /**
     * @brief Get the RPCManager for remote calls
     */
    RPCManager& GetRPCManager() { return m_RPCManager; }
    
    // ========================================================================
    // Update Loop
    // ========================================================================
    
    /**
     * @brief Update networking (call every frame)
     * @param deltaTime Time since last frame
     */
    void Update(float deltaTime);
    
    // ========================================================================
    // Events / Callbacks
    // ========================================================================
    
    std::function<void(uint32_t playerId)> OnPlayerJoined;
    std::function<void(uint32_t playerId)> OnPlayerLeft;
    std::function<void()> OnConnected;
    std::function<void(const std::string& reason)> OnDisconnected;
    std::function<void(const std::string& reason)> OnConnectionFailed;
    std::function<GameObject*(uint32_t prefabId, const Vec3& position)> OnSpawnRequest;
    
private:
    State m_State = State::Disconnected;
    Config m_Config;
    
    std::unique_ptr<NetworkManager> m_NetworkManager;
    ReplicationManager m_ReplicationManager;
    RPCManager m_RPCManager;
    
    uint32_t m_LocalPlayerId = 0;
    std::unordered_map<uint32_t, PlayerInfo> m_Players;
    
    float m_TimeSinceLastSync = 0.0f;
    float m_TimeSinceLastTick = 0.0f;
    
    // Internal handlers
    void OnMessageReceived(const Message& msg, uint32_t peerId);
    void ProcessPlayerJoin(uint32_t peerId);
    void ProcessPlayerLeave(uint32_t peerId);
    void SendStateUpdates();
};
