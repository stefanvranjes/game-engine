#include "MultiplayerManager.hpp"
#include "../../include/GameObject.h"
#include <iostream>

MultiplayerManager::MultiplayerManager()
    : m_State(State::Disconnected)
    , m_LocalPlayerId(0) {}

MultiplayerManager::~MultiplayerManager() {
    Disconnect();
}

bool MultiplayerManager::HostGame(const Config& config) {
    if (m_State != State::Disconnected) {
        Disconnect();
    }
    
    m_Config = config;
    
    m_NetworkManager = std::make_unique<NetworkManager>(
        NetworkManager::Mode::Server, 
        config.port, 
        config.maxPlayers
    );
    
    m_NetworkManager->setMessageHandler(
        [this](const Message& msg, uint32_t peerId) {
            OnMessageReceived(msg, peerId);
        }
    );
    
    if (!m_NetworkManager->initialize()) {
        m_NetworkManager.reset();
        if (OnConnectionFailed) {
            OnConnectionFailed("Failed to start server");
        }
        return false;
    }
    
    m_State = State::Hosting;
    m_LocalPlayerId = 1;  // Server is always player 1
    
    // Add host as a player
    PlayerInfo hostInfo;
    hostInfo.playerId = m_LocalPlayerId;
    hostInfo.name = "Host";
    hostInfo.ping = 0;
    hostInfo.isHost = true;
    hostInfo.isLocal = true;
    m_Players[m_LocalPlayerId] = hostInfo;
    
    std::cout << "[Multiplayer] Hosting game on port " << config.port << std::endl;
    
    if (OnConnected) {
        OnConnected();
    }
    
    return true;
}

bool MultiplayerManager::JoinGame(const std::string& address, uint16_t port) {
    if (m_State != State::Disconnected) {
        Disconnect();
    }
    
    m_Config.serverAddress = address;
    m_Config.port = port;
    
    m_NetworkManager = std::make_unique<NetworkManager>(
        NetworkManager::Mode::Client, 
        port,
        1  // Client only needs 1 peer slot (the server)
    );
    
    m_NetworkManager->setMessageHandler(
        [this](const Message& msg, uint32_t peerId) {
            OnMessageReceived(msg, peerId);
        }
    );
    
    if (!m_NetworkManager->initialize()) {
        m_NetworkManager.reset();
        if (OnConnectionFailed) {
            OnConnectionFailed("Failed to connect to server");
        }
        return false;
    }
    
    m_State = State::Connecting;
    
    std::cout << "[Multiplayer] Connecting to " << address << ":" << port << std::endl;
    
    return true;
}

void MultiplayerManager::Disconnect() {
    if (m_State == State::Disconnected) return;
    
    if (m_NetworkManager) {
        m_NetworkManager->shutdown();
        m_NetworkManager.reset();
    }
    
    m_Players.clear();
    m_LocalPlayerId = 0;
    m_State = State::Disconnected;
    
    std::cout << "[Multiplayer] Disconnected" << std::endl;
    
    if (OnDisconnected) {
        OnDisconnected("User disconnected");
    }
}

std::vector<MultiplayerManager::PlayerInfo> MultiplayerManager::GetConnectedPlayers() const {
    std::vector<PlayerInfo> result;
    result.reserve(m_Players.size());
    for (const auto& [id, info] : m_Players) {
        result.push_back(info);
    }
    return result;
}

const MultiplayerManager::PlayerInfo* MultiplayerManager::GetPlayer(uint32_t playerId) const {
    auto it = m_Players.find(playerId);
    return (it != m_Players.end()) ? &it->second : nullptr;
}

uint32_t MultiplayerManager::GetPlayerCount() const {
    return static_cast<uint32_t>(m_Players.size());
}

uint32_t MultiplayerManager::GetPing() const {
    // For host, ping is 0
    if (IsHost()) return 0;
    
    // TODO: Get actual ping from NetworkManager
    return 0;
}

uint32_t MultiplayerManager::SpawnEntity(uint32_t prefabId, const Vec3& position, 
                                          uint32_t ownerPlayerId) {
    if (!IsHost()) {
        std::cerr << "[Multiplayer] Only host can spawn entities" << std::endl;
        return 0;
    }
    
    // Request spawn from game
    GameObject* obj = nullptr;
    if (OnSpawnRequest) {
        obj = OnSpawnRequest(prefabId, position);
    }
    
    if (!obj) {
        std::cerr << "[Multiplayer] Spawn request failed for prefab " << prefabId << std::endl;
        return 0;
    }
    
    // Register with replication
    uint32_t networkId = m_ReplicationManager.RegisterEntity(obj, 0, ownerPlayerId);
    
    // Send spawn message to all clients
    std::vector<uint8_t> spawnData;
    spawnData.push_back(static_cast<uint8_t>(MessageType::Spawn));
    
    // Network ID
    spawnData.push_back(static_cast<uint8_t>(networkId & 0xFF));
    spawnData.push_back(static_cast<uint8_t>((networkId >> 8) & 0xFF));
    spawnData.push_back(static_cast<uint8_t>((networkId >> 16) & 0xFF));
    spawnData.push_back(static_cast<uint8_t>((networkId >> 24) & 0xFF));
    
    // Prefab ID
    spawnData.push_back(static_cast<uint8_t>(prefabId & 0xFF));
    spawnData.push_back(static_cast<uint8_t>((prefabId >> 8) & 0xFF));
    spawnData.push_back(static_cast<uint8_t>((prefabId >> 16) & 0xFF));
    spawnData.push_back(static_cast<uint8_t>((prefabId >> 24) & 0xFF));
    
    // Position
    auto writeFloat = [&spawnData](float value) {
        uint32_t bits;
        std::memcpy(&bits, &value, sizeof(float));
        spawnData.push_back(static_cast<uint8_t>(bits & 0xFF));
        spawnData.push_back(static_cast<uint8_t>((bits >> 8) & 0xFF));
        spawnData.push_back(static_cast<uint8_t>((bits >> 16) & 0xFF));
        spawnData.push_back(static_cast<uint8_t>((bits >> 24) & 0xFF));
    };
    writeFloat(position.x);
    writeFloat(position.y);
    writeFloat(position.z);
    
    Message spawnMsg(MessageType::Spawn, std::string(spawnData.begin(), spawnData.end()));
    m_NetworkManager->sendMessage(spawnMsg, 0);  // Broadcast
    
    return networkId;
}

void MultiplayerManager::DestroyEntity(uint32_t networkId) {
    // TODO: Check ownership
    
    m_ReplicationManager.UnregisterEntity(networkId);
    
    // Send destroy message
    std::vector<uint8_t> destroyData;
    destroyData.push_back(static_cast<uint8_t>(MessageType::Destroy));
    destroyData.push_back(static_cast<uint8_t>(networkId & 0xFF));
    destroyData.push_back(static_cast<uint8_t>((networkId >> 8) & 0xFF));
    destroyData.push_back(static_cast<uint8_t>((networkId >> 16) & 0xFF));
    destroyData.push_back(static_cast<uint8_t>((networkId >> 24) & 0xFF));
    
    Message destroyMsg(MessageType::Destroy, std::string(destroyData.begin(), destroyData.end()));
    m_NetworkManager->sendMessage(destroyMsg, 0);  // Broadcast
}

uint32_t MultiplayerManager::RegisterEntity(GameObject* obj) {
    return m_ReplicationManager.RegisterEntity(obj, 0, m_LocalPlayerId);
}

void MultiplayerManager::Update(float deltaTime) {
    if (!m_NetworkManager || m_State == State::Disconnected) return;
    
    // Poll network events
    m_NetworkManager->update();
    
    // Update replication
    m_ReplicationManager.Update(deltaTime);
    
    // Send periodic state updates (host only)
    if (IsHost()) {
        m_TimeSinceLastSync += deltaTime;
        float syncInterval = 1.0f / m_Config.syncRate;
        
        if (m_TimeSinceLastSync >= syncInterval) {
            SendStateUpdates();
            m_TimeSinceLastSync = 0.0f;
        }
    }
    
    // Process pending RPCs
    auto pendingRPCs = m_RPCManager.GetPendingRPCs();
    for (const auto& [target, data] : pendingRPCs) {
        Message rpcMsg(MessageType::RPC, std::string(data.begin(), data.end()));
        
        switch (target) {
            case RPCTarget::Server:
                if (!IsHost()) {
                    m_NetworkManager->sendMessage(rpcMsg, 0);
                }
                break;
            case RPCTarget::All:
                m_NetworkManager->sendMessage(rpcMsg, 0);  // Broadcast
                break;
            case RPCTarget::Others:
                // TODO: Implement send to all except self
                m_NetworkManager->sendMessage(rpcMsg, 0);
                break;
            default:
                break;
        }
    }
}

void MultiplayerManager::OnMessageReceived(const Message& msg, uint32_t peerId) {
    switch (msg.getType()) {
        case MessageType::Join:
            ProcessPlayerJoin(peerId);
            break;
            
        case MessageType::Leave:
            ProcessPlayerLeave(peerId);
            break;
            
        case MessageType::Snapshot:
        case MessageType::Delta: {
            auto content = msg.getContent();
            m_ReplicationManager.ApplySnapshot(
                reinterpret_cast<const uint8_t*>(content.data()),
                content.size()
            );
            break;
        }
        
        case MessageType::Spawn: {
            auto content = msg.getContent();
            m_ReplicationManager.ProcessSpawn(
                reinterpret_cast<const uint8_t*>(content.data()),
                content.size()
            );
            break;
        }
        
        case MessageType::Destroy: {
            auto content = msg.getContent();
            m_ReplicationManager.ProcessDestroy(
                reinterpret_cast<const uint8_t*>(content.data()),
                content.size()
            );
            break;
        }
        
        case MessageType::RPC: {
            auto content = msg.getContent();
            m_RPCManager.ProcessRPC(
                reinterpret_cast<const uint8_t*>(content.data()),
                content.size(),
                peerId
            );
            break;
        }
        
        case MessageType::Connect:
            if (m_State == State::Connecting) {
                m_State = State::Connected;
                // Extract assigned player ID from message
                auto content = msg.getContent();
                if (content.size() >= 4) {
                    m_LocalPlayerId = 0;
                    m_LocalPlayerId |= static_cast<uint32_t>(static_cast<uint8_t>(content[0]));
                    m_LocalPlayerId |= static_cast<uint32_t>(static_cast<uint8_t>(content[1])) << 8;
                    m_LocalPlayerId |= static_cast<uint32_t>(static_cast<uint8_t>(content[2])) << 16;
                    m_LocalPlayerId |= static_cast<uint32_t>(static_cast<uint8_t>(content[3])) << 24;
                }
                
                std::cout << "[Multiplayer] Connected as player " << m_LocalPlayerId << std::endl;
                
                if (OnConnected) {
                    OnConnected();
                }
            }
            break;
            
        default:
            break;
    }
}

void MultiplayerManager::ProcessPlayerJoin(uint32_t peerId) {
    PlayerInfo info;
    info.playerId = peerId;
    info.name = "Player " + std::to_string(peerId);
    info.ping = 0;
    info.isHost = false;
    info.isLocal = (peerId == m_LocalPlayerId);
    
    m_Players[peerId] = info;
    
    std::cout << "[Multiplayer] Player " << peerId << " joined" << std::endl;
    
    if (OnPlayerJoined) {
        OnPlayerJoined(peerId);
    }
    
    // Host sends welcome packet with player ID
    if (IsHost()) {
        std::vector<uint8_t> welcomeData;
        welcomeData.push_back(static_cast<uint8_t>(peerId & 0xFF));
        welcomeData.push_back(static_cast<uint8_t>((peerId >> 8) & 0xFF));
        welcomeData.push_back(static_cast<uint8_t>((peerId >> 16) & 0xFF));
        welcomeData.push_back(static_cast<uint8_t>((peerId >> 24) & 0xFF));
        
        Message welcomeMsg(MessageType::Connect, 
                          std::string(welcomeData.begin(), welcomeData.end()));
        m_NetworkManager->sendMessage(welcomeMsg, peerId);
        
        // Send full state snapshot to new player
        std::vector<uint8_t> snapshotData;
        m_ReplicationManager.CreateFullSnapshot(snapshotData);
        Message snapshotMsg(MessageType::Snapshot, 
                           std::string(snapshotData.begin(), snapshotData.end()));
        m_NetworkManager->sendMessage(snapshotMsg, peerId);
    }
}

void MultiplayerManager::ProcessPlayerLeave(uint32_t peerId) {
    m_Players.erase(peerId);
    
    std::cout << "[Multiplayer] Player " << peerId << " left" << std::endl;
    
    if (OnPlayerLeft) {
        OnPlayerLeft(peerId);
    }
}

void MultiplayerManager::SendStateUpdates() {
    std::vector<uint8_t> deltaData;
    m_ReplicationManager.CreateDeltaSnapshot(deltaData, 0);
    
    if (!deltaData.empty()) {
        Message deltaMsg(MessageType::Delta, 
                        std::string(deltaData.begin(), deltaData.end()));
        m_NetworkManager->sendMessage(deltaMsg, 0);  // Broadcast
    }
}
