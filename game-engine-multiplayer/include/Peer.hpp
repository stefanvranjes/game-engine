#pragma once

#include <string>
#include <cstdint>
#include <memory>

// Forward declaration
struct _ENetPeer;

/**
 * @brief Represents a connected network peer (client or server)
 * 
 * Manages the connection state and provides an abstraction over
 * the underlying ENet peer for sending/receiving messages.
 */
class Peer {
public:
    enum class State : uint8_t {
        Disconnected = 0,
        Connecting = 1,
        Connected = 2,
        Disconnecting = 3
    };

    Peer();
    explicit Peer(const std::string& id);
    explicit Peer(_ENetPeer* enetPeer);
    ~Peer();

    // Connection management
    void Connect(const std::string& address, unsigned short port);
    void Disconnect();
    void DisconnectLater();  // Graceful disconnect after pending data sent
    
    // Messaging
    void SendMessage(const std::string& message);
    void SendData(const uint8_t* data, size_t length, bool reliable = true);
    std::string ReceiveMessage();

    // State accessors
    State GetState() const;
    void SetState(State newState);
    
    std::string GetId() const;
    void SetId(const std::string& newId);
    
    uint32_t GetNetworkId() const { return m_NetworkId; }
    void SetNetworkId(uint32_t id) { m_NetworkId = id; }
    
    // Connection quality
    uint32_t GetRoundTripTime() const;  // RTT in milliseconds
    float GetPacketLoss() const;        // 0.0 to 1.0
    
    // ENet peer access (for internal use)
    _ENetPeer* GetENetPeer() const { return m_ENetPeer; }
    void SetENetPeer(_ENetPeer* peer) { m_ENetPeer = peer; }
    
    bool IsConnected() const { return m_State == State::Connected; }

private:
    std::string m_Id;
    uint32_t m_NetworkId = 0;
    State m_State = State::Disconnected;
    _ENetPeer* m_ENetPeer = nullptr;
};