#include "Peer.hpp"
#include <enet/enet.h>

Peer::Peer() 
    : m_State(State::Disconnected)
    , m_ENetPeer(nullptr)
    , m_NetworkId(0) {}

Peer::Peer(const std::string& id)
    : m_Id(id)
    , m_State(State::Disconnected)
    , m_ENetPeer(nullptr)
    , m_NetworkId(0) {}

Peer::Peer(_ENetPeer* enetPeer)
    : m_State(State::Connected)
    , m_ENetPeer(enetPeer)
    , m_NetworkId(0) {
    if (enetPeer) {
        m_NetworkId = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(enetPeer->data));
    }
}

Peer::~Peer() {
    if (m_State != State::Disconnected && m_ENetPeer) {
        DisconnectLater();
    }
}

void Peer::Connect(const std::string& address, unsigned short port) {
    // Connection is handled by NetworkManager
    m_State = State::Connecting;
}

void Peer::Disconnect() {
    if (m_ENetPeer && m_State == State::Connected) {
        enet_peer_disconnect(m_ENetPeer, 0);
        m_State = State::Disconnecting;
    }
}

void Peer::DisconnectLater() {
    if (m_ENetPeer && m_State == State::Connected) {
        enet_peer_disconnect_later(m_ENetPeer, 0);
        m_State = State::Disconnecting;
    }
}

void Peer::SendMessage(const std::string& message) {
    SendData(reinterpret_cast<const uint8_t*>(message.data()), message.size(), true);
}

void Peer::SendData(const uint8_t* data, size_t length, bool reliable) {
    if (!m_ENetPeer || m_State != State::Connected) return;
    
    ENetPacket* packet = enet_packet_create(
        data, 
        length, 
        reliable ? ENET_PACKET_FLAG_RELIABLE : 0
    );
    
    enet_peer_send(m_ENetPeer, 0, packet);
}

std::string Peer::ReceiveMessage() {
    // Receiving is handled by NetworkManager's event loop
    return "";
}

Peer::State Peer::GetState() const {
    return m_State;
}

void Peer::SetState(State newState) {
    m_State = newState;
}

std::string Peer::GetId() const {
    return m_Id;
}

void Peer::SetId(const std::string& newId) {
    m_Id = newId;
}

uint32_t Peer::GetRoundTripTime() const {
    if (m_ENetPeer) {
        return m_ENetPeer->roundTripTime;
    }
    return 0;
}

float Peer::GetPacketLoss() const {
    if (m_ENetPeer) {
        return static_cast<float>(m_ENetPeer->packetLoss) / ENET_PEER_PACKET_LOSS_SCALE;
    }
    return 0.0f;
}