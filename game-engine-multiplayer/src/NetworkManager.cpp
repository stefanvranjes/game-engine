#include "../include/NetworkManager.hpp"
#include <enet/enet.h>
#include <iostream>
#include <unordered_map>
#include <queue>
#include <mutex>

struct NetworkManager::Impl {
    Mode mode;
    uint16_t port;
    size_t max_peers;

    ENetHost* host = nullptr;
    std::unordered_map<uint32_t, ENetPeer*> peers; // peer_id -> ENetPeer*
    std::function<void(const Message&, uint32_t)> handler;
    std::atomic<bool> running{false};
    std::mutex peer_mtx;

    bool initServer() {
        ENetAddress address;
        address.host = ENET_HOST_ANY;
        address.port = port;

        host = enet_host_create(&address, max_peers, 2, 0, 0);
        if (!host) {
            std::cerr << "[ENet] Failed to create server host on port " << port << std::endl;
            return false;
        }
        std::cout << "[ENet] Server initialized on port " << port << std::endl;
        return true;
    }

    bool initClient() {
        host = enet_host_create(nullptr, 1, 2, 0, 0);
        if (!host) {
            std::cerr << "[ENet] Failed to create client host" << std::endl;
            return false;
        }

        ENetAddress address;
        enet_address_set_host(&address, "127.0.0.1");
        address.port = port;

        ENetPeer* peer = enet_host_connect(host, &address, 2, 0);
        if (!peer) {
            std::cerr << "[ENet] Failed to connect to server" << std::endl;
            enet_host_destroy(host);
            host = nullptr;
            return false;
        }

        std::lock_guard<std::mutex> lk(peer_mtx);
        peers[0] = peer;
        std::cout << "[ENet] Client connecting to 127.0.0.1:" << port << std::endl;
        return true;
    }

    void pollEvents() {
        if (!host) return;

        ENetEvent event;
        while (enet_host_service(host, &event, 0) > 0) {
            switch (event.type) {
            case ENET_EVENT_TYPE_CONNECT: {
                uint32_t peer_id = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(event.peer->data));
                if (peer_id == 0) {
                    // Assign new peer ID (server-side)
                    static uint32_t next_id = 1;
                    peer_id = next_id++;
                    event.peer->data = reinterpret_cast<void*>(static_cast<uintptr_t>(peer_id));
                }
                {
                    std::lock_guard<std::mutex> lk(peer_mtx);
                    peers[peer_id] = event.peer;
                }
                std::cout << "[ENet] Peer connected: ID=" << peer_id << std::endl;
                break;
            }

            case ENET_EVENT_TYPE_RECEIVE: {
                uint32_t peer_id = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(event.peer->data));
                if (!peer_id) peer_id = 0; // client uses ID 0 for server

                Message msg;
                std::vector<uint8_t> data(event.packet->data, event.packet->data + event.packet->dataLength);
                if (Message::deserialize(data, msg) && handler) {
                    handler(msg, peer_id);
                }
                enet_packet_destroy(event.packet);
                break;
            }

            case ENET_EVENT_TYPE_DISCONNECT: {
                uint32_t peer_id = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(event.peer->data));
                {
                    std::lock_guard<std::mutex> lk(peer_mtx);
                    peers.erase(peer_id);
                }
                std::cout << "[ENet] Peer disconnected: ID=" << peer_id << std::endl;
                event.peer->data = nullptr;
                break;
            }

            default:
                break;
            }
        }
    }

    void cleanup() {
        if (host) {
            enet_host_destroy(host);
            host = nullptr;
        }
    }
};

NetworkManager::NetworkManager(Mode mode, uint16_t port, size_t max_peers)
    : impl(std::make_unique<Impl>()) {
    impl->mode = mode;
    impl->port = port;
    impl->max_peers = max_peers;
}

NetworkManager::~NetworkManager() {
    shutdown();
}

bool NetworkManager::initialize() {
    if (enet_initialize() != 0) {
        std::cerr << "[ENet] Failed to initialize ENet" << std::endl;
        return false;
    }

    bool success = (impl->mode == Mode::Server) ? impl->initServer() : impl->initClient();
    if (success) {
        impl->running.store(true);
    }
    return success;
}

void NetworkManager::shutdown() {
    impl->running.store(false);
    impl->cleanup();
    enet_deinitialize();
}

void NetworkManager::sendMessage(const Message& msg, uint32_t peer_id) {
    if (!impl->host) return;

    std::vector<uint8_t> serialized = msg.serialize();
    ENetPacket* packet = enet_packet_create(serialized.data(), serialized.size(), ENET_PACKET_FLAG_RELIABLE);

    std::lock_guard<std::mutex> lk(impl->peer_mtx);
    if (peer_id == 0) {
        // Send to all peers (server broadcast)
        enet_host_broadcast(impl->host, 0, packet);
    } else if (impl->peers.count(peer_id)) {
        // Send to specific peer
        enet_peer_send(impl->peers[peer_id], 0, packet);
    } else {
        enet_packet_destroy(packet);
    }
}

void NetworkManager::setMessageHandler(std::function<void(const Message&, uint32_t)> handler) {
    impl->handler = std::move(handler);
}

void NetworkManager::update() {
    impl->pollEvents();
}

bool NetworkManager::isRunning() const {
    return impl->running.load();
}