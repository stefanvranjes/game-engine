#pragma once
#include "Message.hpp"
#include <functional>
#include <memory>
#include <cstdint>

class NetworkManager {
public:
    enum class Mode { Server, Client };

    // port: listen port for server, connect port for client
    // max_peers: max concurrent connections (server-side)
    explicit NetworkManager(Mode mode, uint16_t port = 12345, size_t max_peers = 32);
    ~NetworkManager();

    // Initialize and start network (blocking until stop() called, should run in separate thread)
    bool initialize();
    void shutdown();

    // Send message to peer (client sends to server, server broadcasts or sends to specific peer)
    void sendMessage(const Message& msg, uint32_t peer_id = 0);

    // Set callback for received messages
    void setMessageHandler(std::function<void(const Message&, uint32_t peer_id)> handler);

    // Poll network events (call regularly from game loop)
    void update();

    bool isRunning() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};