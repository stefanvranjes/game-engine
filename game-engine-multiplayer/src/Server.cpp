#include "Server.hpp"
#include "NetworkManager.hpp"
#include "Message.hpp"
#include "Peer.hpp"
#include <asio.hpp>
#include <iostream>
#include <vector>
#include <thread>

Server::Server(int port) : port(port), running(false) {}

void Server::start() {
    running = true;
    try {
        asio::io_context io_context;
        asio::ip::tcp::acceptor acceptor(io_context, asio::ip::tcp::endpoint(asio::ip::tcp::v4(), port));
        std::cout << "Server started on port " << port << std::endl;
        
        while (running) {
            std::shared_ptr<Peer> new_peer = std::make_shared<Peer>(io_context);
            acceptor.accept(new_peer->get_socket());
            clients.push_back(new_peer);
            std::cout << "New client connected!" << std::endl;
        }
    } catch (std::exception& e) {
        std::cerr << "Server error: " << e.what() << std::endl;
        running = false;
    }
}

void Server::stop() {
    running = false;
}

void Server::broadcastMessage(const Message& message) {
    std::vector<uint8_t> data = message.serialize();
    for (auto& client : clients) {
        client->SendData(data.data(), data.size());
    }
}

void Server::acceptConnections() {
    // Handled in start() for now
}

void Server::handleClient(std::shared_ptr<Peer> client) {
    // Basic implementation
}