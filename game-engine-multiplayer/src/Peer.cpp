#include "Peer.hpp"

Peer::Peer() : state(Disconnected) {}

void Peer::connect(const std::string& address, unsigned short port) {
    // Implementation for connecting to a server
}

void Peer::disconnect() {
    // Implementation for disconnecting from the server
}

void Peer::sendMessage(const Message& message) {
    // Implementation for sending a message to the connected peer
}

void Peer::receiveMessage(Message& message) {
    // Implementation for receiving a message from the connected peer
}

Peer::State Peer::getState() const {
    return state;
}

void Peer::setState(State newState) {
    state = newState;
}