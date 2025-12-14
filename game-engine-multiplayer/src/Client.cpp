#include "Client.hpp"
#include "NetworkManager.hpp"
#include <iostream>
#include <cstring>

Client::Client(const std::string& serverAddress, int serverPort)
    : serverAddress(serverAddress), serverPort(serverPort), socket_fd(-1) {}

bool Client::connectToServer() {
    socket_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (socket_fd < 0) {
        std::cerr << "Error creating socket." << std::endl;
        return false;
    }

    sockaddr_in serverAddr;
    std::memset(&serverAddr, 0, sizeof(serverAddr));
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(serverPort);
    inet_pton(AF_INET, serverAddress.c_str(), &serverAddr.sin_addr);

    if (connect(socket_fd, (struct sockaddr*)&serverAddr, sizeof(serverAddr)) < 0) {
        std::cerr << "Error connecting to server." << std::endl;
        close(socket_fd);
        return false;
    }

    return true;
}

void Client::sendMessage(const Message& message) {
    // Serialize and send the message
    // Implementation of serialization is assumed to be handled in the Message class
    send(socket_fd, message.serialize().c_str(), message.serialize().size(), 0);
}

Message Client::receiveMessage() {
    char buffer[1024];
    ssize_t bytesReceived = recv(socket_fd, buffer, sizeof(buffer) - 1, 0);
    if (bytesReceived > 0) {
        buffer[bytesReceived] = '\0';
        return Message::deserialize(buffer); // Assuming a deserialize method exists
    }
    return Message(); // Return an empty message on failure
}

Client::~Client() {
    if (socket_fd != -1) {
        close(socket_fd);
    }
}