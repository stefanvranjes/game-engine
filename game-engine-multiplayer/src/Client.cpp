#include "Client.hpp"
#include "NetworkManager.hpp"
#include <iostream>
#include <cstring>

#ifdef _WIN32
    #include <winsock2.h>
    #include <ws2tcpip.h>
    #define close closesocket
    typedef int ssize_t;
#else
    #include <sys/types.h>
    #include <sys/socket.h>
    #include <netinet/in.h>
    #include <arpa/inet.h>
    #include <unistd.h>
#endif

Client::Client(const std::string& serverAddress, int serverPort)
    : serverAddress(serverAddress), serverPort(serverPort), socketFD(-1) {}

bool Client::connectToServer() {
    socketFD = socket(AF_INET, SOCK_STREAM, 0);
    if (socketFD < 0) {
        std::cerr << "Error creating socket." << std::endl;
        return false;
    }

    sockaddr_in serverAddr;
    std::memset(&serverAddr, 0, sizeof(serverAddr));
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(serverPort);
    inet_pton(AF_INET, serverAddress.c_str(), &serverAddr.sin_addr);

    if (connect(socketFD, (struct sockaddr*)&serverAddr, sizeof(serverAddr)) < 0) {
        std::cerr << "Error connecting to server." << std::endl;
        close(socketFD);
        return false;
    }

    return true;
}

void Client::sendMessage(const Message& message) {
    auto data = message.serialize();
    send(socketFD, reinterpret_cast<const char*>(data.data()), data.size(), 0);
}

Message Client::receiveMessage() {
    char buffer[1024];
    ssize_t bytesReceived = recv(socketFD, buffer, sizeof(buffer) - 1, 0);
    if (bytesReceived > 0) {
        // Convert char buffer to vector<uint8_t> for deserialization
        std::vector<uint8_t> data(reinterpret_cast<uint8_t*>(buffer), 
                                   reinterpret_cast<uint8_t*>(buffer) + bytesReceived);
        return Message::deserialize(data);
    }
    return Message(); // Return an empty message on failure
}

Client::~Client() {
    if (socketFD != -1) {
        close(socketFD);
    }
}