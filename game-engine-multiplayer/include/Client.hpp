#ifndef CLIENT_HPP
#define CLIENT_HPP

#include <string>
#include <iostream>
#include "Message.hpp"

class Client {
public:
    Client(const std::string& serverAddress, int serverPort);
    ~Client();

    bool connectToServer();
    void sendMessage(const Message& message);
    Message receiveMessage();

private:
    std::string serverAddress;
    int serverPort;
    int socketFD; // File descriptor for the socket
};

#endif // CLIENT_HPP