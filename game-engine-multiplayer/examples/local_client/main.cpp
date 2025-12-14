#include <iostream>
#include <memory>
#include "Client.hpp"

int main() {
    std::string serverAddress;
    int serverPort;

    std::cout << "Enter server address: ";
    std::cin >> serverAddress;
    std::cout << "Enter server port: ";
    std::cin >> serverPort;

    auto client = std::make_unique<Client>(serverAddress, serverPort);

    if (!client->connectToServer()) {
        std::cerr << "Failed to connect to server." << std::endl;
        return 1;
    }

    std::cout << "Connected to server. You can start sending messages." << std::endl;

    std::string message;
    while (true) {
        std::cout << "Enter message (type 'exit' to quit): ";
        std::getline(std::cin, message);
        if (message == "exit") {
            break;
        }
        client->sendMessage(message);
    }

    client->disconnect();
    return 0;
}