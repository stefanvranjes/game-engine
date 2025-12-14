#include <iostream>
#include "NetworkManager.hpp"

int main() {
    NetworkManager networkManager;

    // Start the server
    if (!networkManager.startServer(8080)) {
        std::cerr << "Failed to start the server." << std::endl;
        return 1;
    }

    std::cout << "Server is running on port 8080." << std::endl;

    // Main loop to keep the server running
    while (true) {
        networkManager.update();
    }

    return 0;
}