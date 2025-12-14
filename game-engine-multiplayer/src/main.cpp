#include <iostream>
#include "NetworkManager.hpp"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <server|client>" << std::endl;
        return 1;
    }

    std::string mode = argv[1];
    NetworkManager networkManager;

    if (mode == "server") {
        networkManager.startServer();
    } else if (mode == "client") {
        networkManager.startClient();
    } else {
        std::cerr << "Invalid mode. Use 'server' or 'client'." << std::endl;
        return 1;
    }

    return 0;
}