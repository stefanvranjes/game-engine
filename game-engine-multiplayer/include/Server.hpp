#ifndef SERVER_HPP
#define SERVER_HPP

#include <vector>
#include <memory>
#include "Peer.hpp"
#include "Message.hpp"

class Server {
public:
    Server(int port);
    void start();
    void stop();
    void broadcastMessage(const Message& message);
    
private:
    void acceptConnections();
    void handleClient(std::shared_ptr<Peer> client);
    
    int port;
    std::vector<std::shared_ptr<Peer>> clients;
    bool running;
};

#endif // SERVER_HPP