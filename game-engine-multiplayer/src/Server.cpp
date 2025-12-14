#include "Server.hpp"
#include "NetworkManager.hpp"
#include "Message.hpp"
#include "Peer.hpp"
#include <iostream>
#include <vector>
#include <thread>
#include <asio.hpp>

using asio::ip::tcp;

class Server {
public:
    Server(asio::io_context& io_context, short port)
        : acceptor_(io_context, tcp::endpoint(tcp::v4(), port)) {
        start_accept();
    }

private:
    void start_accept() {
        Peer* new_peer = new Peer(acceptor_.get_io_context());
        acceptor_.async_accept(new_peer->get_socket(),
            std::bind(&Server::handle_accept, this, new_peer, std::placeholders::_1));
    }

    void handle_accept(Peer* new_peer, const asio::error_code& error) {
        if (!error) {
            peers_.push_back(new_peer);
            new_peer->start();
            std::cout << "New client connected!" << std::endl;
            start_accept();
        } else {
            delete new_peer;
        }
    }

    tcp::acceptor acceptor_;
    std::vector<Peer*> peers_;
};