#include <gtest/gtest.h>
#include "NetworkManager.hpp"
#include "Server.hpp"
#include "Client.hpp"
#include "Message.hpp"
#include "Peer.hpp"

class NetworkTests : public ::testing::Test {
protected:
    NetworkManager networkManager;
    Server server;
    Client client;

    void SetUp() override {
        // Initialize server and client for testing
        server.start(12345); // Start server on port 12345
        client.connect("localhost", 12345); // Connect client to server
    }

    void TearDown() override {
        // Clean up after tests
        client.disconnect();
        server.stop();
    }
};

TEST_F(NetworkTests, ServerAcceptsClientConnection) {
    ASSERT_TRUE(server.hasClient(client.getPeerId()));
}

TEST_F(NetworkTests, ClientSendsMessage) {
    Message message("Hello, Server!");
    client.sendMessage(message);
    
    ASSERT_TRUE(server.hasReceivedMessage(message));
}

TEST_F(NetworkTests, ServerBroadcastsMessage) {
    Message message("Broadcast Message");
    server.broadcastMessage(message);
    
    ASSERT_TRUE(client.hasReceivedMessage(message));
}

TEST_F(NetworkTests, PeerManagement) {
    Peer peer = client.getPeer();
    ASSERT_EQ(peer.getId(), client.getPeerId());
    ASSERT_TRUE(peer.isConnected());
}