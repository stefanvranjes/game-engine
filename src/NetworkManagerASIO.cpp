// ASIO async networking implementation for NetworkManager
// This file contains the production ASIO implementation

#ifdef HAS_ASIO

#include "NetworkManager.h"
#include <iostream>

// ASIO-specific connection implementation
void NetworkManager::Impl::StartAsyncAccept() {
    if (!acceptor) return;
    
    auto socket = std::make_shared<tcp::socket>(ioContext);
    
    acceptor->async_accept(*socket, [this, socket](const asio::error_code& error) {
        if (!error) {
            // Get remote endpoint info
            auto endpoint = socket->remote_endpoint();
            std::string address = endpoint.address().to_string();
            uint16_t port = endpoint.port();
            
            std::lock_guard<std::mutex> lock(connectionsMutex);
            
            Connection conn;
            conn.nodeId = nextNodeId++;
            conn.socket = socket;
            conn.address = address;
            conn.port = port;
            conn.lastHeartbeat = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            
            connections[conn.nodeId] = std::move(conn);
            
            std::cout << "Worker node " << conn.nodeId << " connected from " 
                      << address << ":" << port << std::endl;
            
            // Start async read for this connection
            StartAsyncRead(conn.nodeId);
        }
        
        // Continue accepting
        if (running) {
            StartAsyncAccept();
        }
    });
}

void NetworkManager::Impl::StartAsyncRead(int nodeId) {
    std::lock_guard<std::mutex> lock(connectionsMutex);
    
    auto it = connections.find(nodeId);
    if (it == connections.end()) return;
    
    auto& conn = it->second;
    
    conn.socket->async_read_some(
        asio::buffer(conn.readBuffer),
        [this, nodeId](const asio::error_code& error, std::size_t bytes_transferred) {
            if (!error && bytes_transferred > 0) {
                // Process received data
                ProcessReceivedData(nodeId, bytes_transferred);
                
                // Continue reading
                if (running) {
                    StartAsyncRead(nodeId);
                }
            } else {
                // Connection error or closed
                std::cout << "Connection error for node " << nodeId << ": " 
                          << error.message() << std::endl;
                DisconnectNode(nodeId);
            }
        }
    );
}

void NetworkManager::Impl::ProcessReceivedData(int nodeId, size_t bytes) {
    std::lock_guard<std::mutex> lock(connectionsMutex);
    
    auto it = connections.find(nodeId);
    if (it == connections.end()) return;
    
    // TODO: Parse message from readBuffer
    // For now, just update heartbeat
    it->second.lastHeartbeat = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    
    stats.messagesReceived++;
    stats.bytesReceived += bytes;
}

bool NetworkManager::ConnectToMasterASIO(const std::string& address, uint16_t port) {
    try {
        auto socket = std::make_shared<tcp::socket>(m_Impl->ioContext);
        
        tcp::resolver resolver(m_Impl->ioContext);
        auto endpoints = resolver.resolve(address, std::to_string(port));
        
        asio::connect(*socket, endpoints);
        
        // Add master connection
        std::lock_guard<std::mutex> lock(m_Impl->connectionsMutex);
        Impl::Connection conn;
        conn.nodeId = 0;  // Master is always node 0
        conn.socket = socket;
        conn.address = address;
        conn.port = port;
        conn.lastHeartbeat = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        
        m_Impl->connections[0] = std::move(conn);
        m_Impl->localNodeId = 1;
        m_Impl->isMaster = false;
        
        // Start IO thread
        m_Impl->running = true;
        m_Impl->ioThread = std::thread([this]() {
            m_Impl->ioContext.run();
        });
        
        // Start async read
        m_Impl->StartAsyncRead(0);
        
        std::cout << "Connected to master at " << address << ":" << port << std::endl;
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to connect to master: " << e.what() << std::endl;
        return false;
    }
}

bool NetworkManager::StartMasterServerASIO(uint16_t port) {
    try {
        tcp::endpoint endpoint(tcp::v4(), port);
        m_Impl->acceptor = std::make_unique<tcp::acceptor>(m_Impl->ioContext, endpoint);
        
        m_Impl->localNodeId = 0;
        m_Impl->isMaster = true;
        m_Impl->running = true;
        m_Impl->nextNodeId = 1;
        
        // Start accepting connections
        m_Impl->StartAsyncAccept();
        
        // Start IO thread
        m_Impl->ioThread = std::thread([this]() {
            m_Impl->ioContext.run();
        });
        
        std::cout << "Master server started on port " << port << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to start master server: " << e.what() << std::endl;
        return false;
    }
}

bool NetworkManager::SendMessageASIO(int nodeId, const Message& msg) {
    std::lock_guard<std::mutex> lock(m_Impl->connectionsMutex);
    
    auto it = m_Impl->connections.find(nodeId);
    if (it == m_Impl->connections.end()) {
        return false;
    }
    
    // Serialize message (simplified)
    std::vector<uint8_t> buffer;
    buffer.push_back(static_cast<uint8_t>(msg.type));
    // TODO: Add full serialization
    
    try {
        asio::write(*it->second.socket, asio::buffer(buffer));
        
        std::lock_guard<std::mutex> statsLock(m_Impl->statsMutex);
        m_Impl->stats.messagesSent++;
        m_Impl->stats.bytesSent += buffer.size();
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to send message to node " << nodeId 
                  << ": " << e.what() << std::endl;
        return false;
    }
}

void NetworkManager::SendMessageAsyncASIO(int nodeId, const Message& msg,
                                         std::function<void(bool)> callback) {
    std::lock_guard<std::mutex> lock(m_Impl->connectionsMutex);
    
    auto it = m_Impl->connections.find(nodeId);
    if (it == m_Impl->connections.end()) {
        if (callback) callback(false);
        return;
    }
    
    // Serialize message
    auto buffer = std::make_shared<std::vector<uint8_t>>();
    buffer->push_back(static_cast<uint8_t>(msg.type));
    // TODO: Add full serialization
    
    asio::async_write(*it->second.socket, asio::buffer(*buffer),
        [this, buffer, callback](const asio::error_code& error, std::size_t bytes_transferred) {
            if (!error) {
                std::lock_guard<std::mutex> statsLock(m_Impl->statsMutex);
                m_Impl->stats.messagesSent++;
                m_Impl->stats.bytesSent += bytes_transferred;
            }
            
            if (callback) {
                callback(!error);
            }
        }
    );
}

#endif // HAS_ASIO
