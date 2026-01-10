#include "NetworkManager.h"
#include <iostream>
#include <chrono>
#include <thread>
#include <mutex>
#include <queue>
#include <atomic>

#ifdef HAS_ASIO
    #ifdef ASIO_STANDALONE
        #include <asio.hpp>
    #else
        #include <boost/asio.hpp>
        namespace asio = boost::asio;
    #endif
    using asio::ip::tcp;
#else
    // Platform-specific includes for fallback
    #ifdef _WIN32
        #include <winsock2.h>
        #include <ws2tcpip.h>
        #pragma comment(lib, "ws2_32.lib")
        typedef SOCKET SocketType;
        #define INVALID_SOCKET_VALUE INVALID_SOCKET
        #define SOCKET_ERROR_VALUE SOCKET_ERROR
    #else
        #include <sys/socket.h>
        #include <netinet/in.h>
        #include <arpa/inet.h>
        #include <unistd.h>
        #include <fcntl.h>
        typedef int SocketType;
        #define INVALID_SOCKET_VALUE -1
        #define SOCKET_ERROR_VALUE -1
        #define closesocket close
    #endif
#endif

struct NetworkManager::Impl {
#ifdef HAS_ASIO
    // ASIO-based implementation
    asio::io_context ioContext;
    std::unique_ptr<tcp::acceptor> acceptor;
    std::thread ioThread;
    
    struct Connection {
        int nodeId;
        std::shared_ptr<tcp::socket> socket;
        std::string address;
        uint16_t port;
        std::queue<Message> messageQueue;
        std::mutex queueMutex;
        uint64_t lastHeartbeat;
        float currentLoad;
        size_t gpuCount;
        uint64_t totalMemoryMB;
        std::array<uint8_t, 8192> readBuffer;  // Read buffer for async operations
    };
    
#else
    // Connection state
    bool isMaster = false;
    int localNodeId = -1;
    SocketType serverSocket = INVALID_SOCKET_VALUE;
    
    // Node connections
    struct Connection {
        int nodeId;
        SocketType socket;
        std::string address;
        uint16_t port;
        std::queue<Message> messageQueue;
        std::mutex queueMutex;
        uint64_t lastHeartbeat;
        float currentLoad;
        size_t gpuCount;
        uint64_t totalMemoryMB;
    };
    
    std::unordered_map<int, Connection> connections;
    std::mutex connectionsMutex;
    
    // Message handling
    std::function<void(int, const Message&)> messageCallback;
    std::atomic<uint32_t> nextSequenceNumber{0};
    
    // Statistics
    NetworkStats stats;
    std::mutex statsMutex;
    
    // Worker threads
    std::atomic<bool> running{false};
    std::thread acceptThread;
    std::thread receiveThread;
    std::thread heartbeatThread;
    
    ~Impl() {
        running = false;
        
        if (acceptThread.joinable()) acceptThread.join();
        if (receiveThread.joinable()) receiveThread.join();
        if (heartbeatThread.joinable()) heartbeatThread.join();
        
        // Close all sockets
        for (auto& [id, conn] : connections) {
            if (conn.socket != INVALID_SOCKET_VALUE) {
                closesocket(conn.socket);
            }
        }
        
        if (serverSocket != INVALID_SOCKET_VALUE) {
            closesocket(serverSocket);
        }
        
#ifdef _WIN32
        WSACleanup();
#endif
    }
};

NetworkManager::NetworkManager() : m_Impl(std::make_unique<Impl>()) {
#ifdef _WIN32
    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        std::cerr << "WSAStartup failed" << std::endl;
    }
#endif
}

NetworkManager::~NetworkManager() {
    Shutdown();
}

bool NetworkManager::ConnectToMaster(const std::string& address, uint16_t port) {
#ifdef HAS_ASIO
    return ConnectToMasterASIO(address, port);
#else
    SocketType sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (sock == INVALID_SOCKET_VALUE) {
        std::cerr << "Failed to create socket" << std::endl;
        return false;
    }
    
    sockaddr_in serverAddr{};
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(port);
    inet_pton(AF_INET, address.c_str(), &serverAddr.sin_addr);
    
    if (connect(sock, (sockaddr*)&serverAddr, sizeof(serverAddr)) == SOCKET_ERROR_VALUE) {
        std::cerr << "Failed to connect to master at " << address << ":" << port << std::endl;
        closesocket(sock);
        return false;
    }
    
    // Add master connection
    std::lock_guard<std::mutex> lock(m_Impl->connectionsMutex);
    Impl::Connection conn;
    conn.nodeId = 0;  // Master is always node 0
    conn.socket = sock;
    conn.address = address;
    conn.port = port;
    conn.lastHeartbeat = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    
    m_Impl->connections[0] = std::move(conn);
    m_Impl->localNodeId = 1;  // Worker nodes start at 1
    m_Impl->isMaster = false;
    
    // Start worker threads
    m_Impl->running = true;
    m_Impl->receiveThread = std::thread([this]() { ProcessIncomingMessages(); });
    m_Impl->heartbeatThread = std::thread([this]() { UpdateHeartbeats(); });
    
    std::cout << "Connected to master at " << address << ":" << port << std::endl;
    
    // Send registration message
    Message registerMsg;
    registerMsg.type = MessageType::NODE_REGISTER;
    registerMsg.sourceNode = m_Impl->localNodeId;
    registerMsg.targetNode = 0;
    SendMessage(0, registerMsg);
    
    return true;
}

bool NetworkManager::StartMasterServer(uint16_t port) {
#ifdef HAS_ASIO
    return StartMasterServerASIO(port);
#else
    m_Impl->serverSocket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (m_Impl->serverSocket == INVALID_SOCKET_VALUE) {
        std::cerr << "Failed to create server socket" << std::endl;
        return false;
    }
    
    // Set socket options
    int opt = 1;
    setsockopt(m_Impl->serverSocket, SOL_SOCKET, SO_REUSEADDR, (char*)&opt, sizeof(opt));
    
    sockaddr_in serverAddr{};
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_addr.s_addr = INADDR_ANY;
    serverAddr.sin_port = htons(port);
    
    if (bind(m_Impl->serverSocket, (sockaddr*)&serverAddr, sizeof(serverAddr)) == SOCKET_ERROR_VALUE) {
        std::cerr << "Failed to bind to port " << port << std::endl;
        closesocket(m_Impl->serverSocket);
        m_Impl->serverSocket = INVALID_SOCKET_VALUE;
        return false;
    }
    
    if (listen(m_Impl->serverSocket, 10) == SOCKET_ERROR_VALUE) {
        std::cerr << "Failed to listen on port " << port << std::endl;
        closesocket(m_Impl->serverSocket);
        m_Impl->serverSocket = INVALID_SOCKET_VALUE;
        return false;
    }
    
    m_Impl->localNodeId = 0;  // Master is node 0
    m_Impl->isMaster = true;
    m_Impl->running = true;
    
    // Start accept thread
    m_Impl->acceptThread = std::thread([this]() {
        int nextNodeId = 1;
        
        while (m_Impl->running) {
            sockaddr_in clientAddr{};
            socklen_t clientLen = sizeof(clientAddr);
            
            SocketType clientSocket = accept(m_Impl->serverSocket, 
                                            (sockaddr*)&clientAddr, &clientLen);
            
            if (clientSocket == INVALID_SOCKET_VALUE) {
                if (m_Impl->running) {
                    std::cerr << "Accept failed" << std::endl;
                }
                continue;
            }
            
            char clientIp[INET_ADDRSTRLEN];
            inet_ntop(AF_INET, &clientAddr.sin_addr, clientIp, INET_ADDRSTRLEN);
            
            std::lock_guard<std::mutex> lock(m_Impl->connectionsMutex);
            Impl::Connection conn;
            conn.nodeId = nextNodeId++;
            conn.socket = clientSocket;
            conn.address = clientIp;
            conn.port = ntohs(clientAddr.sin_port);
            conn.lastHeartbeat = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            
            m_Impl->connections[conn.nodeId] = std::move(conn);
            
            std::cout << "Worker node " << conn.nodeId << " connected from " 
                      << clientIp << ":" << conn.port << std::endl;
        }
    });
    
    // Start receive and heartbeat threads
    m_Impl->receiveThread = std::thread([this]() { ProcessIncomingMessages(); });
    m_Impl->heartbeatThread = std::thread([this]() { UpdateHeartbeats(); });
    
    std::cout << "Master server started on port " << port << std::endl;
    return true;
}

void NetworkManager::DisconnectNode(int nodeId) {
    std::lock_guard<std::mutex> lock(m_Impl->connectionsMutex);
    
    auto it = m_Impl->connections.find(nodeId);
    if (it != m_Impl->connections.end()) {
        closesocket(it->second.socket);
        m_Impl->connections.erase(it);
        std::cout << "Disconnected from node " << nodeId << std::endl;
    }
}

void NetworkManager::Shutdown() {
    m_Impl->running = false;
    
    // Close server socket to unblock accept
    if (m_Impl->serverSocket != INVALID_SOCKET_VALUE) {
        closesocket(m_Impl->serverSocket);
        m_Impl->serverSocket = INVALID_SOCKET_VALUE;
    }
    
    // Wait for threads
    if (m_Impl->acceptThread.joinable()) m_Impl->acceptThread.join();
    if (m_Impl->receiveThread.joinable()) m_Impl->receiveThread.join();
    if (m_Impl->heartbeatThread.joinable()) m_Impl->heartbeatThread.join();
    
    // Close all connections
    std::lock_guard<std::mutex> lock(m_Impl->connectionsMutex);
    for (auto& [id, conn] : m_Impl->connections) {
        closesocket(conn.socket);
    }
    m_Impl->connections.clear();
    
    std::cout << "Network manager shutdown complete" << std::endl;
}

bool NetworkManager::SendMessage(int nodeId, const Message& msg) {
    std::lock_guard<std::mutex> lock(m_Impl->connectionsMutex);
    
    auto it = m_Impl->connections.find(nodeId);
    if (it == m_Impl->connections.end()) {
        return false;
    }
    
    // Serialize message (simplified - would use proper serialization in production)
    std::vector<uint8_t> buffer;
    buffer.push_back(static_cast<uint8_t>(msg.type));
    // TODO: Add full serialization
    
    int sent = send(it->second.socket, (char*)buffer.data(), buffer.size(), 0);
    
    if (sent == SOCKET_ERROR_VALUE) {
        std::cerr << "Failed to send message to node " << nodeId << std::endl;
        return false;
    }
    
    std::lock_guard<std::mutex> statsLock(m_Impl->statsMutex);
    m_Impl->stats.messagesSent++;
    m_Impl->stats.bytesSent += sent;
    
    return true;
}

void NetworkManager::BroadcastMessage(const Message& msg) {
    std::lock_guard<std::mutex> lock(m_Impl->connectionsMutex);
    
    for (const auto& [nodeId, conn] : m_Impl->connections) {
        SendMessage(nodeId, msg);
    }
}

NetworkManager::Message NetworkManager::ReceiveMessage(int nodeId, uint32_t timeoutMs) {
    // Simplified implementation - would use proper async I/O in production
    Message msg;
    return msg;
}

std::vector<NetworkManager::NodeInfo> NetworkManager::GetConnectedNodes() const {
    std::lock_guard<std::mutex> lock(m_Impl->connectionsMutex);
    
    std::vector<NodeInfo> nodes;
    for (const auto& [id, conn] : m_Impl->connections) {
        NodeInfo info;
        info.nodeId = id;
        info.address = conn.address;
        info.port = conn.port;
        info.isConnected = true;
        info.lastHeartbeat = conn.lastHeartbeat;
        info.currentLoad = conn.currentLoad;
        info.gpuCount = conn.gpuCount;
        info.totalMemoryMB = conn.totalMemoryMB;
        nodes.push_back(info);
    }
    
    return nodes;
}

NetworkManager::NetworkStats NetworkManager::GetStatistics() const {
    std::lock_guard<std::mutex> lock(m_Impl->statsMutex);
    return m_Impl->stats;
}

int NetworkManager::GetLocalNodeId() const {
    return m_Impl->localNodeId;
}

bool NetworkManager::IsMaster() const {
    return m_Impl->isMaster;
}

void NetworkManager::ProcessIncomingMessages() {
    // Simplified - would implement proper async message processing
    while (m_Impl->running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

void NetworkManager::UpdateHeartbeats() {
    while (m_Impl->running) {
        if (!m_Impl->isMaster) {
            // Workers send heartbeats to master
            Message heartbeat;
            heartbeat.type = MessageType::HEARTBEAT;
            heartbeat.sourceNode = m_Impl->localNodeId;
            heartbeat.targetNode = 0;
            SendMessage(0, heartbeat);
        }
        
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

void NetworkManager::CleanupDeadConnections() {
    // Check for dead connections based on heartbeat timeout
    const uint64_t HEARTBEAT_TIMEOUT_MS = 5000;
    
    auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    
    std::lock_guard<std::mutex> lock(m_Impl->connectionsMutex);
    
    for (auto it = m_Impl->connections.begin(); it != m_Impl->connections.end();) {
        if (now - it->second.lastHeartbeat > HEARTBEAT_TIMEOUT_MS) {
            std::cout << "Node " << it->first << " timed out, disconnecting" << std::endl;
            closesocket(it->second.socket);
            it = m_Impl->connections.erase(it);
        } else {
            ++it;
        }
    }
}
