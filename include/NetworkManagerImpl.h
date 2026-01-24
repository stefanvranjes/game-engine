#pragma once

#include "NetworkManager.h"
#include <iostream>
#include <chrono>
#include <thread>
#include <mutex>
#include <queue>
#include <atomic>
#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <random>
#include <array>

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

// PendingAck struct from NetworkManagerReliability.cpp
struct PendingAck {
    NetworkManager::Message message;
    uint64_t sentTime;
    uint32_t retryCount;
    int targetNode;
    uint64_t nextRetryTime;  // When to retry next
};

struct MessageBatch {
    std::vector<NetworkManager::Message> messages;
    uint64_t firstMessageTime;
    size_t totalSize;
};

struct NetworkManager::Impl {
    // --- COMMON MEMBERS ---
    // Message handling
    std::function<void(int, const Message&)> messageCallback;
    std::atomic<uint32_t> nextSequenceNumber{0};
    
    // Statistics
    NetworkStats stats;
    std::mutex statsMutex;
    
    // Worker threads (Common control)
    std::atomic<bool> running{false};

    // Reliability tracking
    ReliabilityConfig reliabilityConfig;
    std::unordered_map<uint32_t, PendingAck> pendingAcks;  // seqNum -> pending message
    std::unordered_set<uint32_t> receivedSequenceNumbers;   // For duplicate detection
    std::mutex ackMutex;
    std::thread ackThread;
    std::mt19937 rng;  // Random number generator for jitter

    // Bandwidth optimization
    std::unordered_map<int, MessageBatch> pendingBatches;  // nodeId -> batch
    std::mutex batchMutex;
    std::thread batchThread;
    
    // Bandwidth throttling
    uint64_t lastSendTime = 0;
    size_t bytesThisSecond = 0;
    std::mutex throttleMutex;

#ifdef HAS_ASIO
    // ASIO-based implementation
    asio::io_context ioContext;
    std::unique_ptr<tcp::acceptor> acceptor;
    std::thread ioThread;
    
    // Connection state
    bool isMaster = false;
    int localNodeId = -1;
    int nextNodeId = 1;
    
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
    
    std::unordered_map<int, std::unique_ptr<Connection>> connections;
    std::mutex connectionsMutex;

    Impl() {
        stats = {};
        #ifdef _WIN32
        WSACleanup(); // Is this needed for ASIO? Probably safer to have standard init if we mix.
                      // But ASIO handles winsock init usually.
        #endif
    }

    ~Impl() {
        running = false;
        if (ackThread.joinable()) ackThread.join();
        if (batchThread.joinable()) batchThread.join();
        if (ioThread.joinable()) ioThread.join();
        // Socket cleanup handled by RAII (shared_ptr<socket>, unique_ptr<acceptor>)
    }
    
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
    
    std::unordered_map<int, std::unique_ptr<Connection>> connections;
    std::mutex connectionsMutex;
    
    std::thread acceptThread;
    std::thread receiveThread;
    std::thread heartbeatThread;

    Impl() {
        stats = {}; // Initialize stats
        #ifdef _WIN32
        WSACleanup(); // Clean previous if any? Or mostly init happens in global?
        // Actually, NetworkManager likely calls WSAStartup in Initialize.
        #endif
    }

    ~Impl() {
        running = false;
        
        if (acceptThread.joinable()) acceptThread.join();
        if (receiveThread.joinable()) receiveThread.join();
        if (heartbeatThread.joinable()) heartbeatThread.join();
        if (ackThread.joinable()) ackThread.join();
        if (batchThread.joinable()) batchThread.join();
        
        // Close all sockets
        for (auto& [id, conn] : connections) {
            if (conn && conn->socket != INVALID_SOCKET_VALUE) {
                closesocket(conn->socket);
            }
        }
        
        if (serverSocket != INVALID_SOCKET_VALUE) {
            closesocket(serverSocket);
        }
        
        #ifdef _WIN32
        WSACleanup();
        #endif
    }
#endif

    // Methods
    bool ShouldCompress(const std::vector<uint8_t>& data);
    std::vector<uint8_t> CompressMessageHeader(const Message& msg);
    void ProcessBatches();
    void SendBatch(int nodeId, const MessageBatch& batch);
    void ThrottleBandwidth(size_t bytes);

    // ... reliability methods ...
    
    // From NetworkManager.cpp (renamed from SendMessage)
    bool SendMessageInternal(int nodeId, const Message& msg);
    void DisconnectNode(int nodeId);

    // From NetworkManagerReliability.cpp
    uint32_t CalculateRetryDelay(uint32_t retryCount);
    void ProcessAcknowledgments();
    void HandleAck(uint32_t ackSeqNum);
    bool IsDuplicate(uint32_t seqNum);
    void SendAck(int nodeId, uint32_t seqNum);
    
#ifdef HAS_ASIO
    // ASIO-specific methods (defined in NetworkManagerASIO.cpp)
    void StartAsyncAccept();
    void StartAsyncRead(int nodeId);
    void ProcessReceivedData(int nodeId, size_t bytes);
#endif
};
