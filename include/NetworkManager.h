#pragma once

#include <string>
#include <vector>
#include <functional>
#include <cstdint>
#include <memory>
#include <unordered_map>

/**
 * @brief Network manager for distributed physics simulation
 * 
 * Handles TCP connections for reliable batch transfer and UDP for
 * low-latency state updates across multiple nodes.
 */
class NetworkManager {
public:
    /**
     * @brief Message types for network communication
     */
    enum class MessageType : uint8_t {
        BATCH_ASSIGN = 0,      // Master → Worker: Assign batch of soft bodies
        BATCH_RESULT = 1,      // Worker → Master: Return simulation results
        STATE_SYNC_FULL = 2,   // Full soft body state synchronization
        STATE_SYNC_DELTA = 3,  // Delta update for soft body state
        HEARTBEAT = 4,         // Worker → Master: Health check
        LOAD_REPORT = 5,       // Worker → Master: Current load metrics
        MIGRATION = 6,         // Master → Worker: Move soft body between nodes
        NODE_REGISTER = 7,     // Worker → Master: Register as worker
        NODE_UNREGISTER = 8,   // Worker → Master: Unregister worker
        ACK = 9,               // Acknowledgment
        ERROR = 10,            // Error message
        VOTE_REQUEST = 11,     // Candidate → All: Request vote for election
        VOTE_RESPONSE = 12,    // All → Candidate: Vote response
        LEADER_ANNOUNCEMENT = 13, // New Leader → All: Announce leadership
        STATE_REQUEST = 14     // New Leader → Workers: Request state for reconstruction
    };
    
    /**
     * @brief Network message structure
     */
    struct Message {
        MessageType type;
        int sourceNode;
        int targetNode;
        uint64_t timestamp;
        uint32_t sequenceNumber;
        std::vector<uint8_t> data;
        
        // Reliability fields
        bool requiresAck;           // Does this message need acknowledgment?
        uint32_t ackSequenceNumber; // Sequence number being acknowledged
        uint32_t retryCount;        // Number of retries attempted
        uint64_t sentTimestamp;     // When message was sent (for timeout)
        
        Message() : type(MessageType::HEARTBEAT), sourceNode(-1), 
                   targetNode(-1), timestamp(0), sequenceNumber(0),
                   requiresAck(false), ackSequenceNumber(0), 
                   retryCount(0), sentTimestamp(0) {}
    };
    
    /**
     * @brief Node information
     */
    struct NodeInfo {
        int nodeId;
        std::string address;
        uint16_t port;
        bool isConnected;
        uint64_t lastHeartbeat;
        float currentLoad;
        size_t gpuCount;
        uint64_t totalMemoryMB;
    };
    
    /**
     * @brief Network statistics
     */
    struct NetworkStats {
        size_t messagesSent;
        size_t messagesReceived;
        size_t bytesSent;
        size_t bytesReceived;
        float avgLatencyMs;
        size_t droppedMessages;
        size_t activeConnections;
        
        // Reliability stats
        size_t acksReceived;
        size_t acksSent;
        size_t retriedMessages;
        size_t timedOutMessages;
        size_t duplicatesReceived;
        
        // Bandwidth optimization stats
        size_t messagesBatched;
        size_t batchesSent;
        size_t bytesBeforeCompression;
        size_t bytesAfterCompression;
        float compressionRatio;
        float currentBandwidthKBps;
    };
    
    /**
     * @brief Retry strategy for failed messages
     */
    enum class RetryStrategy {
        FIXED_DELAY,        // Fixed delay between retries
        EXPONENTIAL_BACKOFF, // Exponentially increasing delay
        LINEAR_BACKOFF      // Linearly increasing delay
    };
    
    /**
     * @brief Reliability configuration
     */
    struct ReliabilityConfig {
        bool enableAcks;              // Enable acknowledgments
        uint32_t ackTimeoutMs;        // Timeout for ACK (default: 1000ms)
        uint32_t maxRetries;          // Max retry attempts (default: 3)
        uint32_t retryDelayMs;        // Base delay between retries (default: 500ms)
        bool enableDuplicateDetection; // Track and ignore duplicates
        
        // Advanced retry options
        RetryStrategy retryStrategy;   // Retry strategy (default: EXPONENTIAL_BACKOFF)
        float backoffMultiplier;       // Multiplier for backoff (default: 2.0)
        uint32_t maxRetryDelayMs;      // Max delay cap (default: 10000ms)
        bool enableJitter;             // Add random jitter to prevent thundering herd
        float jitterFactor;            // Jitter randomness (0-1, default: 0.1)
        
        // Bandwidth optimization
        bool enableMessageBatching;    // Batch multiple messages together
        uint32_t maxBatchSize;         // Max messages per batch (default: 10)
        uint32_t batchTimeoutMs;       // Max time to wait for batch (default: 10ms)
        bool enableHeaderCompression;  // Compress message headers
        bool adaptiveCompression;      // Use compression based on message size
        uint32_t compressionThreshold; // Min size for compression (default: 1024 bytes)
        uint32_t maxBandwidthKBps;     // Max bandwidth in KB/s (0 = unlimited)
        
        ReliabilityConfig() : enableAcks(true), ackTimeoutMs(1000),
                             maxRetries(3), retryDelayMs(500),
                             enableDuplicateDetection(true),
                             retryStrategy(RetryStrategy::EXPONENTIAL_BACKOFF),
                             backoffMultiplier(2.0f), maxRetryDelayMs(10000),
                             enableJitter(true), jitterFactor(0.1f),
                             enableMessageBatching(true), maxBatchSize(10),
                             batchTimeoutMs(10), enableHeaderCompression(true),
                             adaptiveCompression(true), compressionThreshold(1024),
                             maxBandwidthKBps(0) {}
    };
    
    NetworkManager();
    ~NetworkManager();
    
    // Connection management
    
    /**
     * @brief Connect to master node as a worker
     * @param address Master node address
     * @param port Master node port
     * @return True if connection successful
     */
    bool ConnectToMaster(const std::string& address, uint16_t port);
    
    /**
     * @brief Start master server
     * @param port Port to listen on
     * @return True if server started successfully
     */
    bool StartMasterServer(uint16_t port);
    
    /**
     * @brief Disconnect from a node
     * @param nodeId Node ID to disconnect
     */
    void DisconnectNode(int nodeId);
    
    /**
     * @brief Shutdown all connections
     */
    void Shutdown();
    
    // Message handling
    
    /**
     * @brief Send message to specific node
     * @param nodeId Target node ID
     * @param msg Message to send
     * @param reliable Use reliable delivery (ACKs + retries)
     * @return True if sent successfully
     */
    bool SendMessage(int nodeId, const Message& msg, bool reliable = true);
    
    /**
     * @brief Configure reliability settings
     * @param config Reliability configuration
     */
    void SetReliabilityConfig(const ReliabilityConfig& config);
    
    /**
     * @brief Get current reliability configuration
     * @return Reliability configuration
     */
    ReliabilityConfig GetReliabilityConfig() const;
    
    /**
     * @brief Broadcast message to all connected nodes
     * @param msg Message to broadcast
     */
    void BroadcastMessage(const Message& msg);
    
    /**
     * @brief Send message asynchronously
     * @param nodeId Target node ID
     * @param msg Message to send
     * @param callback Callback on completion (success/failure)
     */
    void SendMessageAsync(int nodeId, const Message& msg, 
                         std::function<void(bool)> callback);
    
    /**
     * @brief Receive message from specific node (blocking)
     * @param nodeId Source node ID
     * @param timeoutMs Timeout in milliseconds (0 = no timeout)
     * @return Received message (empty if timeout/error)
     */
    Message ReceiveMessage(int nodeId, uint32_t timeoutMs = 0);
    
    /**
     * @brief Check if messages are available
     * @param nodeId Source node ID
     * @return True if messages available
     */
    bool HasMessages(int nodeId) const;
    
    /**
     * @brief Set message received callback
     * @param callback Function called when message received
     */
    void SetMessageCallback(std::function<void(int nodeId, const Message&)> callback);
    
    // Node management
    
    /**
     * @brief Get list of connected nodes
     * @return Vector of node information
     */
    std::vector<NodeInfo> GetConnectedNodes() const;
    
    /**
     * @brief Get node information
     * @param nodeId Node ID
     * @return Node information (empty if not found)
     */
    NodeInfo GetNodeInfo(int nodeId) const;
    
    /**
     * @brief Check if node is connected
     * @param nodeId Node ID
     * @return True if connected
     */
    bool IsNodeConnected(int nodeId) const;
    
    // Statistics
    
    /**
     * @brief Get network statistics
     * @return Network statistics
     */
    NetworkStats GetStatistics() const;
    
    /**
     * @brief Reset statistics
     */
    void ResetStatistics();
    
    /**
     * @brief Get local node ID
     * @return Local node ID (-1 if not connected)
     */
    int GetLocalNodeId() const;
    
    /**
     * @brief Check if running as master
     * @return True if master node
     */
    bool IsMaster() const;

private:
    struct Impl;
    std::unique_ptr<Impl> m_Impl;
    
    // Internal helpers
    void ProcessIncomingMessages();
    void UpdateHeartbeats();
    void CleanupDeadConnections();
};
