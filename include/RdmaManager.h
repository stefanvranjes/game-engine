#pragma once

#ifdef HAS_RDMA
#include <infiniband/verbs.h>
#include <rdma/rdma_cma.h>
#endif

#include <memory>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <string>

/**
 * @brief RDMA Manager for GPU-Direct RDMA operations
 * 
 * Enables direct GPU-to-GPU data transfers across network nodes,
 * bypassing CPU and system memory for ultra-low-latency communication.
 * 
 * Requirements:
 * - NVIDIA GPUs with GPUDirect RDMA support (Kepler or newer)
 * - InfiniBand or RoCE network
 * - RDMA-capable NICs (Mellanox ConnectX-5 or newer)
 * - OFED drivers installed
 */
class RdmaManager {
public:
    /**
     * @brief RDMA context for registered GPU memory
     */
    struct RdmaContext {
        void* gpuBuffer;           // GPU memory pointer
        size_t bufferSize;         // Size in bytes
        uint32_t lkey;             // Local key for RDMA
        uint32_t rkey;             // Remote key for RDMA
        uint64_t remoteAddr;       // Remote GPU address
    };
    
    /**
     * @brief RDMA connection to remote node
     */
    struct RdmaConnection {
        int nodeId;
        bool isConnected;
        uint64_t bytesTransferred;
        uint64_t operationsCompleted;
    };
    
    /**
     * @brief RDMA statistics
     */
    struct RdmaStats {
        uint64_t totalWrites;
        uint64_t totalReads;
        uint64_t totalBytesTransferred;
        uint64_t failedOperations;
        double avgLatencyUs;
        double avgBandwidthMBps;
    };
    
    RdmaManager();
    ~RdmaManager();
    
    // Initialization
    
    /**
     * @brief Initialize RDMA subsystem
     * @return True if RDMA is available and initialized
     */
    bool Initialize();
    
    /**
     * @brief Check if RDMA is available on this system
     * @return True if RDMA hardware and drivers are present
     */
    bool IsRdmaAvailable() const;
    
    /**
     * @brief Shutdown RDMA subsystem
     */
    void Shutdown();
    
    // Connection management
    
    /**
     * @brief Connect to remote node via RDMA
     * @param nodeId Node identifier
     * @param address Remote node IP address
     * @param port Remote node port
     * @return True if connection successful
     */
    bool ConnectToNode(int nodeId, const std::string& address, uint16_t port);
    
    /**
     * @brief Disconnect from remote node
     * @param nodeId Node identifier
     */
    void DisconnectNode(int nodeId);
    
    /**
     * @brief Check if connected to node
     * @param nodeId Node identifier
     * @return True if connected
     */
    bool IsConnected(int nodeId) const;
    
    // Memory registration
    
    /**
     * @brief Register GPU memory for RDMA operations
     * @param gpuPtr GPU memory pointer (from cudaMalloc)
     * @param size Size in bytes
     * @param lkey Output: local key
     * @param rkey Output: remote key
     * @return True if registration successful
     */
    bool RegisterGpuMemory(void* gpuPtr, size_t size, uint32_t& lkey, uint32_t& rkey);
    
    /**
     * @brief Unregister GPU memory
     * @param gpuPtr GPU memory pointer
     */
    void UnregisterGpuMemory(void* gpuPtr);
    
    /**
     * @brief Get RDMA context for registered memory
     * @param gpuPtr GPU memory pointer
     * @return RDMA context or nullptr if not registered
     */
    const RdmaContext* GetRdmaContext(void* gpuPtr) const;
    
    // RDMA operations
    
    /**
     * @brief RDMA write (GPU to remote GPU)
     * @param targetNode Target node ID
     * @param localGpuPtr Local GPU memory pointer
     * @param remoteGpuPtr Remote GPU memory pointer
     * @param size Size in bytes
     * @param signaled Request completion notification
     * @return True if operation posted successfully
     */
    bool RdmaWrite(int targetNode, void* localGpuPtr, void* remoteGpuPtr, 
                   size_t size, bool signaled = true);
    
    /**
     * @brief RDMA read (remote GPU to GPU)
     * @param targetNode Target node ID
     * @param localGpuPtr Local GPU memory pointer
     * @param remoteGpuPtr Remote GPU memory pointer
     * @param size Size in bytes
     * @param signaled Request completion notification
     * @return True if operation posted successfully
     */
    bool RdmaRead(int targetNode, void* localGpuPtr, void* remoteGpuPtr, 
                  size_t size, bool signaled = true);
    
    // Synchronization
    
    /**
     * @brief Wait for RDMA operation completion
     * @param nodeId Node ID
     * @param timeoutMs Timeout in milliseconds
     * @return True if operation completed successfully
     */
    bool WaitForCompletion(int nodeId, uint32_t timeoutMs = 1000);
    
    /**
     * @brief Poll for completion (non-blocking)
     * @param nodeId Node ID
     * @return True if operation completed
     */
    bool PollCompletion(int nodeId);
    
    // Statistics
    
    /**
     * @brief Get RDMA statistics
     * @return Statistics
     */
    RdmaStats GetStatistics() const;
    
    /**
     * @brief Get connection info
     * @param nodeId Node ID
     * @return Connection info or nullptr if not connected
     */
    const RdmaConnection* GetConnection(int nodeId) const;
    
private:
    struct Impl;
    std::unique_ptr<Impl> m_Impl;
    
    // Internal helpers
    bool InitializeDevice();
    bool CreateQueuePair(int nodeId);
    bool ExchangeConnectionInfo(int nodeId, const std::string& address, uint16_t port);
};
