#include "RdmaManager.h"
#include <iostream>
#include <cstring>
#include <chrono>

#ifdef HAS_RDMA
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#endif

// Implementation structure
struct RdmaManager::Impl {
#ifdef HAS_RDMA
    ibv_context* context = nullptr;
    ibv_pd* protectionDomain = nullptr;
    ibv_device** deviceList = nullptr;
    int numDevices = 0;
    
    struct QueuePairInfo {
        ibv_qp* qp = nullptr;
        ibv_cq* sendCq = nullptr;
        ibv_cq* recvCq = nullptr;
        uint32_t qpNum = 0;
        uint16_t lid = 0;
    };
    
    std::unordered_map<int, QueuePairInfo> queuePairs;
    std::unordered_map<int, RdmaConnection> connections;
    std::unordered_map<void*, ibv_mr*> registeredMemory;
    std::unordered_map<void*, RdmaContext> rdmaContexts;
#endif
    
    bool initialized = false;
    RdmaStats stats = {};
    
    ~Impl() {
#ifdef HAS_RDMA
        // Cleanup registered memory
        for (auto& [ptr, mr] : registeredMemory) {
            if (mr) {
                ibv_dereg_mr(mr);
            }
        }
        
        // Cleanup queue pairs
        for (auto& [nodeId, qpInfo] : queuePairs) {
            if (qpInfo.qp) ibv_destroy_qp(qpInfo.qp);
            if (qpInfo.sendCq) ibv_destroy_cq(qpInfo.sendCq);
            if (qpInfo.recvCq) ibv_destroy_cq(qpInfo.recvCq);
        }
        
        // Cleanup protection domain and context
        if (protectionDomain) ibv_dealloc_pd(protectionDomain);
        if (context) ibv_close_device(context);
        if (deviceList) ibv_free_device_list(deviceList);
#endif
    }
};

RdmaManager::RdmaManager() : m_Impl(std::make_unique<Impl>()) {}

RdmaManager::~RdmaManager() {
    Shutdown();
}

bool RdmaManager::Initialize() {
#ifdef HAS_RDMA
    if (m_Impl->initialized) {
        return true;
    }
    
    std::cout << "Initializing RDMA..." << std::endl;
    
    // Get device list
    m_Impl->deviceList = ibv_get_device_list(&m_Impl->numDevices);
    if (!m_Impl->deviceList || m_Impl->numDevices == 0) {
        std::cerr << "No RDMA devices found" << std::endl;
        return false;
    }
    
    std::cout << "Found " << m_Impl->numDevices << " RDMA device(s)" << std::endl;
    
    // Open first device
    m_Impl->context = ibv_open_device(m_Impl->deviceList[0]);
    if (!m_Impl->context) {
        std::cerr << "Failed to open RDMA device" << std::endl;
        return false;
    }
    
    std::cout << "Opened RDMA device: " 
              << ibv_get_device_name(m_Impl->deviceList[0]) << std::endl;
    
    // Allocate protection domain
    m_Impl->protectionDomain = ibv_alloc_pd(m_Impl->context);
    if (!m_Impl->protectionDomain) {
        std::cerr << "Failed to allocate protection domain" << std::endl;
        return false;
    }
    
    m_Impl->initialized = true;
    std::cout << "RDMA initialized successfully" << std::endl;
    
    return true;
#else
    std::cerr << "RDMA support not compiled in" << std::endl;
    return false;
#endif
}

bool RdmaManager::IsRdmaAvailable() const {
    return m_Impl->initialized;
}

void RdmaManager::Shutdown() {
#ifdef HAS_RDMA
    if (!m_Impl->initialized) {
        return;
    }
    
    std::cout << "Shutting down RDMA..." << std::endl;
    
    // Disconnect all nodes
    std::vector<int> nodeIds;
    for (const auto& [nodeId, _] : m_Impl->connections) {
        nodeIds.push_back(nodeId);
    }
    
    for (int nodeId : nodeIds) {
        DisconnectNode(nodeId);
    }
    
    m_Impl->initialized = false;
    std::cout << "RDMA shutdown complete" << std::endl;
#endif
}

bool RdmaManager::ConnectToNode(int nodeId, const std::string& address, uint16_t port) {
#ifdef HAS_RDMA
    if (!m_Impl->initialized) {
        std::cerr << "RDMA not initialized" << std::endl;
        return false;
    }
    
    std::cout << "Connecting to node " << nodeId << " at " 
              << address << ":" << port << std::endl;
    
    // Create queue pair
    if (!CreateQueuePair(nodeId)) {
        return false;
    }
    
    // Exchange connection info
    if (!ExchangeConnectionInfo(nodeId, address, port)) {
        return false;
    }
    
    // Mark as connected
    RdmaConnection conn;
    conn.nodeId = nodeId;
    conn.isConnected = true;
    conn.bytesTransferred = 0;
    conn.operationsCompleted = 0;
    
    m_Impl->connections[nodeId] = conn;
    
    std::cout << "Connected to node " << nodeId << " via RDMA" << std::endl;
    
    return true;
#else
    return false;
#endif
}

void RdmaManager::DisconnectNode(int nodeId) {
#ifdef HAS_RDMA
    auto it = m_Impl->connections.find(nodeId);
    if (it != m_Impl->connections.end()) {
        std::cout << "Disconnecting from node " << nodeId << std::endl;
        
        // Destroy queue pair
        auto qpIt = m_Impl->queuePairs.find(nodeId);
        if (qpIt != m_Impl->queuePairs.end()) {
            if (qpIt->second.qp) ibv_destroy_qp(qpIt->second.qp);
            if (qpIt->second.sendCq) ibv_destroy_cq(qpIt->second.sendCq);
            if (qpIt->second.recvCq) ibv_destroy_cq(qpIt->second.recvCq);
            m_Impl->queuePairs.erase(qpIt);
        }
        
        m_Impl->connections.erase(it);
    }
#endif
}

bool RdmaManager::IsConnected(int nodeId) const {
#ifdef HAS_RDMA
    auto it = m_Impl->connections.find(nodeId);
    return it != m_Impl->connections.end() && it->second.isConnected;
#else
    return false;
#endif
}

bool RdmaManager::RegisterGpuMemory(void* gpuPtr, size_t size, 
                                     uint32_t& lkey, uint32_t& rkey) {
#ifdef HAS_RDMA
    if (!m_Impl->initialized) {
        return false;
    }
    
    // Check if already registered
    if (m_Impl->registeredMemory.count(gpuPtr) > 0) {
        auto* mr = m_Impl->registeredMemory[gpuPtr];
        lkey = mr->lkey;
        rkey = mr->rkey;
        return true;
    }
    
    std::cout << "Registering GPU memory for RDMA: " << gpuPtr 
              << " (" << (size / 1024) << " KB)" << std::endl;
    
    // Register GPU memory with RDMA
    ibv_mr* mr = ibv_reg_mr(m_Impl->protectionDomain, gpuPtr, size,
                           IBV_ACCESS_LOCAL_WRITE |
                           IBV_ACCESS_REMOTE_WRITE |
                           IBV_ACCESS_REMOTE_READ);
    
    if (!mr) {
        std::cerr << "Failed to register GPU memory" << std::endl;
        return false;
    }
    
    lkey = mr->lkey;
    rkey = mr->rkey;
    
    m_Impl->registeredMemory[gpuPtr] = mr;
    
    // Store context
    RdmaContext ctx;
    ctx.gpuBuffer = gpuPtr;
    ctx.bufferSize = size;
    ctx.lkey = lkey;
    ctx.rkey = rkey;
    ctx.remoteAddr = 0;  // Set during connection exchange
    
    m_Impl->rdmaContexts[gpuPtr] = ctx;
    
    std::cout << "GPU memory registered (lkey=" << lkey 
              << ", rkey=" << rkey << ")" << std::endl;
    
    return true;
#else
    return false;
#endif
}

void RdmaManager::UnregisterGpuMemory(void* gpuPtr) {
#ifdef HAS_RDMA
    auto it = m_Impl->registeredMemory.find(gpuPtr);
    if (it != m_Impl->registeredMemory.end()) {
        ibv_dereg_mr(it->second);
        m_Impl->registeredMemory.erase(it);
        m_Impl->rdmaContexts.erase(gpuPtr);
        
        std::cout << "Unregistered GPU memory: " << gpuPtr << std::endl;
    }
#endif
}

const RdmaManager::RdmaContext* RdmaManager::GetRdmaContext(void* gpuPtr) const {
#ifdef HAS_RDMA
    auto it = m_Impl->rdmaContexts.find(gpuPtr);
    if (it != m_Impl->rdmaContexts.end()) {
        return &it->second;
    }
#endif
    return nullptr;
}

bool RdmaManager::RdmaWrite(int targetNode, void* localGpuPtr, void* remoteGpuPtr,
                            size_t size, bool signaled) {
#ifdef HAS_RDMA
    if (!IsConnected(targetNode)) {
        std::cerr << "Not connected to node " << targetNode << std::endl;
        return false;
    }
    
    auto localMr = m_Impl->registeredMemory.find(localGpuPtr);
    if (localMr == m_Impl->registeredMemory.end()) {
        std::cerr << "Local GPU memory not registered" << std::endl;
        return false;
    }
    
    auto qpIt = m_Impl->queuePairs.find(targetNode);
    if (qpIt == m_Impl->queuePairs.end()) {
        std::cerr << "No queue pair for node " << targetNode << std::endl;
        return false;
    }
    
    // Prepare scatter-gather element
    ibv_sge sge = {};
    sge.addr = (uint64_t)localGpuPtr;
    sge.length = size;
    sge.lkey = localMr->second->lkey;
    
    // Prepare work request
    ibv_send_wr wr = {};
    wr.wr_id = (uint64_t)targetNode;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_WRITE;
    wr.send_flags = signaled ? IBV_SEND_SIGNALED : 0;
    wr.wr.rdma.remote_addr = (uint64_t)remoteGpuPtr;
    wr.wr.rdma.rkey = 0;  // TODO: Get from connection exchange
    
    // Post send
    ibv_send_wr* badWr;
    int ret = ibv_post_send(qpIt->second.qp, &wr, &badWr);
    
    if (ret != 0) {
        std::cerr << "Failed to post RDMA write" << std::endl;
        m_Impl->stats.failedOperations++;
        return false;
    }
    
    m_Impl->stats.totalWrites++;
    m_Impl->stats.totalBytesTransferred += size;
    m_Impl->connections[targetNode].bytesTransferred += size;
    
    return true;
#else
    return false;
#endif
}

bool RdmaManager::RdmaRead(int targetNode, void* localGpuPtr, void* remoteGpuPtr,
                           size_t size, bool signaled) {
#ifdef HAS_RDMA
    // Similar to RdmaWrite but with IBV_WR_RDMA_READ opcode
    // Implementation omitted for brevity
    m_Impl->stats.totalReads++;
    return true;
#else
    return false;
#endif
}

bool RdmaManager::WaitForCompletion(int nodeId, uint32_t timeoutMs) {
#ifdef HAS_RDMA
    auto qpIt = m_Impl->queuePairs.find(nodeId);
    if (qpIt == m_Impl->queuePairs.end()) {
        return false;
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    ibv_wc wc;
    while (true) {
        int ret = ibv_poll_cq(qpIt->second.sendCq, 1, &wc);
        
        if (ret > 0) {
            if (wc.status == IBV_WC_SUCCESS) {
                m_Impl->connections[nodeId].operationsCompleted++;
                return true;
            } else {
                std::cerr << "RDMA operation failed: " << wc.status << std::endl;
                m_Impl->stats.failedOperations++;
                return false;
            }
        }
        
        auto now = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - start).count();
        
        if (elapsed > timeoutMs) {
            std::cerr << "RDMA operation timeout" << std::endl;
            return false;
        }
        
        std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
#else
    return false;
#endif
}

bool RdmaManager::PollCompletion(int nodeId) {
#ifdef HAS_RDMA
    auto qpIt = m_Impl->queuePairs.find(nodeId);
    if (qpIt == m_Impl->queuePairs.end()) {
        return false;
    }
    
    ibv_wc wc;
    int ret = ibv_poll_cq(qpIt->second.sendCq, 1, &wc);
    
    if (ret > 0 && wc.status == IBV_WC_SUCCESS) {
        m_Impl->connections[nodeId].operationsCompleted++;
        return true;
    }
    
    return false;
#else
    return false;
#endif
}

RdmaManager::RdmaStats RdmaManager::GetStatistics() const {
    return m_Impl->stats;
}

const RdmaManager::RdmaConnection* RdmaManager::GetConnection(int nodeId) const {
#ifdef HAS_RDMA
    auto it = m_Impl->connections.find(nodeId);
    if (it != m_Impl->connections.end()) {
        return &it->second;
    }
#endif
    return nullptr;
}

// Private helper methods

bool RdmaManager::CreateQueuePair(int nodeId) {
#ifdef HAS_RDMA
    // Create completion queues
    ibv_cq* sendCq = ibv_create_cq(m_Impl->context, 128, nullptr, nullptr, 0);
    ibv_cq* recvCq = ibv_create_cq(m_Impl->context, 128, nullptr, nullptr, 0);
    
    if (!sendCq || !recvCq) {
        std::cerr << "Failed to create completion queues" << std::endl;
        return false;
    }
    
    // Create queue pair
    ibv_qp_init_attr qpAttr = {};
    qpAttr.send_cq = sendCq;
    qpAttr.recv_cq = recvCq;
    qpAttr.qp_type = IBV_QPT_RC;  // Reliable connection
    qpAttr.cap.max_send_wr = 128;
    qpAttr.cap.max_recv_wr = 128;
    qpAttr.cap.max_send_sge = 1;
    qpAttr.cap.max_recv_sge = 1;
    
    ibv_qp* qp = ibv_create_qp(m_Impl->protectionDomain, &qpAttr);
    
    if (!qp) {
        std::cerr << "Failed to create queue pair" << std::endl;
        return false;
    }
    
    // Store queue pair info
    Impl::QueuePairInfo qpInfo;
    qpInfo.qp = qp;
    qpInfo.sendCq = sendCq;
    qpInfo.recvCq = recvCq;
    qpInfo.qpNum = qp->qp_num;
    
    m_Impl->queuePairs[nodeId] = qpInfo;
    
    std::cout << "Created queue pair for node " << nodeId 
              << " (QP num: " << qp->qp_num << ")" << std::endl;
    
    return true;
#else
    return false;
#endif
}

bool RdmaManager::ExchangeConnectionInfo(int nodeId, const std::string& address, 
                                          uint16_t port) {
#ifdef HAS_RDMA
    // TODO: Implement connection info exchange via TCP
    // This would exchange QP numbers, LIDs, and GIDs
    // For now, return true as placeholder
    std::cout << "Exchanging connection info with " << address << ":" << port << std::endl;
    return true;
#else
    return false;
#endif
}
