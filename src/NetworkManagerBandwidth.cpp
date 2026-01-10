// Bandwidth optimization implementation
#include "NetworkManager.h"
#include <queue>
#include <chrono>

struct MessageBatch {
    std::vector<NetworkManager::Message> messages;
    uint64_t firstMessageTime;
    size_t totalSize;
};

struct NetworkManager::Impl {
    // ... existing members ...
    
    // Bandwidth optimization
    std::unordered_map<int, MessageBatch> pendingBatches;  // nodeId -> batch
    std::mutex batchMutex;
    std::thread batchThread;
    
    // Bandwidth throttling
    uint64_t lastSendTime = 0;
    size_t bytesThisSecond = 0;
    std::mutex throttleMutex;
    
    bool ShouldCompress(const std::vector<uint8_t>& data) {
        if (!reliabilityConfig.adaptiveCompression) {
            return true;  // Always compress if adaptive is off
        }
        
        // Only compress if data is larger than threshold
        return data.size() >= reliabilityConfig.compressionThreshold;
    }
    
    std::vector<uint8_t> CompressMessageHeader(const Message& msg) {
        std::vector<uint8_t> header;
        
        if (!reliabilityConfig.enableHeaderCompression) {
            // Standard header (uncompressed)
            header.push_back(static_cast<uint8_t>(msg.type));
            // Add other fields...
            return header;
        }
        
        // Compressed header format:
        // - Use variable-length encoding for integers
        // - Omit default values
        // - Pack flags into single byte
        
        uint8_t flags = 0;
        flags |= (static_cast<uint8_t>(msg.type) & 0x0F);  // 4 bits for type
        flags |= (msg.requiresAck ? 0x10 : 0);
        flags |= (msg.retryCount > 0 ? 0x20 : 0);
        
        header.push_back(flags);
        
        // Variable-length encoding for sequence number
        uint32_t seq = msg.sequenceNumber;
        while (seq > 0x7F) {
            header.push_back((seq & 0x7F) | 0x80);
            seq >>= 7;
        }
        header.push_back(seq & 0x7F);
        
        return header;
    }
    
    void ProcessBatches() {
        while (running) {
            auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            
            std::lock_guard<std::mutex> lock(batchMutex);
            
            for (auto it = pendingBatches.begin(); it != pendingBatches.end();) {
                auto& batch = it->second;
                int nodeId = it->first;
                
                // Send batch if:
                // 1. Batch is full
                // 2. Timeout exceeded
                bool shouldSend = (batch.messages.size() >= reliabilityConfig.maxBatchSize) ||
                                 (now - batch.firstMessageTime >= reliabilityConfig.batchTimeoutMs);
                
                if (shouldSend && !batch.messages.empty()) {
                    SendBatch(nodeId, batch);
                    it = pendingBatches.erase(it);
                } else {
                    ++it;
                }
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    }
    
    void SendBatch(int nodeId, const MessageBatch& batch) {
        // Serialize batch
        std::vector<uint8_t> batchData;
        
        // Batch header
        batchData.push_back(0xFF);  // Batch marker
        batchData.push_back(static_cast<uint8_t>(batch.messages.size()));
        
        // Serialize each message
        for (const auto& msg : batch.messages) {
            auto header = CompressMessageHeader(msg);
            batchData.insert(batchData.end(), header.begin(), header.end());
            batchData.insert(batchData.end(), msg.data.begin(), msg.data.end());
        }
        
        // Compress if beneficial
        size_t originalSize = batchData.size();
        if (ShouldCompress(batchData)) {
            // Use existing compression (LZ4 via StateSerializer)
            // batchData = Compress(batchData);
        }
        
        // Apply bandwidth throttling
        ThrottleBandwidth(batchData.size());
        
        // Send batch
        SendMessageInternal(nodeId, batchData);
        
        // Update statistics
        stats.messagesBatched += batch.messages.size();
        stats.batchesSent++;
        stats.bytesBeforeCompression += originalSize;
        stats.bytesAfterCompression += batchData.size();
        
        if (stats.bytesBeforeCompression > 0) {
            stats.compressionRatio = static_cast<float>(stats.bytesAfterCompression) / 
                                    stats.bytesBeforeCompression;
        }
    }
    
    void ThrottleBandwidth(size_t bytes) {
        if (reliabilityConfig.maxBandwidthKBps == 0) {
            return;  // No throttling
        }
        
        std::lock_guard<std::mutex> lock(throttleMutex);
        
        auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        
        // Reset counter every second
        if (now - lastSendTime >= 1000) {
            bytesThisSecond = 0;
            lastSendTime = now;
        }
        
        bytesThisSecond += bytes;
        
        // Calculate current bandwidth
        stats.currentBandwidthKBps = bytesThisSecond / 1024.0f;
        
        // Check if we've exceeded limit
        size_t maxBytesPerSecond = reliabilityConfig.maxBandwidthKBps * 1024;
        if (bytesThisSecond > maxBytesPerSecond) {
            // Calculate sleep time to stay under limit
            uint64_t elapsed = now - lastSendTime;
            uint64_t sleepTime = 1000 - elapsed;
            
            if (sleepTime > 0 && sleepTime < 1000) {
                std::cout << "Bandwidth throttling: sleeping " << sleepTime 
                          << "ms (current: " << stats.currentBandwidthKBps << " KB/s)" 
                          << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(sleepTime));
            }
        }
    }
};

bool NetworkManager::SendMessageBatched(int nodeId, const Message& msg) {
    if (!m_Impl->reliabilityConfig.enableMessageBatching) {
        return SendMessage(nodeId, msg);
    }
    
    std::lock_guard<std::mutex> lock(m_Impl->batchMutex);
    
    auto& batch = m_Impl->pendingBatches[nodeId];
    
    if (batch.messages.empty()) {
        batch.firstMessageTime = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
    }
    
    batch.messages.push_back(msg);
    batch.totalSize += msg.data.size();
    
    // If batch is full, send immediately
    if (batch.messages.size() >= m_Impl->reliabilityConfig.maxBatchSize) {
        m_Impl->SendBatch(nodeId, batch);
        m_Impl->pendingBatches.erase(nodeId);
    }
    
    return true;
}

void NetworkManager::FlushBatches() {
    std::lock_guard<std::mutex> lock(m_Impl->batchMutex);
    
    for (auto& [nodeId, batch] : m_Impl->pendingBatches) {
        if (!batch.messages.empty()) {
            m_Impl->SendBatch(nodeId, batch);
        }
    }
    
    m_Impl->pendingBatches.clear();
}

void NetworkManager::StartBatchProcessing() {
    m_Impl->batchThread = std::thread([this]() {
        m_Impl->ProcessBatches();
    });
}

void NetworkManager::StopBatchProcessing() {
    if (m_Impl->batchThread.joinable()) {
        m_Impl->batchThread.join();
    }
}
