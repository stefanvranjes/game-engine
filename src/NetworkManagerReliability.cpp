// Message acknowledgment and reliability implementation
#include "NetworkManagerImpl.h" // Includes NetworkManager.h and defines Impl/PendingAck

// Implementations of NetworkManager::Impl methods defined in NetworkManagerReliability.cpp

uint32_t NetworkManager::Impl::CalculateRetryDelay(uint32_t retryCount) {
    uint32_t baseDelay = reliabilityConfig.retryDelayMs;
    uint32_t delay = baseDelay;
    
    switch (reliabilityConfig.retryStrategy) {
        case RetryStrategy::FIXED_DELAY:
            delay = baseDelay;
            break;
            
        case RetryStrategy::EXPONENTIAL_BACKOFF:
            delay = static_cast<uint32_t>(
                baseDelay * std::pow(reliabilityConfig.backoffMultiplier, retryCount)
            );
            break;
            
        case RetryStrategy::LINEAR_BACKOFF:
            delay = baseDelay * (retryCount + 1);
            break;
    }
    
    // Cap at max delay
    delay = std::min(delay, reliabilityConfig.maxRetryDelayMs);
    
    // Add jitter if enabled
    if (reliabilityConfig.enableJitter) {
        std::uniform_real_distribution<float> dist(
            1.0f - reliabilityConfig.jitterFactor,
            1.0f + reliabilityConfig.jitterFactor
        );
        delay = static_cast<uint32_t>(delay * dist(rng));
    }
    
    return delay;
}

void NetworkManager::Impl::ProcessAcknowledgments() {
    while (running) {
        auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        
        std::lock_guard<std::mutex> lock(ackMutex);
        
        // Check for timeouts and retry
        for (auto it = pendingAcks.begin(); it != pendingAcks.end();) {
            auto& pending = it->second;
            
            // Check if it's time to retry
            if (now >= pending.nextRetryTime) {
                if (pending.retryCount < reliabilityConfig.maxRetries) {
                    // Calculate next retry delay
                    uint32_t retryDelay = CalculateRetryDelay(pending.retryCount);
                    
                    // Update retry info
                    pending.retryCount++;
                    pending.sentTime = now;
                    pending.nextRetryTime = now + retryDelay;
                    pending.message.retryCount = pending.retryCount;
                    
                    std::cout << "Retrying message seq=" << it->first 
                              << " (attempt " << pending.retryCount 
                              << ", delay " << retryDelay << "ms)" << std::endl;
                    
                    // Resend message
                    SendMessageInternal(pending.targetNode, pending.message);
                    
                    stats.retriedMessages++;
                    ++it;
                } else {
                    // Max retries exceeded
                    std::cerr << "Message seq=" << it->first << " timed out after " 
                              << pending.retryCount << " retries" << std::endl;
                    
                    stats.timedOutMessages++;
                    stats.droppedMessages++;
                    it = pendingAcks.erase(it);
                }
            } else {
                ++it;
            }
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}

void NetworkManager::Impl::HandleAck(uint32_t ackSeqNum) {
    std::lock_guard<std::mutex> lock(ackMutex);
    
    auto it = pendingAcks.find(ackSeqNum);
    if (it != pendingAcks.end()) {
        // Calculate latency
        auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        float latency = static_cast<float>(now - it->second.sentTime);
        
        // Update average latency
        if (stats.acksReceived == 0) {
            stats.avgLatencyMs = latency;
        } else {
            stats.avgLatencyMs = (stats.avgLatencyMs * stats.acksReceived + latency) / 
                                (stats.acksReceived + 1);
        }
        
        stats.acksReceived++;
        pendingAcks.erase(it);
        
        std::cout << "ACK received for seq=" << ackSeqNum 
                  << " (latency: " << latency << "ms)" << std::endl;
    }
}

bool NetworkManager::Impl::IsDuplicate(uint32_t seqNum) {
    if (!reliabilityConfig.enableDuplicateDetection) {
        return false;
    }
    
    std::lock_guard<std::mutex> lock(ackMutex);
    
    if (receivedSequenceNumbers.count(seqNum) > 0) {
        stats.duplicatesReceived++;
        return true;
    }
    
    receivedSequenceNumbers.insert(seqNum);
    
    // Limit set size to prevent unbounded growth
    if (receivedSequenceNumbers.size() > 10000) {
        // Remove oldest entries (simplified - would use circular buffer in production)
        receivedSequenceNumbers.clear();
    }
    
    return false;
}

void NetworkManager::Impl::SendAck(int nodeId, uint32_t seqNum) {
    Message ackMsg;
    ackMsg.type = MessageType::ACK;
    ackMsg.sourceNode = localNodeId;
    ackMsg.targetNode = nodeId;
    ackMsg.ackSequenceNumber = seqNum;
    ackMsg.requiresAck = false;  // ACKs don't need ACKs
    
    SendMessageInternal(nodeId, ackMsg);
    stats.acksSent++;
}

bool NetworkManager::SendMessageToNode(int nodeId, const Message& msg, bool reliable) {
    Message sendMsg = msg;
    
    // Assign sequence number
    sendMsg.sequenceNumber = m_Impl->nextSequenceNumber++;
    sendMsg.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    
    if (reliable && m_Impl->reliabilityConfig.enableAcks) {
        sendMsg.requiresAck = true;
        sendMsg.sentTimestamp = sendMsg.timestamp;
        
        // Track for ACK
        std::lock_guard<std::mutex> lock(m_Impl->ackMutex);
        PendingAck pending;
        pending.message = sendMsg;
        pending.sentTime = sendMsg.timestamp;
        pending.retryCount = 0;
        pending.targetNode = nodeId;
        pending.nextRetryTime = sendMsg.timestamp + m_Impl->reliabilityConfig.ackTimeoutMs;
        
        m_Impl->pendingAcks[sendMsg.sequenceNumber] = pending;
    }
    
    return SendMessageInternal(nodeId, sendMsg);
}

void NetworkManager::SetReliabilityConfig(const ReliabilityConfig& config) {
    m_Impl->reliabilityConfig = config;
    
    std::cout << "Reliability config updated: ACKs=" << config.enableAcks
              << ", timeout=" << config.ackTimeoutMs << "ms"
              << ", maxRetries=" << config.maxRetries << std::endl;
}

NetworkManager::ReliabilityConfig NetworkManager::GetReliabilityConfig() const {
    return m_Impl->reliabilityConfig;
}

// Message processing with ACK handling
void NetworkManager::ProcessReceivedMessage(int nodeId, const Message& msg) {
    // Check for duplicates
    if (m_Impl->IsDuplicate(msg.sequenceNumber)) {
        std::cout << "Duplicate message seq=" << msg.sequenceNumber 
                  << " from node " << nodeId << ", ignoring" << std::endl;
        return;
    }
    
    // Handle ACK messages
    if (msg.type == MessageType::ACK) {
        m_Impl->HandleAck(msg.ackSequenceNumber);
        return;
    }
    
    // Send ACK if required
    if (msg.requiresAck) {
        m_Impl->SendAck(nodeId, msg.sequenceNumber);
    }
    
    // Process message normally
    if (m_Impl->messageCallback) {
        m_Impl->messageCallback(nodeId, msg);
    } else {
        // Queue message
        std::lock_guard<std::mutex> lock(m_Impl->connectionsMutex);
        auto it = m_Impl->connections.find(nodeId);
        if (it != m_Impl->connections.end()) {
            std::lock_guard<std::mutex> queueLock(it->second.queueMutex);
            it->second.messageQueue.push(msg);
        }
    }
}

// Initialize ACK thread
void NetworkManager::StartAckProcessing() {
    m_Impl->ackThread = std::thread([this]() {
        m_Impl->ProcessAcknowledgments();
    });
}

void NetworkManager::StopAckProcessing() {
    if (m_Impl->ackThread.joinable()) {
        m_Impl->ackThread.join();
    }
}
