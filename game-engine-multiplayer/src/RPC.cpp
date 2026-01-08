#include "RPC.hpp"
#include <cstring>

void RPCManager::RegisterHandler(const std::string& methodName,
                                  std::function<void(const RPCInfo&)> handler) {
    m_Handlers[methodName] = [handler](const RPCInfo& info, const std::vector<uint8_t>&) {
        handler(info);
    };
}

void RPCManager::RegisterHandler(const std::string& methodName,
                                  std::function<void(const RPCInfo&, const std::string&)> handler) {
    m_Handlers[methodName] = [handler](const RPCInfo& info, const std::vector<uint8_t>& data) {
        std::string param(reinterpret_cast<const char*>(data.data()), data.size());
        handler(info, param);
    };
}

void RPCManager::RegisterHandler(const std::string& methodName,
                                  std::function<void(const RPCInfo&, int32_t)> handler) {
    m_Handlers[methodName] = [handler](const RPCInfo& info, const std::vector<uint8_t>& data) {
        if (data.size() >= 4) {
            int32_t value = 0;
            value |= static_cast<int32_t>(data[0]);
            value |= static_cast<int32_t>(data[1]) << 8;
            value |= static_cast<int32_t>(data[2]) << 16;
            value |= static_cast<int32_t>(data[3]) << 24;
            handler(info, value);
        }
    };
}

void RPCManager::RegisterHandler(const std::string& methodName,
                                  std::function<void(const RPCInfo&, float, float, float)> handler) {
    m_Handlers[methodName] = [handler](const RPCInfo& info, const std::vector<uint8_t>& data) {
        if (data.size() >= 12) {
            auto readFloat = [&data](size_t offset) -> float {
                uint32_t bits = 0;
                bits |= static_cast<uint32_t>(data[offset]);
                bits |= static_cast<uint32_t>(data[offset + 1]) << 8;
                bits |= static_cast<uint32_t>(data[offset + 2]) << 16;
                bits |= static_cast<uint32_t>(data[offset + 3]) << 24;
                float value;
                std::memcpy(&value, &bits, sizeof(float));
                return value;
            };
            handler(info, readFloat(0), readFloat(4), readFloat(8));
        }
    };
}

void RPCManager::UnregisterHandler(const std::string& methodName) {
    m_Handlers.erase(methodName);
}

void RPCManager::Invoke(const std::string& methodName, RPCTarget target,
                         RPCMode mode, uint32_t entityId) {
    PendingRPC rpc;
    rpc.methodName = methodName;
    rpc.target = target;
    rpc.mode = mode;
    rpc.entityId = entityId;
    m_PendingRPCs.push_back(std::move(rpc));
}

void RPCManager::Invoke(const std::string& methodName, RPCTarget target,
                         const std::string& param1, RPCMode mode) {
    PendingRPC rpc;
    rpc.methodName = methodName;
    rpc.target = target;
    rpc.mode = mode;
    rpc.entityId = 0;
    rpc.params.assign(param1.begin(), param1.end());
    m_PendingRPCs.push_back(std::move(rpc));
}

void RPCManager::Invoke(const std::string& methodName, RPCTarget target,
                         int32_t param1, RPCMode mode) {
    PendingRPC rpc;
    rpc.methodName = methodName;
    rpc.target = target;
    rpc.mode = mode;
    rpc.entityId = 0;
    rpc.params.push_back(static_cast<uint8_t>(param1 & 0xFF));
    rpc.params.push_back(static_cast<uint8_t>((param1 >> 8) & 0xFF));
    rpc.params.push_back(static_cast<uint8_t>((param1 >> 16) & 0xFF));
    rpc.params.push_back(static_cast<uint8_t>((param1 >> 24) & 0xFF));
    m_PendingRPCs.push_back(std::move(rpc));
}

void RPCManager::Invoke(const std::string& methodName, RPCTarget target,
                         float x, float y, float z, RPCMode mode) {
    PendingRPC rpc;
    rpc.methodName = methodName;
    rpc.target = target;
    rpc.mode = mode;
    rpc.entityId = 0;
    
    auto writeFloat = [&rpc](float value) {
        uint32_t bits;
        std::memcpy(&bits, &value, sizeof(float));
        rpc.params.push_back(static_cast<uint8_t>(bits & 0xFF));
        rpc.params.push_back(static_cast<uint8_t>((bits >> 8) & 0xFF));
        rpc.params.push_back(static_cast<uint8_t>((bits >> 16) & 0xFF));
        rpc.params.push_back(static_cast<uint8_t>((bits >> 24) & 0xFF));
    };
    
    writeFloat(x);
    writeFloat(y);
    writeFloat(z);
    
    m_PendingRPCs.push_back(std::move(rpc));
}

void RPCManager::ProcessRPC(const uint8_t* data, size_t length, uint32_t senderId) {
    std::string methodName;
    uint32_t entityId;
    std::vector<uint8_t> params;
    
    if (!DeserializeRPC(data, length, methodName, entityId, params)) {
        return;
    }
    
    auto it = m_Handlers.find(methodName);
    if (it == m_Handlers.end()) {
        return;  // Unknown RPC
    }
    
    RPCInfo info;
    info.callerId = senderId;
    info.entityId = entityId;
    info.timestamp = 0.0f;  // TODO: Add proper timing
    
    it->second(info, params);
}

std::vector<std::pair<RPCTarget, std::vector<uint8_t>>> RPCManager::GetPendingRPCs() {
    std::vector<std::pair<RPCTarget, std::vector<uint8_t>>> result;
    
    for (const auto& rpc : m_PendingRPCs) {
        std::vector<uint8_t> data;
        SerializeRPC(data, rpc);
        result.emplace_back(rpc.target, std::move(data));
    }
    
    m_PendingRPCs.clear();
    return result;
}

void RPCManager::ClearPending() {
    m_PendingRPCs.clear();
}

void RPCManager::SerializeRPC(std::vector<uint8_t>& outData, const PendingRPC& rpc) const {
    // Message type
    outData.push_back(static_cast<uint8_t>(MessageType::RPC));
    
    // Method name length (1 byte) + name
    uint8_t nameLen = static_cast<uint8_t>(std::min(rpc.methodName.size(), size_t(255)));
    outData.push_back(nameLen);
    outData.insert(outData.end(), rpc.methodName.begin(), rpc.methodName.begin() + nameLen);
    
    // Entity ID (4 bytes)
    outData.push_back(static_cast<uint8_t>(rpc.entityId & 0xFF));
    outData.push_back(static_cast<uint8_t>((rpc.entityId >> 8) & 0xFF));
    outData.push_back(static_cast<uint8_t>((rpc.entityId >> 16) & 0xFF));
    outData.push_back(static_cast<uint8_t>((rpc.entityId >> 24) & 0xFF));
    
    // Params length (2 bytes) + params
    uint16_t paramsLen = static_cast<uint16_t>(rpc.params.size());
    outData.push_back(static_cast<uint8_t>(paramsLen & 0xFF));
    outData.push_back(static_cast<uint8_t>((paramsLen >> 8) & 0xFF));
    outData.insert(outData.end(), rpc.params.begin(), rpc.params.end());
}

bool RPCManager::DeserializeRPC(const uint8_t* data, size_t length,
                                 std::string& methodName, uint32_t& entityId,
                                 std::vector<uint8_t>& params) const {
    if (length < 8) return false;
    
    size_t offset = 0;
    
    // Skip message type (caller handles this)
    if (data[offset] == static_cast<uint8_t>(MessageType::RPC)) {
        offset++;
    }
    
    // Method name
    uint8_t nameLen = data[offset++];
    if (offset + nameLen > length) return false;
    methodName.assign(reinterpret_cast<const char*>(data + offset), nameLen);
    offset += nameLen;
    
    // Entity ID
    if (offset + 4 > length) return false;
    entityId = 0;
    entityId |= static_cast<uint32_t>(data[offset++]);
    entityId |= static_cast<uint32_t>(data[offset++]) << 8;
    entityId |= static_cast<uint32_t>(data[offset++]) << 16;
    entityId |= static_cast<uint32_t>(data[offset++]) << 24;
    
    // Params
    if (offset + 2 > length) return false;
    uint16_t paramsLen = 0;
    paramsLen |= static_cast<uint16_t>(data[offset++]);
    paramsLen |= static_cast<uint16_t>(data[offset++]) << 8;
    
    if (offset + paramsLen > length) return false;
    params.assign(data + offset, data + offset + paramsLen);
    
    return true;
}
