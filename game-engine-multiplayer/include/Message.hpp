#pragma once
#include <cstdint>
#include <string>
#include <vector>

enum class MessageType : uint8_t {
    // Basic messages
    Chat = 0,
    Join = 1,
    Leave = 2,
    Ping = 3,
    State = 4,
    
    // Replication messages
    Snapshot = 5,       // Full state update
    Delta = 6,          // Incremental state update
    Input = 7,          // Player input data
    RPC = 8,            // Remote procedure call
    Spawn = 9,          // Entity creation
    Destroy = 10,       // Entity removal
    
    // Connection management
    Connect = 11,       // Connection request/response
    Disconnect = 12,    // Graceful disconnect
    Ack = 13            // Acknowledgment
};

class Message {
public:
    Message(MessageType type = MessageType::Chat, const std::string& content = "");

    MessageType getType() const;
    std::string getContent() const;
    void setContent(const std::string& newContent);

    // PascalCase Aliases (as used in tests)
    MessageType GetType() const { return getType(); }
    std::string GetContent() const { return getContent(); }
    void SetContent(const std::string& newContent) { setContent(newContent); }

    // Simple binary serialization: [1 byte type][4 bytes length (uint32 little-endian)][data bytes]
    std::vector<uint8_t> serialize() const;
    
    // Returns true if deserialize succeeded
    static bool deserialize(const std::vector<uint8_t>& data, Message& out);
    
    // Static PascalCase Alias
    static Message deserialize(const std::vector<uint8_t>& data) {
        Message out;
        deserialize(data, out);
        return out;
    }

private:
    MessageType type;
    std::string content;
};