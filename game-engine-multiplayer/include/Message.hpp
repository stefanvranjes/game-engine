#pragma once
#include <cstdint>
#include <string>
#include <vector>

enum class MessageType : uint8_t {
    Chat = 0,
    Join = 1,
    Leave = 2,
    Ping = 3,
    State = 4
};

class Message {
public:
    Message(MessageType type = MessageType::Chat, const std::string& content = "");

    MessageType getType() const;
    std::string getContent() const;
    void setContent(const std::string& newContent);

    // Simple binary serialization: [1 byte type][4 bytes length (uint32 little-endian)][data bytes]
    std::vector<uint8_t> serialize() const;
    // Returns true if deserialize succeeded
    static bool deserialize(const std::vector<uint8_t>& data, Message& out);

private:
    MessageType type;
    std::string content;
};