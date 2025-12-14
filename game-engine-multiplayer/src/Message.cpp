#include "../include/Message.hpp"
#include <cstring>
#include <stdexcept>

Message::Message(MessageType type, const std::string& content)
    : type(type), content(content) {}

MessageType Message::getType() const {
    return type;
}

std::string Message::getContent() const {
    return content;
}

void Message::setContent(const std::string& newContent) {
    content = newContent;
}

// Serialization implementation
std::vector<uint8_t> Message::serialize() const {
    std::vector<uint8_t> out;
    out.reserve(1 + 4 + content.size());
    out.push_back(static_cast<uint8_t>(type));

    uint32_t len = static_cast<uint32_t>(content.size());
    // little-endian store
    out.push_back(static_cast<uint8_t>(len & 0xFF));
    out.push_back(static_cast<uint8_t>((len >> 8) & 0xFF));
    out.push_back(static_cast<uint8_t>((len >> 16) & 0xFF));
    out.push_back(static_cast<uint8_t>((len >> 24) & 0xFF));

    out.insert(out.end(), content.begin(), content.end());
    return out;
}

bool Message::deserialize(const std::vector<uint8_t>& data, Message& out) {
    if (data.size() < 5) return false;
    MessageType t = static_cast<MessageType>(data[0]);
    uint32_t len = 0;
    len |= static_cast<uint32_t>(data[1]);
    len |= static_cast<uint32_t>(data[2]) << 8;
    len |= static_cast<uint32_t>(data[3]) << 16;
    len |= static_cast<uint32_t>(data[4]) << 24;
    if (data.size() < 5 + len) return false;
    std::string content;
    content.assign(reinterpret_cast<const char*>(data.data() + 5), len);
    out = Message(t, content);
    return true;
}