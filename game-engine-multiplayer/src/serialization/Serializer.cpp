#include "serialization/Serializer.hpp"
#include "Message.hpp"
#include <stdexcept>
#include <cstring>
#include <cstdint>

Serializer::Serializer() {}

std::vector<char> Serializer::serialize(const Message& message) {
    std::vector<char> buffer;
    
    MessageType type = message.getType();
    std::string content = message.getContent();
    uint32_t length = static_cast<uint32_t>(content.length());

    // Calculate size: 1 byte type + 4 bytes length + content
    size_t size = sizeof(uint8_t) + sizeof(uint32_t) + length;

    buffer.resize(size);
    char* ptr = buffer.data();

    // Write type (1 byte)
    uint8_t typeVal = static_cast<uint8_t>(type);
    std::memcpy(ptr, &typeVal, sizeof(uint8_t));
    ptr += sizeof(uint8_t);

    // Write length (4 bytes)
    std::memcpy(ptr, &length, sizeof(uint32_t));
    ptr += sizeof(uint32_t);

    // Write content
    if (length > 0) {
        std::memcpy(ptr, content.data(), length);
    }

    return buffer;
}

Message Serializer::deserialize(const std::vector<char>& buffer) {
    // Minimum size: 1 byte type + 4 bytes length
    if (buffer.size() < sizeof(uint8_t) + sizeof(uint32_t)) {
        throw std::runtime_error("Buffer too small for deserialization");
    }

    const char* ptr = buffer.data();

    // Read type
    uint8_t typeVal;
    std::memcpy(&typeVal, ptr, sizeof(uint8_t));
    MessageType type = static_cast<MessageType>(typeVal);
    ptr += sizeof(uint8_t);

    // Read length
    uint32_t length;
    std::memcpy(&length, ptr, sizeof(uint32_t));
    ptr += sizeof(uint32_t);

    // Verify buffer has enough data
    size_t headerSize = sizeof(uint8_t) + sizeof(uint32_t);
    if (buffer.size() < headerSize + length) {
        throw std::runtime_error("Buffer incomplete for content deserialization");
    }

    // Read content
    std::string content;
    if (length > 0) {
        content.assign(ptr, length);
    }

    return Message(type, content);
}