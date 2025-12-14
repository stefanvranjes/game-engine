#include "serialization/Serializer.hpp"
#include "Message.hpp"
#include <stdexcept>
#include <cstring>

Serializer::Serializer() {}

std::vector<char> Serializer::serialize(const Message& message) {
    std::vector<char> buffer;
    size_t size = sizeof(message.type) + sizeof(message.length) + message.length;

    buffer.resize(size);
    char* ptr = buffer.data();

    std::memcpy(ptr, &message.type, sizeof(message.type));
    ptr += sizeof(message.type);
    std::memcpy(ptr, &message.length, sizeof(message.length));
    ptr += sizeof(message.length);
    std::memcpy(ptr, message.data, message.length);

    return buffer;
}

Message Serializer::deserialize(const std::vector<char>& buffer) {
    if (buffer.size() < sizeof(Message::type) + sizeof(Message::length)) {
        throw std::runtime_error("Buffer too small for deserialization");
    }

    Message message;
    const char* ptr = buffer.data();

    std::memcpy(&message.type, ptr, sizeof(message.type));
    ptr += sizeof(message.type);
    std::memcpy(&message.length, ptr, sizeof(message.length));
    ptr += sizeof(message.length);

    message.data = new char[message.length];
    std::memcpy(message.data, ptr, message.length);

    return message;
}