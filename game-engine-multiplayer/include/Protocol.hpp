#ifndef PROTOCOL_HPP
#define PROTOCOL_HPP

enum class MessageType {
    CONNECT,
    DISCONNECT,
    TEXT,
    BINARY,
    PING,
    PONG
};

struct Message {
    MessageType type;
    std::string payload;

    Message(MessageType type, const std::string& payload)
        : type(type), payload(payload) {}
};

#endif // PROTOCOL_HPP