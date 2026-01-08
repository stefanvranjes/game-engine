#ifndef PROTOCOL_HPP
#define PROTOCOL_HPP

#include <cstdint>

/**
 * @brief Protocol-level message types for low-level network communication
 * 
 * These are distinct from Message.hpp's MessageType which handles
 * game-level messages. Protocol types are for connection management.
 */
enum class ProtocolCommand : uint8_t {
    // Handshake
    Hello = 0,          // Initial connection request
    Welcome = 1,        // Server acceptance response
    Denied = 2,         // Server rejection response
    
    // Connection status
    Heartbeat = 3,      // Keep-alive ping
    HeartbeatAck = 4,   // Keep-alive response
    
    // Flow control
    Reliable = 5,       // Reliable ordered packet wrapper
    Unreliable = 6,     // Unreliable unordered packet wrapper
    Fragment = 7,       // Fragmented large packet
    FragmentEnd = 8     // Last fragment of sequence
};

/**
 * @brief Protocol header prepended to all network packets
 */
struct ProtocolHeader {
    uint8_t magic[4] = {'G', 'E', 'N', 'T'};  // "GENT" - Game Engine Network
    uint8_t version = 1;                       // Protocol version
    ProtocolCommand command;                   // Packet type
    uint16_t sequence;                         // Sequence number for ordering
    uint32_t ack;                              // Last received sequence (for reliable)
    uint32_t ackBits;                          // Bitfield of received packets
    uint16_t payloadSize;                      // Size of following data
};

/**
 * @brief Connection state machine states
 */
enum class ConnectionState : uint8_t {
    Disconnected = 0,
    Connecting = 1,
    Connected = 2,
    Disconnecting = 3
};

/**
 * @brief Quality of Service settings for different message types
 */
struct QoSSettings {
    bool reliable = true;           // Guaranteed delivery
    bool ordered = true;            // Maintain order
    uint8_t priority = 0;           // 0 = highest priority
    uint16_t maxRetries = 10;       // Max retransmission attempts
    uint32_t timeoutMs = 5000;      // Retransmission timeout
};

// Predefined QoS presets
namespace QoS {
    inline constexpr QoSSettings ReliableOrdered   = {true,  true,  0, 10, 5000};
    inline constexpr QoSSettings ReliableUnordered = {true,  false, 1, 10, 5000};
    inline constexpr QoSSettings Unreliable        = {false, false, 2, 0,  0};
    inline constexpr QoSSettings StateUpdate       = {false, false, 3, 0,  0};  // For frequent updates
}

#endif // PROTOCOL_HPP