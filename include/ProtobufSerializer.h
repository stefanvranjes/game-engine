#pragma once

#include "StateSerializer.h"

#ifdef HAS_PROTOBUF
#include "distributed_physics.pb.h"
#endif

/**
 * @brief Protocol Buffers adapter for StateSerializer
 * 
 * Provides conversion between PhysXSoftBody and Protocol Buffers messages
 * for type-safe network serialization.
 */
class ProtobufSerializer {
public:
#ifdef HAS_PROTOBUF
    /**
     * @brief Convert soft body to protobuf message
     * @param softBody Soft body to convert
     * @param compress Use LZ4 compression for vertex data
     * @return Protobuf message
     */
    static distributed_physics::SoftBodyState ToProtobuf(
        const PhysXSoftBody* softBody, 
        bool compress = true);
    
    /**
     * @brief Convert protobuf message to soft body state
     * @param message Protobuf message
     * @param softBody Target soft body
     * @return True if successful
     */
    static bool FromProtobuf(
        const distributed_physics::SoftBodyState& message,
        PhysXSoftBody* softBody);
    
    /**
     * @brief Create delta protobuf message
     * @param softBody Current soft body state
     * @param previousState Previous state
     * @param compress Use LZ4 compression
     * @return Delta message
     */
    static distributed_physics::SoftBodyDelta ToDeltaProtobuf(
        const PhysXSoftBody* softBody,
        const StateSerializer::SoftBodyState& previousState,
        bool compress = true);
    
    /**
     * @brief Apply delta protobuf message
     * @param message Delta message
     * @param softBody Target soft body
     * @return True if successful
     */
    static bool FromDeltaProtobuf(
        const distributed_physics::SoftBodyDelta& message,
        PhysXSoftBody* softBody);
    
    /**
     * @brief Serialize protobuf message to bytes
     * @param message Protobuf message
     * @return Serialized bytes
     */
    template<typename T>
    static std::vector<uint8_t> SerializeMessage(const T& message) {
        std::vector<uint8_t> buffer(message.ByteSizeLong());
        message.SerializeToArray(buffer.data(), buffer.size());
        return buffer;
    }
    
    /**
     * @brief Deserialize bytes to protobuf message
     * @param data Serialized bytes
     * @param message Output message
     * @return True if successful
     */
    template<typename T>
    static bool DeserializeMessage(const std::vector<uint8_t>& data, T& message) {
        return message.ParseFromArray(data.data(), data.size());
    }
    
private:
    static StateSerializer s_Serializer;
#endif
};
