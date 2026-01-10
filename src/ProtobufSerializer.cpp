#include "ProtobufSerializer.h"
#include "PhysXSoftBody.h"

#ifdef HAS_PROTOBUF

StateSerializer ProtobufSerializer::s_Serializer;

distributed_physics::SoftBodyState ProtobufSerializer::ToProtobuf(
    const PhysXSoftBody* softBody, 
    bool compress) {
    
    distributed_physics::SoftBodyState message;
    
    if (!softBody) {
        return message;
    }
    
    // Set metadata
    message.set_vertex_count(static_cast<uint32_t>(softBody->GetVertexCount()));
    message.set_tetrahedron_count(static_cast<uint32_t>(softBody->GetTetrahedronCount()));
    message.set_frame_number(0);  // TODO: Get actual frame number
    
    // Set flags
    message.set_has_tearing(softBody->HasTearing());
    message.set_has_plasticity(softBody->HasPlasticity());
    message.set_has_self_collision(softBody->HasSelfCollision());
    
    // Get vertex data
    uint32_t vertexCount = message.vertex_count();
    const float* positions = softBody->GetVertexPositions();
    const float* velocities = softBody->GetVertexVelocities();
    
    if (compress) {
        // Compress and store as bytes
        std::vector<uint8_t> posData(reinterpret_cast<const uint8_t*>(positions),
                                     reinterpret_cast<const uint8_t*>(positions + vertexCount * 3));
        std::vector<uint8_t> velData(reinterpret_cast<const uint8_t*>(velocities),
                                     reinterpret_cast<const uint8_t*>(velocities + vertexCount * 3));
        
        auto compressedPos = s_Serializer.Compress(posData);
        auto compressedVel = s_Serializer.Compress(velData);
        
        message.set_compressed_positions(compressedPos.data(), compressedPos.size());
        message.set_compressed_velocities(compressedVel.data(), compressedVel.size());
        message.set_is_compressed(true);
    } else {
        // Store uncompressed
        for (uint32_t i = 0; i < vertexCount * 3; ++i) {
            message.add_positions(positions[i]);
            message.add_velocities(velocities[i]);
        }
        message.set_is_compressed(false);
    }
    
    return message;
}

bool ProtobufSerializer::FromProtobuf(
    const distributed_physics::SoftBodyState& message,
    PhysXSoftBody* softBody) {
    
    if (!softBody) {
        return false;
    }
    
    uint32_t vertexCount = message.vertex_count();
    
    // Validate vertex count
    if (vertexCount != softBody->GetVertexCount()) {
        std::cerr << "Vertex count mismatch in protobuf message" << std::endl;
        return false;
    }
    
    std::vector<float> positions(vertexCount * 3);
    std::vector<float> velocities(vertexCount * 3);
    
    if (message.is_compressed()) {
        // Decompress data
        std::vector<uint8_t> compressedPos(message.compressed_positions().begin(),
                                           message.compressed_positions().end());
        std::vector<uint8_t> compressedVel(message.compressed_velocities().begin(),
                                           message.compressed_velocities().end());
        
        auto posData = s_Serializer.Decompress(compressedPos);
        auto velData = s_Serializer.Decompress(compressedVel);
        
        if (posData.size() != vertexCount * 3 * sizeof(float) ||
            velData.size() != vertexCount * 3 * sizeof(float)) {
            std::cerr << "Decompressed data size mismatch" << std::endl;
            return false;
        }
        
        std::memcpy(positions.data(), posData.data(), posData.size());
        std::memcpy(velocities.data(), velData.data(), velData.size());
    } else {
        // Copy uncompressed data
        if (message.positions_size() != static_cast<int>(vertexCount * 3) ||
            message.velocities_size() != static_cast<int>(vertexCount * 3)) {
            std::cerr << "Position/velocity array size mismatch" << std::endl;
            return false;
        }
        
        for (uint32_t i = 0; i < vertexCount * 3; ++i) {
            positions[i] = message.positions(i);
            velocities[i] = message.velocities(i);
        }
    }
    
    // Apply to soft body
    softBody->SetVertexPositions(positions.data(), vertexCount);
    softBody->SetVertexVelocities(velocities.data(), vertexCount);
    
    return true;
}

distributed_physics::SoftBodyDelta ProtobufSerializer::ToDeltaProtobuf(
    const PhysXSoftBody* softBody,
    const StateSerializer::SoftBodyState& previousState,
    bool compress) {
    
    distributed_physics::SoftBodyDelta message;
    
    if (!softBody) {
        return message;
    }
    
    message.set_frame_number(0);  // TODO: Get actual frame number
    message.set_base_frame_number(previousState.frameNumber);
    
    // Calculate deltas
    uint32_t vertexCount = static_cast<uint32_t>(softBody->GetVertexCount());
    const float* currentPos = softBody->GetVertexPositions();
    const float* currentVel = softBody->GetVertexVelocities();
    
    std::vector<float> deltaPos(vertexCount * 3);
    std::vector<float> deltaVel(vertexCount * 3);
    
    for (uint32_t i = 0; i < vertexCount * 3; ++i) {
        deltaPos[i] = currentPos[i] - previousState.positions[i];
        deltaVel[i] = currentVel[i] - previousState.velocities[i];
    }
    
    if (compress) {
        // Compress deltas
        std::vector<uint8_t> posData(reinterpret_cast<const uint8_t*>(deltaPos.data()),
                                     reinterpret_cast<const uint8_t*>(deltaPos.data() + deltaPos.size()));
        std::vector<uint8_t> velData(reinterpret_cast<const uint8_t*>(deltaVel.data()),
                                     reinterpret_cast<const uint8_t*>(deltaVel.data() + deltaVel.size()));
        
        auto compressedPos = s_Serializer.Compress(posData);
        auto compressedVel = s_Serializer.Compress(velData);
        
        message.set_compressed_position_deltas(compressedPos.data(), compressedPos.size());
        message.set_compressed_velocity_deltas(compressedVel.data(), compressedVel.size());
        message.set_is_compressed(true);
    } else {
        // Store uncompressed deltas
        for (float delta : deltaPos) {
            message.add_position_deltas(delta);
        }
        for (float delta : deltaVel) {
            message.add_velocity_deltas(delta);
        }
        message.set_is_compressed(false);
    }
    
    return message;
}

bool ProtobufSerializer::FromDeltaProtobuf(
    const distributed_physics::SoftBodyDelta& message,
    PhysXSoftBody* softBody) {
    
    if (!softBody) {
        return false;
    }
    
    uint32_t vertexCount = static_cast<uint32_t>(softBody->GetVertexCount());
    
    // Get current state
    const float* currentPos = softBody->GetVertexPositions();
    const float* currentVel = softBody->GetVertexVelocities();
    
    std::vector<float> deltaPos(vertexCount * 3);
    std::vector<float> deltaVel(vertexCount * 3);
    
    if (message.is_compressed()) {
        // Decompress deltas
        std::vector<uint8_t> compressedPos(message.compressed_position_deltas().begin(),
                                           message.compressed_position_deltas().end());
        std::vector<uint8_t> compressedVel(message.compressed_velocity_deltas().begin(),
                                           message.compressed_velocity_deltas().end());
        
        auto posData = s_Serializer.Decompress(compressedPos);
        auto velData = s_Serializer.Decompress(compressedVel);
        
        std::memcpy(deltaPos.data(), posData.data(), posData.size());
        std::memcpy(deltaVel.data(), velData.data(), velData.size());
    } else {
        // Copy uncompressed deltas
        for (uint32_t i = 0; i < vertexCount * 3; ++i) {
            deltaPos[i] = message.position_deltas(i);
            deltaVel[i] = message.velocity_deltas(i);
        }
    }
    
    // Apply deltas
    std::vector<float> newPos(vertexCount * 3);
    std::vector<float> newVel(vertexCount * 3);
    
    for (uint32_t i = 0; i < vertexCount * 3; ++i) {
        newPos[i] = currentPos[i] + deltaPos[i];
        newVel[i] = currentVel[i] + deltaVel[i];
    }
    
    // Update soft body
    softBody->SetVertexPositions(newPos.data(), vertexCount);
    softBody->SetVertexVelocities(newVel.data(), vertexCount);
    
    return true;
}

#endif // HAS_PROTOBUF
