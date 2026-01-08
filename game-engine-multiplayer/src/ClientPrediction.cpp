#include "ClientPrediction.hpp"
#include <cstring>
#include <cmath>
#include <algorithm>

// ============================================================================
// PlayerInput
// ============================================================================

void PlayerInput::Serialize(std::vector<uint8_t>& outData) const {
    // Tick (4 bytes)
    outData.push_back(static_cast<uint8_t>(tick & 0xFF));
    outData.push_back(static_cast<uint8_t>((tick >> 8) & 0xFF));
    outData.push_back(static_cast<uint8_t>((tick >> 16) & 0xFF));
    outData.push_back(static_cast<uint8_t>((tick >> 24) & 0xFF));
    
    // Delta time (4 bytes as float)
    uint32_t dtBits;
    std::memcpy(&dtBits, &deltaTime, sizeof(float));
    outData.push_back(static_cast<uint8_t>(dtBits & 0xFF));
    outData.push_back(static_cast<uint8_t>((dtBits >> 8) & 0xFF));
    outData.push_back(static_cast<uint8_t>((dtBits >> 16) & 0xFF));
    outData.push_back(static_cast<uint8_t>((dtBits >> 24) & 0xFF));
    
    // Movement axes (quantized to bytes for bandwidth efficiency)
    outData.push_back(static_cast<uint8_t>((moveX + 1.0f) * 127.5f));
    outData.push_back(static_cast<uint8_t>((moveY + 1.0f) * 127.5f));
    
    // Boolean flags packed into single byte
    uint8_t flags = 0;
    if (jump) flags |= 0x01;
    if (crouch) flags |= 0x02;
    if (sprint) flags |= 0x04;
    if (primaryAction) flags |= 0x08;
    if (secondaryAction) flags |= 0x10;
    if (interact) flags |= 0x20;
    if (reload) flags |= 0x40;
    outData.push_back(flags);
    
    // Look deltas (2 bytes each, quantized)
    auto quantizeAngle = [&outData](float angle) {
        int16_t q = static_cast<int16_t>(std::clamp(angle, -180.0f, 180.0f) * 182.0f);
        outData.push_back(static_cast<uint8_t>(q & 0xFF));
        outData.push_back(static_cast<uint8_t>((q >> 8) & 0xFF));
    };
    quantizeAngle(lookYaw);
    quantizeAngle(lookPitch);
    
    // Custom flags (2 bytes)
    outData.push_back(static_cast<uint8_t>(customFlags & 0xFF));
    outData.push_back(static_cast<uint8_t>((customFlags >> 8) & 0xFF));
}

PlayerInput PlayerInput::Deserialize(const uint8_t* data, size_t length, size_t& bytesRead) {
    PlayerInput input;
    bytesRead = 0;
    
    if (length < 17) return input;  // Minimum size
    
    size_t offset = 0;
    
    // Tick
    input.tick = 0;
    input.tick |= static_cast<uint32_t>(data[offset++]);
    input.tick |= static_cast<uint32_t>(data[offset++]) << 8;
    input.tick |= static_cast<uint32_t>(data[offset++]) << 16;
    input.tick |= static_cast<uint32_t>(data[offset++]) << 24;
    
    // Delta time
    uint32_t dtBits = 0;
    dtBits |= static_cast<uint32_t>(data[offset++]);
    dtBits |= static_cast<uint32_t>(data[offset++]) << 8;
    dtBits |= static_cast<uint32_t>(data[offset++]) << 16;
    dtBits |= static_cast<uint32_t>(data[offset++]) << 24;
    std::memcpy(&input.deltaTime, &dtBits, sizeof(float));
    
    // Movement axes
    input.moveX = (static_cast<float>(data[offset++]) / 127.5f) - 1.0f;
    input.moveY = (static_cast<float>(data[offset++]) / 127.5f) - 1.0f;
    
    // Flags
    uint8_t flags = data[offset++];
    input.jump = (flags & 0x01) != 0;
    input.crouch = (flags & 0x02) != 0;
    input.sprint = (flags & 0x04) != 0;
    input.primaryAction = (flags & 0x08) != 0;
    input.secondaryAction = (flags & 0x10) != 0;
    input.interact = (flags & 0x20) != 0;
    input.reload = (flags & 0x40) != 0;
    
    // Look deltas
    auto dequantizeAngle = [&data, &offset]() -> float {
        int16_t q = 0;
        q |= static_cast<int16_t>(data[offset++]);
        q |= static_cast<int16_t>(data[offset++]) << 8;
        return static_cast<float>(q) / 182.0f;
    };
    input.lookYaw = dequantizeAngle();
    input.lookPitch = dequantizeAngle();
    
    // Custom flags
    input.customFlags = 0;
    input.customFlags |= static_cast<uint16_t>(data[offset++]);
    input.customFlags |= static_cast<uint16_t>(data[offset++]) << 8;
    
    bytesRead = offset;
    return input;
}

// ============================================================================
// InputBuffer
// ============================================================================

void InputBuffer::Push(const PlayerInput& input) {
    PlayerInput copy = input;
    copy.tick = ++m_CurrentTick;
    
    m_Buffer.push_back(copy);
    
    // Limit buffer size
    while (m_Buffer.size() > MAX_BUFFER_SIZE) {
        m_Buffer.pop_front();
    }
}

const PlayerInput* InputBuffer::GetByTick(uint32_t tick) const {
    for (const auto& input : m_Buffer) {
        if (input.tick == tick) {
            return &input;
        }
    }
    return nullptr;
}

std::vector<PlayerInput> InputBuffer::GetInputsAfter(uint32_t afterTick) const {
    std::vector<PlayerInput> result;
    for (const auto& input : m_Buffer) {
        if (input.tick > afterTick) {
            result.push_back(input);
        }
    }
    return result;
}

const PlayerInput* InputBuffer::GetLatest() const {
    if (m_Buffer.empty()) return nullptr;
    return &m_Buffer.back();
}

void InputBuffer::Clear() {
    m_Buffer.clear();
}

void InputBuffer::AcknowledgeTick(uint32_t tick) {
    while (!m_Buffer.empty() && m_Buffer.front().tick <= tick) {
        m_Buffer.pop_front();
    }
}

// ============================================================================
// ClientPrediction
// ============================================================================

void ClientPrediction::RecordInput(const PlayerInput& input) {
    m_InputBuffer.Push(input);
}

Vec3 ClientPrediction::ApplyPrediction(const Vec3& currentPosition, 
                                        const Vec3& currentVelocity, 
                                        float moveSpeed) {
    if (!m_Settings.enablePrediction) {
        return currentPosition;
    }
    
    const PlayerInput* latestInput = m_InputBuffer.GetLatest();
    if (!latestInput) {
        return currentPosition;
    }
    
    m_PredictedPosition = SimulateInput(currentPosition, currentVelocity, *latestInput, moveSpeed);
    
    // Apply smooth correction offset
    if (m_CorrectionOffset.x != 0 || m_CorrectionOffset.y != 0 || m_CorrectionOffset.z != 0) {
        float blend = m_Settings.correctionSpeed;
        m_CorrectionOffset.x *= (1.0f - blend);
        m_CorrectionOffset.y *= (1.0f - blend);
        m_CorrectionOffset.z *= (1.0f - blend);
        
        // Zero out tiny offsets
        if (std::abs(m_CorrectionOffset.x) < 0.001f) m_CorrectionOffset.x = 0;
        if (std::abs(m_CorrectionOffset.y) < 0.001f) m_CorrectionOffset.y = 0;
        if (std::abs(m_CorrectionOffset.z) < 0.001f) m_CorrectionOffset.z = 0;
    }
    
    return Vec3(
        m_PredictedPosition.x + m_CorrectionOffset.x,
        m_PredictedPosition.y + m_CorrectionOffset.y,
        m_PredictedPosition.z + m_CorrectionOffset.z
    );
}

Vec3 ClientPrediction::Reconcile(uint32_t serverTick, const Vec3& serverPosition, 
                                  const Vec3& serverVelocity) {
    if (!m_Settings.enableReconciliation) {
        return serverPosition;
    }
    
    // Get client's predicted position at that tick
    const PlayerInput* inputAtTick = m_InputBuffer.GetByTick(serverTick);
    
    // Calculate prediction error
    Vec3 diff = Vec3(
        m_PredictedPosition.x - serverPosition.x,
        m_PredictedPosition.y - serverPosition.y,
        m_PredictedPosition.z - serverPosition.z
    );
    
    m_LastPredictionError = std::sqrt(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z);
    
    // If error is below threshold, no correction needed
    if (m_LastPredictionError < m_Settings.reconciliationThreshold) {
        AcknowledgeServerTick(serverTick);
        return m_PredictedPosition;
    }
    
    // Re-simulate from server state using buffered inputs
    Vec3 correctedPosition = serverPosition;
    Vec3 correctedVelocity = serverVelocity;
    
    std::vector<PlayerInput> pendingInputs = m_InputBuffer.GetInputsAfter(serverTick);
    
    for (const auto& input : pendingInputs) {
        correctedPosition = SimulateInput(correctedPosition, correctedVelocity, input, 5.0f);
    }
    
    // Calculate correction offset for smooth interpolation
    m_CorrectionOffset = Vec3(
        m_PredictedPosition.x - correctedPosition.x,
        m_PredictedPosition.y - correctedPosition.y,
        m_PredictedPosition.z - correctedPosition.z
    );
    
    m_PredictedPosition = correctedPosition;
    
    AcknowledgeServerTick(serverTick);
    
    return correctedPosition;
}

std::vector<PlayerInput> ClientPrediction::GetPendingInputs() const {
    std::vector<PlayerInput> result;
    for (size_t i = 0; i < m_InputBuffer.Size(); ++i) {
        // Get all inputs (they're all pending until acknowledged)
        const PlayerInput* input = m_InputBuffer.GetLatest();
        if (input) {
            result.push_back(*input);
        }
    }
    return m_InputBuffer.GetInputsAfter(0);
}

void ClientPrediction::AcknowledgeServerTick(uint32_t tick) {
    m_InputBuffer.AcknowledgeTick(tick);
}

Vec3 ClientPrediction::SimulateInput(const Vec3& position, const Vec3& velocity,
                                      const PlayerInput& input, float moveSpeed) const {
    // Simple movement simulation
    float dt = input.deltaTime;
    float speed = moveSpeed * (input.sprint ? 1.5f : 1.0f) * (input.crouch ? 0.5f : 1.0f);
    
    Vec3 movement(
        input.moveX * speed * dt,
        0.0f,  // Y handled by physics for gravity
        input.moveY * speed * dt
    );
    
    return Vec3(
        position.x + movement.x,
        position.y + movement.y,
        position.z + movement.z
    );
}
