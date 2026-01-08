#pragma once

#include "../../include/Math/Vec3.h"
#include <vector>
#include <cstdint>
#include <queue>

/**
 * @brief Player input data for network transmission
 */
struct PlayerInput {
    uint32_t tick = 0;              // Input sequence number
    float deltaTime = 0.0f;         // Time between inputs
    
    // Movement
    float moveX = 0.0f;             // -1 to 1 horizontal
    float moveY = 0.0f;             // -1 to 1 vertical (forward/back)
    bool jump = false;
    bool crouch = false;
    bool sprint = false;
    
    // Looking
    float lookYaw = 0.0f;           // Mouse X delta
    float lookPitch = 0.0f;         // Mouse Y delta
    
    // Actions
    bool primaryAction = false;     // Left click / fire
    bool secondaryAction = false;   // Right click / aim
    bool interact = false;
    bool reload = false;
    
    // Custom
    uint16_t customFlags = 0;       // Game-specific flags
    
    // Serialization
    void Serialize(std::vector<uint8_t>& outData) const;
    static PlayerInput Deserialize(const uint8_t* data, size_t length, size_t& bytesRead);
};

/**
 * @brief Circular buffer storing recent player inputs for prediction and rollback
 */
class InputBuffer {
public:
    static constexpr size_t MAX_BUFFER_SIZE = 128;
    
    InputBuffer() = default;
    ~InputBuffer() = default;
    
    /**
     * @brief Record a new input
     * @param input The input to record
     */
    void Push(const PlayerInput& input);
    
    /**
     * @brief Get input by tick number
     * @return Pointer to input, or nullptr if not found
     */
    const PlayerInput* GetByTick(uint32_t tick) const;
    
    /**
     * @brief Get all inputs after a specific tick
     * @param afterTick Get inputs with tick > afterTick
     * @return Vector of inputs
     */
    std::vector<PlayerInput> GetInputsAfter(uint32_t afterTick) const;
    
    /**
     * @brief Get the most recent input
     */
    const PlayerInput* GetLatest() const;
    
    /**
     * @brief Get current tick counter
     */
    uint32_t GetCurrentTick() const { return m_CurrentTick; }
    
    /**
     * @brief Clear all buffered inputs
     */
    void Clear();
    
    /**
     * @brief Remove all inputs up to and including the specified tick
     */
    void AcknowledgeTick(uint32_t tick);
    
    /**
     * @brief Get number of buffered inputs
     */
    size_t Size() const { return m_Buffer.size(); }
    
private:
    std::deque<PlayerInput> m_Buffer;
    uint32_t m_CurrentTick = 0;
};

/**
 * @brief Handles client-side prediction and server reconciliation
 */
class ClientPrediction {
public:
    struct Settings {
        float reconciliationThreshold = 0.1f;   // Distance threshold for correction
        float correctionSpeed = 0.2f;           // Interpolation speed for smooth correction
        bool enablePrediction = true;
        bool enableReconciliation = true;
    };
    
    ClientPrediction() = default;
    ~ClientPrediction() = default;
    
    /**
     * @brief Record local input for prediction
     */
    void RecordInput(const PlayerInput& input);
    
    /**
     * @brief Apply prediction to local player
     * @param currentPosition Current position
     * @param currentVelocity Current velocity
     * @param moveSpeed Movement speed multiplier
     * @return Predicted position after applying input
     */
    Vec3 ApplyPrediction(const Vec3& currentPosition, const Vec3& currentVelocity, float moveSpeed);
    
    /**
     * @brief Reconcile server authoritative state with local prediction
     * @param serverTick The tick this server state corresponds to
     * @param serverPosition Server's authoritative position
     * @param serverVelocity Server's authoritative velocity
     * @return Corrected position
     */
    Vec3 Reconcile(uint32_t serverTick, const Vec3& serverPosition, const Vec3& serverVelocity);
    
    /**
     * @brief Get inputs to send to server
     */
    std::vector<PlayerInput> GetPendingInputs() const;
    
    /**
     * @brief Acknowledge that server received inputs up to this tick
     */
    void AcknowledgeServerTick(uint32_t tick);
    
    // Settings
    Settings& GetSettings() { return m_Settings; }
    const Settings& GetSettings() const { return m_Settings; }
    
    // Debug info
    float GetPredictionError() const { return m_LastPredictionError; }
    uint32_t GetPendingInputCount() const { return static_cast<uint32_t>(m_InputBuffer.Size()); }
    
private:
    Settings m_Settings;
    InputBuffer m_InputBuffer;
    
    Vec3 m_PredictedPosition;
    Vec3 m_PredictedVelocity;
    Vec3 m_CorrectionOffset;
    float m_LastPredictionError = 0.0f;
    
    // Apply a single input to position
    Vec3 SimulateInput(const Vec3& position, const Vec3& velocity, 
                       const PlayerInput& input, float moveSpeed) const;
};
