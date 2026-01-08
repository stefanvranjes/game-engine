#pragma once

#include "Math/Vec3.h"
#include "IPhysicsShape.h"
#include <memory>

/**
 * @brief Abstract interface for character controller physics
 * 
 * Character controllers are specialized physics objects optimized for
 * player and NPC movement with features like grounded detection and jumping.
 */
class IPhysicsCharacterController {
public:
    virtual ~IPhysicsCharacterController() = default;

    /**
     * @brief Initialize the character controller
     * @param shape Collision shape (typically capsule)
     * @param mass Mass in kilograms
     * @param stepHeight Maximum step height for climbing
     */
    virtual void Initialize(std::shared_ptr<IPhysicsShape> shape, float mass, float stepHeight) = 0;

    /**
     * @brief Update the controller (called each frame)
     * @param deltaTime Time step in seconds
     * @param gravity Gravity vector
     */
    virtual void Update(float deltaTime, const Vec3& gravity) = 0;

    /**
     * @brief Set walk direction
     * @param direction Movement direction and speed
     */
    virtual void SetWalkDirection(const Vec3& direction) = 0;

    /**
     * @brief Get walk direction
     * @return Current walk direction
     */
    virtual Vec3 GetWalkDirection() const = 0;

    /**
     * @brief Make the character jump
     * @param impulse Jump impulse vector
     */
    virtual void Jump(const Vec3& impulse) = 0;

    /**
     * @brief Check if character is on the ground
     * @return True if grounded
     */
    virtual bool IsGrounded() const = 0;

    /**
     * @brief Get current position
     * @return World space position
     */
    virtual Vec3 GetPosition() const = 0;

    /**
     * @brief Set position
     * @param position New world space position
     */
    virtual void SetPosition(const Vec3& position) = 0;

    /**
     * @brief Get vertical velocity
     * @return Vertical velocity in m/s
     */
    virtual float GetVerticalVelocity() const = 0;

    /**
     * @brief Set vertical velocity
     * @param velocity Vertical velocity in m/s
     */
    virtual void SetVerticalVelocity(float velocity) = 0;

    /**
     * @brief Set maximum walk speed
     * @param speed Maximum speed in m/s
     */
    virtual void SetMaxWalkSpeed(float speed) = 0;

    /**
     * @brief Get maximum walk speed
     * @return Maximum speed in m/s
     */
    virtual float GetMaxWalkSpeed() const = 0;

    /**
     * @brief Set fall speed
     * @param speed Fall speed in m/s
     */
    virtual void SetFallSpeed(float speed) = 0;

    /**
     * @brief Get fall speed
     * @return Fall speed in m/s
     */
    virtual float GetFallSpeed() const = 0;

    /**
     * @brief Set step height
     * @param height Maximum step height in meters
     */
    virtual void SetStepHeight(float height) = 0;

    /**
     * @brief Get step height
     * @return Maximum step height in meters
     */
    virtual float GetStepHeight() const = 0;

    /**
     * @brief Get backend-specific controller pointer
     * @return Opaque pointer to native controller
     */
    virtual void* GetNativeController() = 0;
};
