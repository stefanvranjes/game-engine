#pragma once

#include "Math/Vec3.h"
#include "Math/Quat.h"
#include "PhysicsCollisionShape.h"
#include <memory>

// Forward declarations
class btKinematicCharacterController;
class btPairCachingGhostObject;
class PhysicsSystem;

/**
 * @brief Kinematic character controller for player/NPC movement
 * 
 * Handles grounded state, jumping, sliding, and collision response without
 * using traditional dynamic bodies. More suitable for character movement
 * than rigid bodies.
 * 
 * Usage:
 *   auto controller = std::make_shared<KinematicController>();
 *   controller->Initialize(capsuleShape, mass, stepHeight);
 *   gameObject->SetKinematicController(controller);
 *   
 *   // Each frame:
 *   controller->SetWalkDirection({moveX, 0, moveZ});
 *   if (shouldJump) controller->Jump({0, jumpForce, 0});
 *   controller->Update(deltaTime);
 */
class KinematicController : public std::enable_shared_from_this<KinematicController> {
public:
    KinematicController();
    ~KinematicController();

    // No copying
    KinematicController(const KinematicController&) = delete;
    KinematicController& operator=(const KinematicController&) = delete;

    /**
     * @brief Initialize the kinematic controller
     * @param shape Collision shape (typically capsule)
     * @param mass Mass in kg (used for inertia calculations)
     * @param stepHeight How high the character can step up (e.g., 0.3f for stairs)
     */
    void Initialize(const PhysicsCollisionShape& shape, float mass, float stepHeight = 0.35f);

    /**
     * @brief Check if initialized
     */
    bool IsInitialized() const { return m_Controller != nullptr; }

    /**
     * @brief Update controller (call every frame)
     * @param deltaTime Time step in seconds
     * @param gravity Current gravity vector
     */
    void Update(float deltaTime, const Vec3& gravity = Vec3(0, -9.81f, 0));

    /**
     * @brief Set the direction and speed the character should walk
     * @param direction Desired movement direction (world space)
     */
    void SetWalkDirection(const Vec3& direction);

    /**
     * @brief Get current walk direction
     */
    Vec3 GetWalkDirection() const { return m_WalkDirection; }

    /**
     * @brief Apply an impulse (for jumping, knockback, etc.)
     * @param impulse Impulse vector
     */
    void Jump(const Vec3& impulse);

    /**
     * @brief Check if the character is on the ground
     */
    bool IsGrounded() const;

    /**
     * @brief Get the vertical velocity (for jump state detection)
     */
    float GetVerticalVelocity() const { return m_VerticalVelocity; }

    /**
     * @brief Set the fall velocity (used internally, exposed for advanced control)
     */
    void SetVerticalVelocity(float velocity) { m_VerticalVelocity = velocity; }

    /**
     * @brief Get the current position
     */
    Vec3 GetPosition() const;

    /**
     * @brief Set the position
     */
    void SetPosition(const Vec3& position);

    /**
     * @brief Set maximum walk speed
     */
    void SetMaxWalkSpeed(float speed) { m_MaxWalkSpeed = speed; }

    /**
     * @brief Get maximum walk speed
     */
    float GetMaxWalkSpeed() const { return m_MaxWalkSpeed; }

    /**
     * @brief Set falling speed (gravity acceleration)
     */
    void SetFallSpeed(float speed) { m_FallSpeed = speed; }

    /**
     * @brief Get falling speed
     */
    float GetFallSpeed() const { return m_FallSpeed; }

    /**
     * @brief Set step height (for climbing stairs)
     */
    void SetStepHeight(float height);

    /**
     * @brief Get step height
     */
    float GetStepHeight() const;

    /**
     * @brief Enable or disable the controller
     */
    void SetActive(bool active);

    /**
     * @brief Check if controller is active
     */
    bool IsActive() const;

    /**
     * @brief Get the underlying Bullet3D character controller
     */
    btKinematicCharacterController* GetBulletController() const { return m_Controller; }

    /**
     * @brief Get the ghost object (collision object)
     */
    btPairCachingGhostObject* GetGhostObject() const { return m_GhostObject; }

    /**
     * @brief Synchronize position from physics world to GameObject
     * @internal Called by PhysicsSystem
     */
    void SyncTransformFromPhysics(Vec3& outPosition) const;

    /**
     * @brief Synchronize position from GameObject to physics world
     * @internal Called by Update
     */
    void SyncTransformToPhysics(const Vec3& position);

    /**
     * @brief Set collision group and mask
     */
    void SetCollisionFilterGroup(uint16_t group);
    void SetCollisionFilterMask(uint16_t mask);

    // ==================== ADVANCED MOVEMENT QUERIES ====================

    /**
     * @brief Check if there's a wall directly ahead (for wall-slide detection)
     * @param maxDistance How far to check ahead
     * @param outWallNormal Normal of the wall surface
     * @return True if a wall was detected ahead
     */
    bool IsWallAhead(float maxDistance = 1.0f, Vec3* outWallNormal = nullptr) const;

    /**
     * @brief Check if character is on a slope and get the slope angle
     * @param outSlopeAngle Output angle in radians (0 = flat, PI/2 = vertical)
     * @return True if on a slope
     */
    bool IsOnSlope(float* outSlopeAngle = nullptr) const;

    /**
     * @brief Get the ground normal beneath the character
     * @param outNormal The normal of the surface below
     * @param maxDistance How far below to check
     * @return True if ground was found
     */
    bool GetGroundNormal(Vec3& outNormal, float maxDistance = 0.5f) const;

    /**
     * @brief Predict if a movement will cause collision
     * @param moveDirection Direction and distance to test (world space)
     * @param outBlockingDirection Output: direction away from obstacle
     * @return True if movement would be blocked
     */
    bool WillMoveCollide(const Vec3& moveDirection, Vec3* outBlockingDirection = nullptr) const;

    /**
     * @brief Get the distance to a collision in a direction
     * @param direction Direction to test (normalized or with magnitude)
     * @return Distance to collision, or max float if no collision
     */
    float GetDistanceToCollision(const Vec3& direction) const;

    /**
     * @brief Check if character can jump (is grounded)
     * Uses internal grounded state rather than raycasting
     */
    bool CanJump() const { return IsGrounded(); }

    /**
     * @brief Get the velocity of the character (includes fall velocity)
     * @return Current velocity vector
     */
    Vec3 GetVelocity() const;

    /**
     * @brief Get current move speed (horizontal)
     */
    float GetMoveSpeed() const;

    /**
     * @brief Check if character is in the air (falling or jumping)
     */
    bool IsInAir() const { return !IsGrounded(); }

private:
    btKinematicCharacterController* m_Controller;
    btPairCachingGhostObject* m_GhostObject;
    
    Vec3 m_WalkDirection;
    float m_VerticalVelocity;
    float m_MaxWalkSpeed;
    float m_FallSpeed;
    float m_Mass;

    PhysicsCollisionShape m_Shape;

    // Friend class for physics system access
    friend class PhysicsSystem;
};
