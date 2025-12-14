#pragma once

#include "Math/Vec3.h"
#include "Math/Quat.h"
#include "PhysicsCollisionShape.h"
#include <memory>

// Forward declarations
class btRigidBody;
class btMotionState;
class PhysicsSystem;

/**
 * @brief Body type for physics simulation
 */
enum class BodyType {
    Static,     // Non-moving bodies (walls, terrain, platforms)
    Dynamic,    // Affected by gravity and forces
    Kinematic   // Moved by code, affects dynamic bodies
};

/**
 * @brief Rigid body component for physics simulation
 * 
 * Attach to GameObjects to give them physical properties.
 * 
 * Usage:
 *   auto rigidBody = std::make_shared<RigidBody>();
 *   rigidBody->Initialize(BodyType::Dynamic, mass, collisionShape);
 *   gameObject->SetRigidBody(rigidBody);
 */
class RigidBody : public std::enable_shared_from_this<RigidBody> {
public:
    RigidBody();
    ~RigidBody();

    // No copying
    RigidBody(const RigidBody&) = delete;
    RigidBody& operator=(const RigidBody&) = delete;

    /**
     * @brief Initialize the rigid body
     * @param type Body type (Static, Dynamic, Kinematic)
     * @param mass Mass in kg (0 for static/kinematic)
     * @param shape Collision shape
     */
    void Initialize(BodyType type, float mass, const PhysicsCollisionShape& shape);

    /**
     * @brief Check if the body is initialized
     */
    bool IsInitialized() const { return m_BtRigidBody != nullptr; }

    /**
     * @brief Get the body type
     */
    BodyType GetBodyType() const { return m_BodyType; }

    /**
     * @brief Set linear velocity
     */
    void SetLinearVelocity(const Vec3& velocity);

    /**
     * @brief Get linear velocity
     */
    Vec3 GetLinearVelocity() const;

    /**
     * @brief Set angular velocity (rotation speed)
     */
    void SetAngularVelocity(const Vec3& velocity);

    /**
     * @brief Get angular velocity
     */
    Vec3 GetAngularVelocity() const;

    /**
     * @brief Apply a force at the center of mass
     * @param force Force vector
     */
    void ApplyForce(const Vec3& force);

    /**
     * @brief Apply a force at a specific point in world space
     * @param force Force vector
     * @param point Point in world space
     */
    void ApplyForceAtPoint(const Vec3& force, const Vec3& point);

    /**
     * @brief Apply an impulse at the center of mass
     * @param impulse Impulse vector
     */
    void ApplyImpulse(const Vec3& impulse);

    /**
     * @brief Apply an impulse at a specific point in world space
     * @param impulse Impulse vector
     * @param point Point in world space
     */
    void ApplyImpulseAtPoint(const Vec3& impulse, const Vec3& point);

    /**
     * @brief Set the mass of the body
     */
    void SetMass(float mass);

    /**
     * @brief Get the mass of the body
     */
    float GetMass() const;

    /**
     * @brief Set linear damping (friction-like effect for linear motion)
     * @param damping Value between 0 and 1 (0 = no damping, 1 = stop immediately)
     */
    void SetLinearDamping(float damping);

    /**
     * @brief Set angular damping (friction-like effect for rotation)
     * @param damping Value between 0 and 1
     */
    void SetAngularDamping(float damping);

    /**
     * @brief Enable or disable gravity for this body
     */
    void SetGravityEnabled(bool enabled);

    /**
     * @brief Check if gravity is enabled for this body
     */
    bool IsGravityEnabled() const { return m_GravityEnabled; }

    /**
     * @brief Set the collision group and mask
     * @param group Which group this body belongs to (1 << n)
     * @param mask Which groups this body collides with
     */
    void SetCollisionFilterGroup(uint16_t group);
    void SetCollisionFilterMask(uint16_t mask);

    /**
     * @brief Get the underlying Bullet3D rigid body
     */
    btRigidBody* GetBulletRigidBody() const { return m_BtRigidBody; }

    /**
     * @brief Synchronize transform from physics world to GameObject
     * @internal Called by PhysicsSystem
     */
    void SyncTransformFromPhysics(Vec3& outPosition, Quat& outRotation) const;

    /**
     * @brief Synchronize transform from GameObject to physics world
     * @internal Called when body is kinematic
     */
    void SyncTransformToPhysics(const Vec3& position, const Quat& rotation);

    /**
     * @brief Reset velocities and forces
     */
    void ResetVelocities();

    /**
     * @brief Enable or disable this body's simulation
     */
    void SetActive(bool active);

    /**
     * @brief Check if this body is active in simulation
     */
    bool IsActive() const;

    /**
     * @brief Get friction coefficient
     */
    float GetFriction() const;

    /**
     * @brief Set friction coefficient (0 = no friction, 1+ = high friction)
     */
    void SetFriction(float friction);

    /**
     * @brief Get restitution (bounciness)
     */
    float GetRestitution() const;

    /**
     * @brief Set restitution (0 = no bounce, 1 = perfect bounce)
     */
    void SetRestitution(float restitution);

private:
    btRigidBody* m_BtRigidBody;
    btMotionState* m_MotionState;
    BodyType m_BodyType;
    float m_Mass;
    bool m_GravityEnabled;
    
    PhysicsCollisionShape m_Shape;

    // Friend class for physics system access
    friend class PhysicsSystem;
};
