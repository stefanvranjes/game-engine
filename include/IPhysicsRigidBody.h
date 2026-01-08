#pragma once

#include "Math/Vec3.h"
#include "Math/Quat.h"
#include "IPhysicsShape.h"
#include <memory>

/**
 * @brief Body type enumeration
 */
enum class BodyType {
    Static,     // Fixed in place, infinite mass
    Dynamic,    // Affected by forces and gravity
    Kinematic   // Moved by code, not affected by forces
};

/**
 * @brief Abstract interface for rigid body physics
 * 
 * Provides a common API for rigid body dynamics across different physics backends.
 */
class IPhysicsRigidBody {
public:
    virtual ~IPhysicsRigidBody() = default;

    /**
     * @brief Initialize the rigid body
     * @param type Body type (Static, Dynamic, Kinematic)
     * @param mass Mass in kilograms (0 for static bodies)
     * @param shape Collision shape
     */
    virtual void Initialize(BodyType type, float mass, std::shared_ptr<IPhysicsShape> shape) = 0;

    /**
     * @brief Set linear velocity
     * @param velocity Velocity vector in m/s
     */
    virtual void SetLinearVelocity(const Vec3& velocity) = 0;

    /**
     * @brief Get linear velocity
     * @return Current velocity vector
     */
    virtual Vec3 GetLinearVelocity() const = 0;

    /**
     * @brief Set angular velocity
     * @param velocity Angular velocity vector in rad/s
     */
    virtual void SetAngularVelocity(const Vec3& velocity) = 0;

    /**
     * @brief Get angular velocity
     * @return Current angular velocity vector
     */
    virtual Vec3 GetAngularVelocity() const = 0;

    /**
     * @brief Apply force at center of mass
     * @param force Force vector in Newtons
     */
    virtual void ApplyForce(const Vec3& force) = 0;

    /**
     * @brief Apply impulse at center of mass
     * @param impulse Impulse vector in Newton-seconds
     */
    virtual void ApplyImpulse(const Vec3& impulse) = 0;

    /**
     * @brief Apply force at a specific point
     * @param force Force vector in Newtons
     * @param point World space point of application
     */
    virtual void ApplyForceAtPoint(const Vec3& force, const Vec3& point) = 0;

    /**
     * @brief Apply impulse at a specific point
     * @param impulse Impulse vector in Newton-seconds
     * @param point World space point of application
     */
    virtual void ApplyImpulseAtPoint(const Vec3& impulse, const Vec3& point) = 0;

    /**
     * @brief Apply torque
     * @param torque Torque vector in Newton-meters
     */
    virtual void ApplyTorque(const Vec3& torque) = 0;

    /**
     * @brief Set mass
     * @param mass Mass in kilograms
     */
    virtual void SetMass(float mass) = 0;

    /**
     * @brief Get mass
     * @return Mass in kilograms
     */
    virtual float GetMass() const = 0;

    /**
     * @brief Set friction coefficient
     * @param friction Friction value (typically 0.0 to 1.0)
     */
    virtual void SetFriction(float friction) = 0;

    /**
     * @brief Get friction coefficient
     * @return Current friction value
     */
    virtual float GetFriction() const = 0;

    /**
     * @brief Set restitution (bounciness)
     * @param restitution Restitution value (0.0 = no bounce, 1.0 = perfect bounce)
     */
    virtual void SetRestitution(float restitution) = 0;

    /**
     * @brief Get restitution
     * @return Current restitution value
     */
    virtual float GetRestitution() const = 0;

    /**
     * @brief Set linear damping
     * @param damping Damping coefficient
     */
    virtual void SetLinearDamping(float damping) = 0;

    /**
     * @brief Set angular damping
     * @param damping Damping coefficient
     */
    virtual void SetAngularDamping(float damping) = 0;

    /**
     * @brief Enable or disable gravity for this body
     * @param enabled If true, gravity affects this body
     */
    virtual void SetGravityEnabled(bool enabled) = 0;

    /**
     * @brief Check if gravity is enabled
     * @return True if gravity affects this body
     */
    virtual bool IsGravityEnabled() const = 0;

    /**
     * @brief Check if body is active (not sleeping)
     * @return True if body is active
     */
    virtual bool IsActive() const = 0;

    /**
     * @brief Set active state
     * @param active If true, wake up the body
     */
    virtual void SetActive(bool active) = 0;

    /**
     * @brief Sync transform from physics simulation
     * @param outPosition Output position
     * @param outRotation Output rotation
     */
    virtual void SyncTransformFromPhysics(Vec3& outPosition, Quat& outRotation) = 0;

    /**
     * @brief Sync transform to physics simulation
     * @param position New position
     * @param rotation New rotation
     */
    virtual void SyncTransformToPhysics(const Vec3& position, const Quat& rotation) = 0;

    /**
     * @brief Get backend-specific rigid body pointer
     * @return Opaque pointer to native rigid body
     */
    virtual void* GetNativeBody() = 0;
};
