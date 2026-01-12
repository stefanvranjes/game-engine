#pragma once

#include "IPhysicsRigidBody.h"
#include "Math/Vec3.h"
#include "Math/Quat.h"
#include <memory>

#ifdef USE_PHYSX

namespace physx {
    class PxRigidActor;
    class PxRigidDynamic;
    class PxRigidStatic;
    class PxShape;
}

class PhysXBackend;

/**
 * @brief PhysX implementation of rigid body physics
 */
class PhysXRigidBody : public IPhysicsRigidBody {
public:
    PhysXRigidBody(PhysXBackend* backend);
    ~PhysXRigidBody() override;

    // IPhysicsRigidBody implementation
    void Initialize(BodyType type, float mass, std::shared_ptr<IPhysicsShape> shape) override;
    void SetLinearVelocity(const Vec3& velocity) override;
    Vec3 GetLinearVelocity() const override;
    void SetAngularVelocity(const Vec3& velocity) override;
    Vec3 GetAngularVelocity() const override;
    void ApplyForce(const Vec3& force) override;
    void ApplyImpulse(const Vec3& impulse) override;
    void ApplyForceAtPoint(const Vec3& force, const Vec3& point) override;
    void ApplyImpulseAtPoint(const Vec3& impulse, const Vec3& point) override;
    void ApplyTorque(const Vec3& torque) override;
    void SetMass(float mass) override;
    float GetMass() const override;
    void SetFriction(float friction) override;
    float GetFriction() const override;
    void SetRestitution(float restitution) override;
    float GetRestitution() const override;
    void SetLinearDamping(float damping) override;
    void SetAngularDamping(float damping) override;
    void SetGravityEnabled(bool enabled) override;
    bool IsGravityEnabled() const override;
    bool IsActive() const override;
    void SetActive(bool active) override;
    void SyncTransformFromPhysics(Vec3& outPosition, Quat& outRotation) override;
    void SyncTransformToPhysics(const Vec3& position, const Quat& rotation) override;
    void SetOnCollisionCallback(OnCollisionCallback callback) override;
    void SetUserData(void* data) override { m_UserData = data; }
    void* GetUserData() const override { return m_UserData; }
    void* GetNativeBody() override;

    // Internal callback handler
    void HandleCollision(const CollisionInfo& info);

private:
    PhysXBackend* m_Backend;
    physx::PxRigidActor* m_Actor;
    physx::PxRigidDynamic* m_DynamicActor; // Non-null for dynamic bodies
    BodyType m_BodyType;
    float m_Mass;
    bool m_GravityEnabled;
    OnCollisionCallback m_CollisionCallback;
    void* m_UserData = nullptr;
};

#endif // USE_PHYSX
