#pragma once

#include "IPhysicsRigidBody.h"
#include "PhysXArticulationJoint.h"
#include <vector>

#ifdef USE_PHYSX
namespace physx {
    class PxArticulationLink;
    class PxShape;
}

class PhysXArticulation;

class PhysXArticulationLink : public IPhysicsRigidBody {
public:
    PhysXArticulationLink(physx::PxArticulationLink* link, PhysXArticulation* articulation);
    ~PhysXArticulationLink() override;

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
    void* GetNativeBody() override;

    // Articulation-specific
    PhysXArticulationJoint* GetInboundJoint() const;
    std::vector<PhysXArticulationLink*> GetChildren() const;
    PhysXArticulation* GetArticulation() const { return m_Articulation; }

    // Helper for collision callbacks
    void HandleCollision(const CollisionInfo& info);

private:
    physx::PxArticulationLink* m_Link;
    PhysXArticulation* m_Articulation;
    PhysXArticulationJoint* m_InboundJoint;
    mutable std::vector<PhysXArticulationLink*> m_ChildrenCached; // Cache children if needed, or query dynamically
    
    OnCollisionCallback m_CollisionCallback;
    BodyType m_BodyType;
};
#endif // USE_PHYSX
