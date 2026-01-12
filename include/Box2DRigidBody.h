#pragma once

#include "IPhysicsRigidBody.h"
#include <box2d/box2d.h>

#ifdef USE_BOX2D

class Box2DBackend;

class Box2DRigidBody : public IPhysicsRigidBody {
public:
    Box2DRigidBody(Box2DBackend* backend);
    ~Box2DRigidBody() override;

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
    
    void SetOnCollisionCallback(OnCollisionCallback callback) override { m_CollisionCallback = callback; }
    void SetOnTriggerEnterCallback(OnTriggerCallback callback) override {}
    void SetOnTriggerExitCallback(OnTriggerCallback callback) override {}
    
    void SetUserData(void* data) override { 
        m_UserData = data; 
        if(m_Body) {
             m_Body->GetUserData().pointer = reinterpret_cast<uintptr_t>(data);
        }
    }
    void* GetUserData() const override { return m_UserData; }
    
    void SetCCDEnabled(bool enabled) override;
    bool IsCCDEnabled() const override;
    
    void SetTriggerVelocityThreshold(float threshold) override {}
    float GetTriggerVelocityThreshold() const override { return 0.0f; }
    float GetTriggerDwellTime(IPhysicsRigidBody* otherBody) const override { return 0.0f; }
    
    void* GetNativeBody() override { return m_Body; }

private:
    Box2DBackend* m_Backend;
    b2Body* m_Body;
    void* m_UserData = nullptr;
    
    BodyType m_Type;
    float m_Mass;
    
    OnCollisionCallback m_CollisionCallback;
};

#endif // USE_BOX2D
