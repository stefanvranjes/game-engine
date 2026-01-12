#pragma once

#include "GameObject.h"
#include "IPhysicsRigidBody.h"
#include <memory>

class PhysXDestructible {
public:
    PhysXDestructible();
    ~PhysXDestructible();

    void Initialize(GameObject* owner, float health, float damageThreshold);
    void SetDestroyedPrefab(std::shared_ptr<GameObject> prefab);
    void SetExplosionForce(float force) { m_ExplosionForce = force; }
    void SetVelocityScale(float scale) { m_VelocityScale = scale; }
    
    // Core Logic
    void ApplyDamage(float amount);
    void Fracture();

private:
    void OnCollision(const IPhysicsRigidBody::CollisionInfo& info);
    void ApplyForcesToDebris(std::shared_ptr<GameObject> debrisRoot, const Vec3& parentVelocity);

    GameObject* m_Owner;
    std::shared_ptr<GameObject> m_DestroyedPrefab;
    IPhysicsRigidBody* m_RigidBody;
    
    float m_Health;
    float m_MaxHealth;
    float m_DamageThreshold;
    float m_ExplosionForce;
    float m_VelocityScale;
    bool m_IsDestroyed;
};
