#include "PhysXDestructible.h"
#include "Transform.h"
#include <iostream>
#include <cstdlib>

PhysXDestructible::PhysXDestructible()
    : m_Owner(nullptr)
    , m_RigidBody(nullptr)
    , m_Health(100.0f)
    , m_MaxHealth(100.0f)
    , m_DamageThreshold(1.0f)
    , m_ExplosionForce(5.0f)
    , m_VelocityScale(1.0f)
    , m_IsDestroyed(false)
{
}

PhysXDestructible::~PhysXDestructible() {
}

void PhysXDestructible::Initialize(GameObject* owner, float health, float damageThreshold) {
    m_Owner = owner;
    m_Health = health;
    m_MaxHealth = health;
    m_DamageThreshold = damageThreshold;
    m_IsDestroyed = false;

    if (m_Owner) {
        // Get Physics RigidBody (PhysX interface)
        m_RigidBody = m_Owner->GetPhysicsRigidBody().get();
        
        if (m_RigidBody) {
            // Register callback
            m_RigidBody->SetOnCollisionCallback([this](const IPhysicsRigidBody::CollisionInfo& info) {
                this->OnCollision(info);
            });
        } else {
             std::cerr << "Warning: PhysXDestructible initialized on object without PhysicsRigidBody!" << std::endl;
        }
    }
}

void PhysXDestructible::SetDestroyedPrefab(std::shared_ptr<GameObject> prefab) {
    m_DestroyedPrefab = prefab;
}

void PhysXDestructible::ApplyDamage(float amount) {
    if (m_IsDestroyed) return;

    m_Health -= amount;

    if (m_Health <= 0.0f) {
        Fracture();
    }
}

void PhysXDestructible::OnCollision(const IPhysicsRigidBody::CollisionInfo& info) {
    if (m_IsDestroyed) return;

    // Calculate damage based on impulse magnitude
    if (info.impulse > m_DamageThreshold) {
        float damage = info.impulse * 0.5f; // Scaling factor
        ApplyDamage(damage);
    }
}

void PhysXDestructible::Fracture() {
    if (m_IsDestroyed || !m_Owner) return;

    m_IsDestroyed = true;

    // Capture state before disabling
    Vec3 parentVelocity = Vec3(0,0,0);
    if (m_RigidBody) {
        parentVelocity = m_RigidBody->GetLinearVelocity();
    }

    // Spawn destroyed version
    if (m_DestroyedPrefab) {
        // Clone the prefab
        std::shared_ptr<GameObject> debris = m_DestroyedPrefab->Clone();
        
        if (debris) {
            // Match transform
            debris->GetTransform().SetPosition(m_Owner->GetTransform().GetPosition());
            debris->GetTransform().SetRotation(m_Owner->GetTransform().GetRotation());
            debris->GetTransform().SetScale(m_Owner->GetTransform().GetScale());
            
            // Add to scene graph (as sibling)
            auto parent = m_Owner->GetParent();
            if (parent) {
                parent->AddChild(debris);
            } else {
                std::cerr << "Warning: Could not spawn debris - Object has no parent!" << std::endl;
            }

            // Apply forces
            ApplyForcesToDebris(debris, parentVelocity);
        }
    }

    // Disable self
    m_Owner->SetActive(false);
    
    std::cout << "Object Fractured: " << m_Owner->GetName() << std::endl;
}

void PhysXDestructible::ApplyForcesToDebris(std::shared_ptr<GameObject> debrisRoot, const Vec3& parentVelocity) {
    if (!debrisRoot) return;

    // Helper to process a single object
    auto processObject = [&](std::shared_ptr<GameObject> obj) {
        auto rb = obj->GetPhysicsRigidBody();
        if (rb) {
            // Inherit velocity
            rb->SetLinearVelocity(parentVelocity * m_VelocityScale);

            // Apply explosion force
            if (m_ExplosionForce > 0.0f) {
                Vec3 center = m_Owner->GetTransform().GetPosition();
                Vec3 objPos = obj->GetTransform().GetPosition();
                Vec3 dir = objPos - center;
                
                // If pieces are perfectly centered, add random noise
                if (dir.LengthSquared() < 0.001f) {
                    dir = Vec3(
                        (float)(rand() % 100 - 50) / 50.0f,
                        (float)(rand() % 100 - 50) / 50.0f,
                        (float)(rand() % 100 - 50) / 50.0f
                    );
                }
                
                dir = dir.Normalized();
                rb->ApplyImpulse(dir * m_ExplosionForce);
            }
        }
    };

    // Check root
    processObject(debrisRoot);

    // Check children (direct debris pieces)
    for (auto& child : debrisRoot->GetChildren()) {
        processObject(child);
    }
}
