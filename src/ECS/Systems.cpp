#include "ECS/Systems.h"
#include "ECS/EntityManager.h"

// === SimplePhysicsSystem ===
 
void SimplePhysicsSystem::Update(EntityManager& manager, float deltaTime) {
    auto entities = manager.GetEntitiesWithComponents<VelocityComponent, RigidbodyComponent>();

    for (const auto& entity : entities) {
        auto* velocity = manager.GetComponent<VelocityComponent>(entity);
        auto* rigidbody = manager.GetComponent<RigidbodyComponent>(entity);
        auto* transform = manager.GetComponent<TransformComponent>(entity);

        if (!velocity || !rigidbody || !transform) continue;
        if (rigidbody->GetBodyType() == RigidbodyComponent::BodyType::Static) continue;

        // Apply gravity
        if (rigidbody->GetUseGravity()) {
            velocity->ApplyForce(m_Gravity * deltaTime);
        }

        // Apply drag
        float drag = 1.0f - (rigidbody->GetDrag() * deltaTime);
        drag = std::max(0.0f, drag);
        velocity->SetVelocity(velocity->GetVelocity() * drag);

        // Apply velocity to position
        transform->Translate(velocity->GetVelocity() * deltaTime);
    }
}

// === TransformSystem ===

void TransformSystem::Update(EntityManager& manager, float deltaTime) {
    auto entities = manager.GetEntitiesWithComponents<TransformComponent>();

    for (const auto& entity : entities) {
        auto* transform = manager.GetComponent<TransformComponent>(entity);
        if (transform && transform->IsDirty()) {
            transform->SetWorldMatrix(transform->GetLocalMatrix());
        }
    }
}

// === RenderSystem ===

void RenderSystem::Update(EntityManager& manager, float deltaTime) {
    m_MeshEntities = manager.GetEntitiesWithComponents<MeshComponent, TransformComponent>();
    m_SpriteEntities = manager.GetEntitiesWithComponents<SpriteComponent, TransformComponent>();
}

// === CollisionSystem ===

void CollisionSystem::Update(EntityManager& manager, float deltaTime) {
    auto colliders = manager.GetEntitiesWithComponents<ColliderComponent, TransformComponent>();

    // Simple AABB collision detection
    for (size_t i = 0; i < colliders.size(); ++i) {
        for (size_t j = i + 1; j < colliders.size(); ++j) {
            const auto& entity1 = colliders[i];
            const auto& entity2 = colliders[j];

            auto* collider1 = manager.GetComponent<ColliderComponent>(entity1);
            auto* collider2 = manager.GetComponent<ColliderComponent>(entity2);
            auto* transform1 = manager.GetComponent<TransformComponent>(entity1);
            auto* transform2 = manager.GetComponent<TransformComponent>(entity2);

            if (!collider1 || !collider2 || !transform1 || !transform2) continue;

            // Simplified AABB collision check
            Vec3 pos1 = transform1->GetPosition();
            Vec3 pos2 = transform2->GetPosition();
            Vec3 size1 = collider1->GetSize();
            Vec3 size2 = collider2->GetSize();

            bool colliding = 
                std::abs(pos1.x - pos2.x) < (size1.x + size2.x) * 0.5f &&
                std::abs(pos1.y - pos2.y) < (size1.y + size2.y) * 0.5f &&
                std::abs(pos1.z - pos2.z) < (size1.z + size2.z) * 0.5f;

            auto pairKey = std::make_pair(
                std::min(entity1.GetID(), entity2.GetID()),
                std::max(entity1.GetID(), entity2.GetID())
            );

            bool wasColliding = m_ActiveCollisions.count(pairKey) > 0;

            if (colliding && !wasColliding) {
                // Collision Enter
                m_ActiveCollisions.insert(pairKey);
                if (m_OnCollisionEnter) {
                    m_OnCollisionEnter(entity1, entity2);
                }
            } else if (!colliding && wasColliding) {
                // Collision Exit
                m_ActiveCollisions.erase(pairKey);
                if (m_OnCollisionExit) {
                    m_OnCollisionExit(entity1, entity2);
                }
            }
        }
    }
}

// === LifetimeSystem ===

void LifetimeSystem::Update(EntityManager& manager, float deltaTime) {
    auto entities = manager.GetEntitiesWithComponents<LifetimeComponent>();

    for (const auto& entity : entities) {
        auto* lifetime = manager.GetComponent<LifetimeComponent>(entity);
        if (lifetime && lifetime->IsExpired()) {
            manager.DestroyEntity(entity);
        }
    }
}
