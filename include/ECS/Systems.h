#pragma once

#include "ECS/System.h"
#include "ECS/Components.h"
#include <set>
#include <utility>

/**
 * @brief Physics system that updates entities with velocity.
 * 
 * Applies gravity and velocity to entities with both VelocityComponent
 * and RigidbodyComponent.
 */
class PhysicsSystem : public System {
public:
    void SetGravity(const Vec3& gravity) { m_Gravity = gravity; }
    Vec3 GetGravity() const { return m_Gravity; }

    void Update(EntityManager& manager, float deltaTime) override;

private:
    Vec3 m_Gravity = Vec3(0.0f, -9.81f, 0.0f);
};

/**
 * @brief Transform system for hierarchical transform updates.
 * 
 * Updates world matrices for all entities with TransformComponent,
 * considering parent-child relationships.
 */
class TransformSystem : public System {
public:
    void Update(EntityManager& manager, float deltaTime) override;
};

/**
 * @brief Basic rendering system that collects renderables.
 * 
 * Gathers all entities with MeshComponent or SpriteComponent
 * for the renderer to process.
 */
class RenderSystem : public System {
public:
    void Update(EntityManager& manager, float deltaTime) override;
    
    const std::vector<Entity>& GetMeshEntities() const { return m_MeshEntities; }
    const std::vector<Entity>& GetSpriteEntities() const { return m_SpriteEntities; }

private:
    std::vector<Entity> m_MeshEntities;
    std::vector<Entity> m_SpriteEntities;
};

/**
 * @brief Collision detection system.
 * 
 * Detects collisions between entities with ColliderComponent
 * and triggers appropriate callbacks.
 */
class CollisionSystem : public System {
public:
    using CollisionCallback = std::function<void(Entity, Entity)>;

    void SetOnCollisionEnter(CollisionCallback callback) { 
        m_OnCollisionEnter = callback; 
    }

    void SetOnCollisionExit(CollisionCallback callback) { 
        m_OnCollisionExit = callback; 
    }

    void Update(EntityManager& manager, float deltaTime) override;

private:
    CollisionCallback m_OnCollisionEnter;
    CollisionCallback m_OnCollisionExit;
    std::set<std::pair<Entity::ID, Entity::ID>> m_ActiveCollisions;
};

/**
 * @brief Lifetime system for entities with limited lifespans.
 * 
 * Automatically destroys entities when their lifetime expires.
 */
class LifetimeComponent : public ComponentBase<LifetimeComponent> {
public:
    LifetimeComponent(float lifetime) : m_Lifetime(lifetime), m_ElapsedTime(0.0f) {}

    void Update(float deltaTime) override {
        m_ElapsedTime += deltaTime;
    }

    float GetRemainingTime() const { return m_Lifetime - m_ElapsedTime; }
    float GetElapsedTime() const { return m_ElapsedTime; }
    bool IsExpired() const { return m_ElapsedTime >= m_Lifetime; }

private:
    float m_Lifetime;
    float m_ElapsedTime;
};

class LifetimeSystem : public System {
public:
    void Update(EntityManager& manager, float deltaTime) override;
};
