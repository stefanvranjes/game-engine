# ECS Quick Reference

## Include Header

```cpp
#include "ECS.h"
```

## Create Entity and Add Components

```cpp
Entity entity = m_EntityManager->CreateEntity();
auto& transform = m_EntityManager->AddComponent<TransformComponent>(entity);
auto& health = m_EntityManager->AddComponent<HealthComponent>(entity, 100.0f);
```

## Query Entities

```cpp
auto entities = m_EntityManager->GetEntitiesWithComponents<TransformComponent, VelocityComponent>();
for (const auto& entity : entities) {
    auto* transform = m_EntityManager->GetComponent<TransformComponent>(entity);
    auto* velocity = m_EntityManager->GetComponent<VelocityComponent>(entity);
}
```

## Check and Get Component

```cpp
if (m_EntityManager->HasComponent<HealthComponent>(entity)) {
    auto* health = m_EntityManager->GetComponent<HealthComponent>(entity);
    health->TakeDamage(10.0f);
}
```

## Create Custom Component

```cpp
class MyComponent : public ComponentBase<MyComponent> {
public:
    MyComponent(float value = 0.0f) : m_Value(value) {}
    void SetValue(float v) { m_Value = v; }
    float GetValue() const { return m_Value; }
private:
    float m_Value;
};
```

## Create Custom System

```cpp
class MySystem : public System {
public:
    void Update(EntityManager& manager, float deltaTime) override {
        auto entities = manager.GetEntitiesWithComponents<MyComponent, TransformComponent>();
        for (const auto& entity : entities) {
            // Process entity
        }
    }
};
```

## Register System

```cpp
m_EntityManager->AddSystem<MySystem>();
auto* system = m_EntityManager->GetSystem<MySystem>();
system->SetPriority(10);
system->SetEnabled(true);
```

## Component Lifecycle

```cpp
class MyComponent : public ComponentBase<MyComponent> {
    void OnEnable() override {
        // Called when added to entity
    }
    
    void OnDisable() override {
        // Called when removed from entity
    }
    
    void Update(float deltaTime) override {
        // Called each frame
    }
};
```

## System Lifecycle

```cpp
class MySystem : public System {
    void OnInitialize() override {
        // Called once when added to manager
    }
    
    void Update(EntityManager& manager, float deltaTime) override {
        // Called each frame
    }
    
    void OnShutdown() override {
        // Called when removed from manager
    }
    
    void OnEntityCreated(Entity entity) override {}
    void OnEntityDestroyed(Entity entity) override {}
    void OnComponentAdded(Entity entity, std::type_index typeIndex) override {}
    void OnComponentRemoved(Entity entity, std::type_index typeIndex) override {}
};
```

## Built-in Components

- `TransformComponent` - Position, rotation, scale
- `VelocityComponent` - Linear and angular velocity
- `RigidbodyComponent` - Physics body configuration
- `ColliderComponent` - Collision shape and trigger flag
- `MeshComponent` - 3D mesh rendering
- `SpriteComponent` - 2D sprite rendering
- `LightComponent` - Light source properties
- `HealthComponent` - Health and damage tracking
- `AIComponent` - AI state machine
- `MovementComponent` - Movement speed and direction
- `DamageComponent` - Damage output and cooldown
- `LifetimeComponent` - Auto-destroy on expiration
- `TagComponent<T>` - Type-safe entity tags

## Built-in Systems

- `TransformSystem` - Updates hierarchical transforms
- `PhysicsSystem` - Physics simulation
- `RenderSystem` - Collects renderable entities
- `CollisionSystem` - AABB collision detection
- `LifetimeSystem` - Destroys expired entities
- `MovementSystem` - Applies movement
- `SimpleAISystem` - Basic AI behavior
- `HealthSystem` - Health tracking
- `CombatSystem` - Damage application

## Common Patterns

### Tag Entities
```cpp
struct PlayerTag {};
m_EntityManager->AddComponent<TagComponent<PlayerTag>>(entity);

auto players = m_EntityManager->GetEntitiesWithComponents<
    TagComponent<PlayerTag>, 
    TransformComponent
>();
```

### Automatic Cleanup
```cpp
// Entity destroyed after 5 seconds
m_EntityManager->AddComponent<LifetimeComponent>(entity, 5.0f);
```

### Physics-Based Movement
```cpp
auto& velocity = m_EntityManager->AddComponent<VelocityComponent>(entity);
auto& rigidbody = m_EntityManager->AddComponent<RigidbodyComponent>(
    entity, RigidbodyComponent::BodyType::Dynamic
);
```

### Collision Detection
```cpp
auto& collider = m_EntityManager->AddComponent<ColliderComponent>(entity);
collider.SetShape(ColliderComponent::Shape::Box);
collider.SetSize(Vec3(1, 1, 1));

// For trigger colliders
collider.SetIsTrigger(true);
```

### Combat System
```cpp
auto& health = m_EntityManager->AddComponent<HealthComponent>(entity, 100.0f);
auto& damage = m_EntityManager->AddComponent<DamageComponent>(entity, 25.0f);
auto& collider = m_EntityManager->AddComponent<ColliderComponent>(entity);
```

### AI Behavior
```cpp
auto& ai = m_EntityManager->AddComponent<AIComponent>(entity);
ai.SetState(AIComponent::AIState::Patrolling);
ai.SetDetectionRange(20.0f);
ai.SetPatrolTarget(Vec3(10, 0, 10));
```
