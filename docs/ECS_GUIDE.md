# ECS (Entity-Component-System) Architecture Guide

## Overview

The Entity-Component-System (ECS) is a powerful architectural pattern for organizing game logic in a data-driven, scalable way. This game engine includes a complete, modern C++20 ECS implementation designed for performance and ease of use.

### Key Benefits

- **Data-Oriented Design**: Components are pure data containers, enabling cache-friendly iteration
- **Separation of Concerns**: Logic is organized into systems that operate on specific component combinations
- **Scalability**: Add new gameplay features by creating new components and systems without modifying existing code
- **Composition Over Inheritance**: Build complex entities by combining simple components
- **Easy Debugging**: Inspect entity state through components, clear data flow through systems

---

## Core Concepts

### Entity

An **Entity** is a unique identifier representing a game object (player, enemy, projectile, etc.).

```cpp
Entity entity = manager.CreateEntity();
```

- Entities themselves contain **no data**
- Identified by a 32-bit ID
- Can be checked for validity: `entity.IsValid()`

### Component

A **Component** is a data container attached to an entity. Components hold state but have **no logic**.

```cpp
class MyComponent : public ComponentBase<MyComponent> {
public:
    float health = 100.0f;
    Vec3 position;
    // Just data, no behavior
};
```

Built-in components:
- `TransformComponent` - Position, rotation, scale
- `VelocityComponent` - Linear and angular velocity
- `RigidbodyComponent` - Physics body type, mass, drag
- `ColliderComponent` - Collision shapes
- `MeshComponent` - 3D mesh rendering
- `SpriteComponent` - 2D sprite rendering
- `LightComponent` - Light source properties
- `HealthComponent` - Health and damage state
- `AIComponent` - AI behavior state
- `MovementComponent` - Movement speed and direction

### System

A **System** contains logic that operates on entities with specific component combinations.

```cpp
class MySystem : public System {
public:
    void Update(EntityManager& manager, float deltaTime) override {
        // Get all entities with specific components
        auto entities = manager.GetEntitiesWithComponents<TransformComponent, VelocityComponent>();
        
        for (const auto& entity : entities) {
            auto* transform = manager.GetComponent<TransformComponent>(entity);
            auto* velocity = manager.GetComponent<VelocityComponent>(entity);
            
            // Apply velocity to transform
            transform->Translate(velocity->GetVelocity() * deltaTime);
        }
    }
};
```

Built-in systems:
- `TransformSystem` - Hierarchical transform updates
- `PhysicsSystem` - Physics simulation with gravity and drag
- `RenderSystem` - Collects renderable entities
- `CollisionSystem` - AABB collision detection
- `LifetimeSystem` - Auto-destroys expired entities
- `MovementSystem` - Velocity-based movement
- `SimpleAISystem` - Basic AI patrol/chase behavior
- `HealthSystem` - Health tracking and death
- `CombatSystem` - Damage application

---

## Usage Guide

### Creating Entities and Components

```cpp
// Create an entity
Entity player = m_EntityManager->CreateEntity();

// Add components to the entity
auto& transform = m_EntityManager->AddComponent<TransformComponent>(player);
transform.SetPosition(Vec3(0, 1, 0));

auto& velocity = m_EntityManager->AddComponent<VelocityComponent>(player);
velocity.SetVelocity(Vec3(5, 0, 0));

auto& health = m_EntityManager->AddComponent<HealthComponent>(player, 100.0f);
```

### Querying Entities

```cpp
// Get all entities with specific components
auto movableEntities = m_EntityManager->GetEntitiesWithComponents<
    TransformComponent, 
    VelocityComponent
>();

for (const auto& entity : movableEntities) {
    auto* transform = m_EntityManager->GetComponent<TransformComponent>(entity);
    auto* velocity = m_EntityManager->GetComponent<VelocityComponent>(entity);
    // Process entity...
}
```

### Checking for Components

```cpp
Entity entity = m_EntityManager->CreateEntity();

// Add a component
m_EntityManager->AddComponent<HealthComponent>(entity, 50.0f);

// Check if entity has component
if (m_EntityManager->HasComponent<HealthComponent>(entity)) {
    auto* health = m_EntityManager->GetComponent<HealthComponent>(entity);
    health->TakeDamage(10.0f);
}
```

### Removing Components

```cpp
m_EntityManager->RemoveComponent<HealthComponent>(entity);
```

### Destroying Entities

```cpp
m_EntityManager->DestroyEntity(entity);
```

---

## Creating Custom Components

### Template Method (Recommended)

```cpp
#include "ECS/Component.h"
#include "Math/Vec3.h"

class PowerUpComponent : public ComponentBase<PowerUpComponent> {
public:
    enum class Type { Shield, Speed, Damage };

    PowerUpComponent(Type type = Type::Shield, float duration = 10.0f)
        : m_Type(type), m_Duration(duration), m_ElapsedTime(0.0f) {}

    void Update(float deltaTime) override {
        m_ElapsedTime += deltaTime;
    }

    Type GetType() const { return m_Type; }
    float GetRemainingTime() const { return m_Duration - m_ElapsedTime; }
    bool IsExpired() const { return m_ElapsedTime >= m_Duration; }

    void OnEnable() override {
        // Called when component is added to entity
    }

    void OnDisable() override {
        // Called when component is removed from entity
    }

private:
    Type m_Type;
    float m_Duration;
    float m_ElapsedTime;
};
```

### Inheritance Method (For Complex Base Classes)

```cpp
class BaseWeaponComponent : public Component {
public:
    virtual std::type_index GetTypeIndex() const override {
        return std::type_index(typeid(*this));
    }
    
    virtual float GetDamage() const = 0;
};

class SwordComponent : public BaseWeaponComponent {
public:
    float GetDamage() const override { return 25.0f; }
};
```

---

## Creating Custom Systems

### Basic System

```cpp
#include "ECS/System.h"

class CameraFollowSystem : public System {
public:
    void SetTargetEntity(Entity target) { m_TargetEntity = target; }

    void Update(EntityManager& manager, float deltaTime) override {
        if (!m_TargetEntity.IsValid()) return;

        // Get target transform
        auto* targetTransform = manager.GetComponent<TransformComponent>(m_TargetEntity);
        if (!targetTransform) return;

        Vec3 targetPos = targetTransform->GetPosition();
        
        // Smooth camera follow logic here
    }

private:
    Entity m_TargetEntity;
};
```

### System with Callbacks

```cpp
class EventSystem : public System {
public:
    using EventCallback = std::function<void(Entity, const std::string&)>;
    
    void SetEventCallback(EventCallback callback) { m_Callback = callback; }

    void Update(EntityManager& manager, float deltaTime) override {
        auto entities = manager.GetEntitiesWithComponents<EventComponent>();
        
        for (const auto& entity : entities) {
            auto* event = manager.GetComponent<EventComponent>(entity);
            if (event->HasEvent()) {
                if (m_Callback) {
                    m_Callback(entity, event->GetEventName());
                }
            }
        }
    }

    void OnComponentAdded(Entity entity, std::type_index typeIndex) override {
        // Called when a component is added
        if (typeIndex == std::type_index(typeid(SomeComponent))) {
            // React to SomeComponent being added
        }
    }

    void OnComponentRemoved(Entity entity, std::type_index typeIndex) override {
        // Called when a component is removed
    }

private:
    EventCallback m_Callback;
};
```

### System Priorities

Systems execute in priority order (highest first):

```cpp
// Create systems with priorities
auto& physicsSystem = m_EntityManager->AddSystem<PhysicsSystem>();
physicsSystem.SetPriority(100);  // Runs first

auto& renderSystem = m_EntityManager->AddSystem<RenderSystem>();
renderSystem.SetPriority(0);     // Runs last
```

---

## Practical Examples

### Example 1: Creating a Player Entity

```cpp
// Create player
Entity player = m_EntityManager->CreateEntity();

// Add transform
auto& transform = m_EntityManager->AddComponent<TransformComponent>(player);
transform.SetPosition(Vec3(0, 1, 0));

// Add velocity for movement
auto& velocity = m_EntityManager->AddComponent<VelocityComponent>(player);

// Add collision
auto& collider = m_EntityManager->AddComponent<ColliderComponent>(player);
collider.SetShape(ColliderComponent::Shape::Capsule);
collider.SetSize(Vec3(0.5f, 2.0f, 0.5f));

// Add health
auto& health = m_EntityManager->AddComponent<HealthComponent>(player, 100.0f);

// Add mesh for rendering
auto& mesh = m_EntityManager->AddComponent<MeshComponent>(player);
mesh.SetMeshID(playerMeshID);
```

### Example 2: Enemy with AI

```cpp
Entity enemy = m_EntityManager->CreateEntity();

// Transform and physics
auto& transform = m_EntityManager->AddComponent<TransformComponent>(enemy);
transform.SetPosition(Vec3(10, 1, 0));

auto& rigidbody = m_EntityManager->AddComponent<RigidbodyComponent>(
    enemy, RigidbodyComponent::BodyType::Dynamic, 1.0f
);

// AI behavior
auto& ai = m_EntityManager->AddComponent<AIComponent>(enemy);
ai.SetDetectionRange(20.0f);
ai.SetState(AIComponent::AIState::Patrolling);
ai.SetPatrolTarget(Vec3(15, 1, 0));

// Movement
auto& movement = m_EntityManager->AddComponent<MovementComponent>(enemy);
movement.SetMoveSpeed(Vec3(3, 0, 3));

// Combat
auto& health = m_EntityManager->AddComponent<HealthComponent>(enemy, 50.0f);
auto& damage = m_EntityManager->AddComponent<DamageComponent>(enemy, 15.0f);

// Tell AI system about the player
auto* aiSystem = m_EntityManager->GetSystem<SimpleAISystem>();
if (aiSystem) {
    aiSystem->SetPlayerEntity(player);
}
```

### Example 3: Projectile with Lifetime

```cpp
Entity projectile = m_EntityManager->CreateEntity();

auto& transform = m_EntityManager->AddComponent<TransformComponent>(projectile);
transform.SetPosition(playerPos);

auto& velocity = m_EntityManager->AddComponent<VelocityComponent>(projectile);
velocity.SetVelocity(direction * 20.0f);  // 20 m/s

auto& collider = m_EntityManager->AddComponent<ColliderComponent>(projectile);
collider.SetShape(ColliderComponent::Shape::Sphere);
collider.SetSize(Vec3(0.2f, 0.2f, 0.2f));
collider.SetIsTrigger(true);

auto& damage = m_EntityManager->AddComponent<DamageComponent>(projectile, 25.0f);

// Auto-destroy after 5 seconds
auto& lifetime = m_EntityManager->AddComponent<LifetimeComponent>(projectile, 5.0f);
```

### Example 4: Pickup Item

```cpp
class PickupComponent : public ComponentBase<PickupComponent> {
public:
    enum class Type { HealthPotion, Ammo, PowerUp };
    
    PickupComponent(Type type) : m_Type(type), m_Collected(false) {}
    
    Type GetType() const { return m_Type; }
    bool IsCollected() const { return m_Collected; }
    void Collect() { m_Collected = true; }
    
private:
    Type m_Type;
    bool m_Collected;
};

// Create pickup
Entity pickup = m_EntityManager->CreateEntity();

auto& transform = m_EntityManager->AddComponent<TransformComponent>(pickup);
transform.SetPosition(Vec3(5, 0.5f, 5));

auto& collider = m_EntityManager->AddComponent<ColliderComponent>(pickup);
collider.SetShape(ColliderComponent::Shape::Sphere);
collider.SetIsTrigger(true);  // Trigger colliders don't block movement

auto& pickupComp = m_EntityManager->AddComponent<PickupComponent>(pickup, 
    PickupComponent::Type::HealthPotion);
```

---

## Performance Considerations

### Data Layout
- Components are stored in contiguous maps per entity for cache efficiency
- Systems iterate over entities with specific component combinations
- Queries are O(n) where n = number of entities

### Optimization Tips

1. **System Priority**: Place frequent, simple systems first
2. **Component Queries**: Use specific component combinations to reduce iteration count
3. **Disable Unused Systems**: Set `system->SetEnabled(false)` for inactive systems
4. **Component Pooling**: Consider pooling frequently created/destroyed entities
5. **Batch Operations**: Process multiple entities together to maximize cache hits

### Memory Usage

- Each entity: ~8 bytes (ID)
- Each component: sizeof(component) + 16 bytes (overhead)
- Queries are temporary vectors; they don't persist

---

## Integration with Existing Systems

The ECS is integrated into `Application` and runs before other engine updates:

```cpp
// In Application::Update()
m_EntityManager->Update(deltaTime);

// Physics system updates
m_PhysicsSystem->Update(deltaTime);

// Renderer updates
m_Renderer->Update(deltaTime);
```

You can mix traditional GameObject-based rendering with ECS:

```cpp
// Get ECS entities with mesh components
auto renderables = m_EntityManager->GetEntitiesWithComponents<MeshComponent>();

for (const auto& entity : renderables) {
    auto* transform = m_EntityManager->GetComponent<TransformComponent>(entity);
    auto* mesh = m_EntityManager->GetComponent<MeshComponent>(entity);
    
    // Render using existing renderer
    m_Renderer->RenderEntity(transform, mesh);
}
```

---

## Best Practices

1. **Keep Components Data-Only**: No game logic in components
2. **Systems Should Be Independent**: Systems shouldn't directly call each other
3. **Use Tags for Categories**: Use `TagComponent<PlayerTag>` to mark entity types
4. **Cache Component Pointers**: In systems, cache pointers from `GetComponent()`
5. **Avoid Unbounded Queries**: Queries in tight loops should use specific component sets
6. **Use Callbacks for Events**: Systems can trigger callbacks without direct coupling

### Example: Using Tags

```cpp
// Define tag types
struct PlayerTag {};
struct EnemyTag {};
struct BossTag {};

// Mark entities with tags
m_EntityManager->AddComponent<TagComponent<PlayerTag>>(playerEntity);
m_EntityManager->AddComponent<TagComponent<EnemyTag>>(enemyEntity);

// Query by tag
auto players = m_EntityManager->GetEntitiesWithComponents<
    TagComponent<PlayerTag>,
    TransformComponent
>();

auto enemies = m_EntityManager->GetEntitiesWithComponents<
    TagComponent<EnemyTag>,
    AIComponent
>();
```

---

## Troubleshooting

### Entity Not Updating

**Problem**: Component changes aren't reflected

**Solution**: 
- Ensure you're using the returned reference or pointer from `GetComponent()`
- Call `component->MarkDirty()` if you modified transform properties

### System Not Called

**Problem**: System's `Update()` method isn't executing

**Solution**:
- Check if system is enabled: `system->IsEnabled()`
- Verify system was added: `manager.GetSystem<MySystem>()`
- Check entity count: `manager.GetEntityCount()`

### Memory Leak

**Problem**: Entities aren't destroyed

**Solution**:
- Call `DestroyEntity()` explicitly
- Use `LifetimeComponent` for automatic cleanup
- Clear manager on shutdown: `m_EntityManager->Clear()`

---

## API Reference

### EntityManager

```cpp
// Entity management
Entity CreateEntity();
void DestroyEntity(Entity entity);
bool IsEntityValid(Entity entity) const;
size_t GetEntityCount() const;
const std::vector<Entity>& GetAllEntities() const;

// Component management
template<typename T, typename... Args>
T& AddComponent(Entity entity, Args&&... args);

template<typename T>
T* GetComponent(Entity entity) const;

template<typename T>
bool HasComponent(Entity entity) const;

template<typename T>
void RemoveComponent(Entity entity);

void ClearComponents(Entity entity);

// System management
template<typename T, typename... Args>
T& AddSystem(Args&&... args);

template<typename T>
T* GetSystem() const;

template<typename T>
void RemoveSystem();

// Query
template<typename... Components>
std::vector<Entity> GetEntitiesWithComponents() const;

// Main update
void Update(float deltaTime);
void Clear();
```

### Component

```cpp
class Component {
public:
    virtual void OnEnable() {}
    virtual void OnDisable() {}
    virtual void Update(float deltaTime) {}
    virtual std::type_index GetTypeIndex() const = 0;
};

template<typename Derived>
class ComponentBase : public Component {
    // CRTP implementation
};
```

### System

```cpp
class System {
public:
    virtual void OnInitialize() {}
    virtual void Update(EntityManager& manager, float deltaTime) {}
    virtual void OnShutdown() {}
    virtual void OnEntityCreated(Entity entity) {}
    virtual void OnEntityDestroyed(Entity entity) {}
    virtual void OnComponentAdded(Entity entity, std::type_index typeIndex) {}
    virtual void OnComponentRemoved(Entity entity, std::type_index typeIndex) {}
    
    void SetPriority(int priority);
    int GetPriority() const;
    void SetEnabled(bool enabled);
    bool IsEnabled() const;
};
```

---

## Future Extensions

Potential enhancements to the ECS:

1. **Event System**: Pub/sub for inter-system communication
2. **Archetype-Based Storage**: Optimize for spatial locality
3. **Job System Integration**: Parallel system execution
4. **Entity Prefabs**: Reusable entity templates with component defaults
5. **Component Queries with Filters**: Advanced filtering (OR, NOT combinations)
6. **Serialization**: Save/load entity state to JSON or binary
7. **Network Replication**: Sync ECS state over network
8. **Entity Pooling**: Object pooling for high-frequency spawning

---

## References

- [Entity-Component-System Wikipedia](https://en.wikipedia.org/wiki/Entity_component_system)
- [ECS Back and Forth](https://www.gamedev.net/blogs/entry/2265481-ecs-back-and-forth/)
- [Evolving a C++ Game Engine](https://www.gamedev.net/tutorials/programming/architecture/evolving-the-general-purpose-game-engine-r3788/)

