# ECS (Entity-Component-System) Implementation Index

## 📋 Quick Links

- **[Implementation Summary](ECS_IMPLEMENTATION_SUMMARY.md)** - Overview of what was delivered
- **[Quick Reference Guide](docs/ECS_QUICK_REFERENCE.md)** - Cheat sheet for common patterns
- **[Complete Guide](docs/ECS_GUIDE.md)** - Full documentation with examples
- **[Code Examples](docs/ECS_EXAMPLES.cpp)** - 11 practical code samples
- **[Implementation Details](docs/ECS_IMPLEMENTATION_COMPLETE.md)** - Architecture and next steps

## 🎯 Start Here

1. **New to ECS?** → Read [Quick Reference](docs/ECS_QUICK_REFERENCE.md) (5 min)
2. **Want Details?** → Read [Complete Guide](docs/ECS_GUIDE.md) (30 min)
3. **Need Examples?** → Check [Code Examples](docs/ECS_EXAMPLES.cpp)
4. **Want Architecture?** → See [Implementation Complete](docs/ECS_IMPLEMENTATION_COMPLETE.md)

## 📦 What's Included

### Core Files
```
include/ECS.h                          # Single include point
include/ECS/
├── Entity.h                           # Entity IDs
├── Component.h                        # Component base
├── System.h                           # System base
├── EntityManager.h                    # Core manager
├── Components.h                       # 8 built-in components
├── Systems.h                          # 5 core systems
└── GameSystems.h                      # Game components & systems

src/ECS/
├── EntityManager.cpp                  # Manager implementation
├── Components.cpp                     # Component implementations
└── GameSystems.cpp                    # Game system implementations
```

### Documentation
```
docs/
├── ECS_QUICK_REFERENCE.md             # Cheat sheet (⭐ Start here)
├── ECS_GUIDE.md                       # Complete guide
├── ECS_EXAMPLES.cpp                   # Code examples
└── ECS_IMPLEMENTATION_COMPLETE.md     # Details & next steps

ECS_IMPLEMENTATION_SUMMARY.md           # This implementation overview
ECS_INDEX.md                            # This file
```

## 🚀 Quick Start

### 1. Include the Header
```cpp
#include "ECS.h"
// EntityManager is m_EntityManager in Application
```

### 2. Create an Entity
```cpp
Entity player = m_EntityManager->CreateEntity();
auto& transform = m_EntityManager->AddComponent<TransformComponent>(player);
transform.SetPosition(Vec3(0, 1, 0));
```

### 3. Query Entities
```cpp
auto entities = m_EntityManager->GetEntitiesWithComponents<TransformComponent, MeshComponent>();
for (const auto& entity : entities) {
    auto* t = m_EntityManager->GetComponent<TransformComponent>(entity);
    // Process...
}
```

## 📚 Documentation Guide

| Document | Purpose | Time | For Whom |
|----------|---------|------|----------|
| [Quick Reference](docs/ECS_QUICK_REFERENCE.md) | API cheat sheet | 5 min | Everyone |
| [Complete Guide](docs/ECS_GUIDE.md) | Full documentation | 30 min | Developers |
| [Code Examples](docs/ECS_EXAMPLES.cpp) | Practical patterns | 15 min | Learners |
| [Implementation Summary](docs/ECS_IMPLEMENTATION_COMPLETE.md) | Architecture & planning | 10 min | Architects |

## 🎮 Key Concepts

### Entity
An ID representing a game object (player, enemy, projectile, etc.)
```cpp
Entity entity = m_EntityManager->CreateEntity();
```

### Component
Data-only container attached to entities
```cpp
class HealthComponent : public ComponentBase<HealthComponent> {
    float health;
};
```

### System
Logic operating on entities with specific components
```cpp
class HealthSystem : public System {
    void Update(EntityManager& manager, float deltaTime) override {
        auto entities = manager.GetEntitiesWithComponents<HealthComponent>();
        // Update health...
    }
};
```

## 💡 Common Patterns

### Create Entity with Components
```cpp
Entity player = m_EntityManager->CreateEntity();
m_EntityManager->AddComponent<TransformComponent>(player);
m_EntityManager->AddComponent<MeshComponent>(player);
m_EntityManager->AddComponent<HealthComponent>(player, 100.0f);
```

### Query and Update
```cpp
auto entities = m_EntityManager->GetEntitiesWithComponents<TransformComponent, VelocityComponent>();
for (const auto& e : entities) {
    auto* t = m_EntityManager->GetComponent<TransformComponent>(e);
    auto* v = m_EntityManager->GetComponent<VelocityComponent>(e);
    t->Translate(v->GetVelocity() * deltaTime);
}
```

### Tag Entities
```cpp
struct PlayerTag {};
m_EntityManager->AddComponent<TagComponent<PlayerTag>>(player);

auto players = m_EntityManager->GetEntitiesWithComponents<TagComponent<PlayerTag>>();
```

### Custom Component
```cpp
class PowerUpComponent : public ComponentBase<PowerUpComponent> {
    enum Type { Shield, Speed, Damage };
    PowerUpComponent(Type t) : type(t) {}
};
```

### Custom System
```cpp
class PowerUpSystem : public System {
    void Update(EntityManager& manager, float deltaTime) override {
        auto entities = manager.GetEntitiesWithComponents<PowerUpComponent>();
        // Handle powerups...
    }
};

m_EntityManager->AddSystem<PowerUpSystem>();
```

## 🔧 Built-in Components (14 total)

**Transform & Physics:**
- TransformComponent - Position, rotation, scale
- VelocityComponent - Linear and angular velocity
- RigidbodyComponent - Physics body properties
- ColliderComponent - Collision shapes

**Rendering:**
- MeshComponent - 3D mesh rendering
- SpriteComponent - 2D sprite rendering
- LightComponent - Light sources

**Gameplay:**
- HealthComponent - Health tracking
- AIComponent - AI state machine
- MovementComponent - Movement properties
- DamageComponent - Damage and cooldowns
- LifetimeComponent - Auto-destruction
- TagComponent<T> - Entity categorization

## ⚙️ Built-in Systems (9 total)

1. **TransformSystem** - Updates transforms
2. **PhysicsSystem** - Physics simulation
3. **RenderSystem** - Collects renderables
4. **CollisionSystem** - Collision detection
5. **LifetimeSystem** - Entity cleanup
6. **MovementSystem** - Movement simulation
7. **SimpleAISystem** - Basic AI
8. **HealthSystem** - Health tracking
9. **CombatSystem** - Damage application

## 📊 Architecture Overview

```
Application
    ↓
EntityManager.Update()
    ├── TransformSystem (priority 100)
    ├── PhysicsSystem (priority 90)
    ├── MovementSystem (priority 85)
    ├── SimpleAISystem (priority 80)
    ├── CollisionSystem (priority 70)
    ├── CombatSystem (priority 60)
    ├── HealthSystem (priority 50)
    ├── LifetimeSystem (priority 40)
    └── RenderSystem (priority 0)

Each System Processes:
    Entities with Required Components
        ├── Entity 0: Transform + Mesh
        ├── Entity 1: Transform + Velocity
        ├── Entity 2: Transform + AI
        └── Entity 3: Transform + Health
```

## 🎯 Typical Game Loop

```cpp
// Initialization
Entity player = m_EntityManager->CreateEntity();
m_EntityManager->AddComponent<TransformComponent>(player);
m_EntityManager->AddComponent<HealthComponent>(player, 100.0f);
// ... add more components

// Game Loop (in Application::Update)
m_EntityManager->Update(deltaTime);  // All systems process entities

// Query results when needed
auto alive = m_EntityManager->GetEntitiesWithComponents<HealthComponent>();
for (const auto& entity : alive) {
    auto* health = m_EntityManager->GetComponent<HealthComponent>(entity);
    if (health->IsDead()) {
        m_EntityManager->DestroyEntity(entity);
    }
}
```

## ✨ Features

✅ Type-safe component system with CRTP
✅ Flexible priority-based system execution
✅ Efficient multi-component queries
✅ Built-in collision, physics, AI, health systems
✅ Easy custom component/system creation
✅ Integrated with existing Application
✅ Comprehensive documentation
✅ Production-ready code

## 🔍 Troubleshooting

**Component not updating?**
- Use reference or pointer from GetComponent()
- Ensure component is on the entity

**System not running?**
- Check `system->IsEnabled()`
- Verify entity has required components

**Memory issues?**
- Call `DestroyEntity()` when done
- Use LifetimeComponent for auto-cleanup

See [ECS_GUIDE.md](docs/ECS_GUIDE.md#troubleshooting) for more.

## 📈 Performance

| Operation | Complexity |
|-----------|-----------|
| CreateEntity | O(1) |
| AddComponent | O(1) |
| GetComponent | O(1) |
| GetEntitiesWithComponents | O(n) entities |
| System Update | O(k) matching entities |

Optimizations:
- Use specific component queries to reduce iteration
- Set system priorities for correct order
- Disable unused systems

## 🚀 Next Steps

1. **Read** [Quick Reference](docs/ECS_QUICK_REFERENCE.md) (5 min)
2. **Copy** examples from [Code Examples](docs/ECS_EXAMPLES.cpp)
3. **Create** your first ECS entity and system
4. **Extend** with custom components as needed
5. **Optimize** based on profiling

## 📖 Additional Resources

- [Complete Guide](docs/ECS_GUIDE.md) - Full API documentation
- [Implementation Complete](docs/ECS_IMPLEMENTATION_COMPLETE.md) - Architecture details
- [Code Examples](docs/ECS_EXAMPLES.cpp) - 11 practical examples

## 💬 Key Takeaways

1. **ECS separates data (components) from logic (systems)**
2. **Entities are just IDs - components hold the data**
3. **Systems process entities with specific component combinations**
4. **This enables data-driven, scalable game logic**
5. **Type-safe queries compile to efficient code**

## 📞 Support

- Check [ECS_QUICK_REFERENCE.md](docs/ECS_QUICK_REFERENCE.md) for common patterns
- See [ECS_GUIDE.md](docs/ECS_GUIDE.md#troubleshooting) for troubleshooting
- Review [ECS_EXAMPLES.cpp](docs/ECS_EXAMPLES.cpp) for implementation patterns

---

**Status**: ✅ Complete & Integrated
**Documentation**: Comprehensive
**Examples**: Included
**Ready to Use**: Yes

Start with [Quick Reference](docs/ECS_QUICK_REFERENCE.md) and expand as needed!

