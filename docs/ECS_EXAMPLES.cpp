/**
 * @file ECS_EXAMPLES.cpp
 * @brief Practical examples of using the ECS system
 * 
 * This file contains code examples demonstrating various ECS patterns.
 * Not meant to be compiled directly; copy patterns as needed.
 */

#include "ECS.h"
#include <iostream>

// ============================================================================
// EXAMPLE 1: Creating a Simple Entity
// ============================================================================

void Example_CreateEntity(EntityManager& manager) {
    // Create an entity
    Entity cube = manager.CreateEntity();
    
    // Add a transform
    auto& transform = manager.AddComponent<TransformComponent>(cube);
    transform.SetPosition(Vec3(5.0f, 2.0f, 0.0f));
    
    // Add a mesh for rendering
    auto& mesh = manager.AddComponent<MeshComponent>(cube);
    mesh.SetMeshID(0);  // Assuming 0 is a valid mesh ID
    
    std::cout << "Created entity with ID: " << cube.GetID() << std::endl;
}

// ============================================================================
// EXAMPLE 2: Querying Entities
// ============================================================================

void Example_QueryEntities(EntityManager& manager) {
    // Get all entities with Transform and Mesh components
    auto meshEntities = manager.GetEntitiesWithComponents<TransformComponent, MeshComponent>();
    
    std::cout << "Found " << meshEntities.size() << " renderable entities" << std::endl;
    
    for (const auto& entity : meshEntities) {
        auto* transform = manager.GetComponent<TransformComponent>(entity);
        auto* mesh = manager.GetComponent<MeshComponent>(entity);
        
        Vec3 pos = transform->GetPosition();
        std::cout << "  Entity " << entity.GetID() << " at (" 
                  << pos.x << ", " << pos.y << ", " << pos.z << ")" << std::endl;
    }
}

// ============================================================================
// EXAMPLE 3: Creating a Custom Component
// ============================================================================

class ScoreComponent : public ComponentBase<ScoreComponent> {
public:
    ScoreComponent(int initialScore = 0)
        : m_Score(initialScore), m_Multiplier(1.0f) {}

    void AddPoints(int points) {
        m_Score += static_cast<int>(points * m_Multiplier);
    }

    void SetMultiplier(float mult) { m_Multiplier = mult; }
    int GetScore() const { return m_Score; }

    void OnEnable() override {
        std::cout << "Score tracking enabled" << std::endl;
    }

    void OnDisable() override {
        std::cout << "Score tracking disabled. Final score: " << m_Score << std::endl;
    }

private:
    int m_Score;
    float m_Multiplier;
};

void Example_CustomComponent(EntityManager& manager) {
    Entity player = manager.CreateEntity();
    
    auto& score = manager.AddComponent<ScoreComponent>(player, 0);
    score.AddPoints(100);
    score.SetMultiplier(1.5f);
    score.AddPoints(50);  // Actually adds 75
    
    std::cout << "Player score: " << score.GetScore() << std::endl;
}

// ============================================================================
// EXAMPLE 4: Creating a Custom System
// ============================================================================

class RotationSystem : public System {
public:
    void SetRotationSpeed(float speed) { m_RotationSpeed = speed; }

    void Update(EntityManager& manager, float deltaTime) override {
        // Rotate all entities with transforms
        auto entities = manager.GetEntitiesWithComponents<TransformComponent>();
        
        for (const auto& entity : entities) {
            auto* transform = manager.GetComponent<TransformComponent>(entity);
            
            // Get current rotation
            Vec3 euler = transform->GetEulerAngles();
            
            // Apply rotation around Y axis
            euler.y += m_RotationSpeed * deltaTime;
            
            // Set new rotation
            transform->SetEulerAngles(euler);
        }
    }

private:
    float m_RotationSpeed = 90.0f;  // degrees per second
};

void Example_CustomSystem(EntityManager& manager) {
    // Add the rotation system
    auto& rotSystem = manager.AddSystem<RotationSystem>();
    rotSystem.SetRotationSpeed(180.0f);  // Rotate 180 degrees per second
    rotSystem.SetPriority(10);  // Execute early in the frame
}

// ============================================================================
// EXAMPLE 5: Player and Enemy Setup
// ============================================================================

void Example_GameEntities(EntityManager& manager) {
    // === Create Player ===
    Entity player = manager.CreateEntity();
    
    // Player transform
    auto& playerTransform = manager.AddComponent<TransformComponent>(player);
    playerTransform.SetPosition(Vec3(0.0f, 1.0f, 0.0f));
    
    // Player mesh
    auto& playerMesh = manager.AddComponent<MeshComponent>(player);
    playerMesh.SetMeshID(0);  // Player mesh ID
    
    // Player health
    auto& playerHealth = manager.AddComponent<HealthComponent>(player, 100.0f);
    
    // Player movement
    auto& playerMovement = manager.AddComponent<MovementComponent>(player);
    playerMovement.SetMoveSpeed(Vec3(8.0f, 0.0f, 8.0f));
    
    // Player collision
    auto& playerCollider = manager.AddComponent<ColliderComponent>(player);
    playerCollider.SetShape(ColliderComponent::Shape::Capsule);
    playerCollider.SetSize(Vec3(0.5f, 2.0f, 0.5f));
    
    // Tag player
    manager.AddComponent<TagComponent<struct PlayerTag>>(player);
    
    // === Create Enemy ===
    Entity enemy = manager.CreateEntity();
    
    // Enemy transform
    auto& enemyTransform = manager.AddComponent<TransformComponent>(enemy);
    enemyTransform.SetPosition(Vec3(10.0f, 1.0f, 10.0f));
    
    // Enemy mesh
    auto& enemyMesh = manager.AddComponent<MeshComponent>(enemy);
    enemyMesh.SetMeshID(1);  // Enemy mesh ID
    
    // Enemy health
    auto& enemyHealth = manager.AddComponent<HealthComponent>(enemy, 50.0f);
    
    // Enemy AI
    auto& enemyAI = manager.AddComponent<AIComponent>(enemy);
    enemyAI.SetDetectionRange(25.0f);
    enemyAI.SetState(AIComponent::AIState::Patrolling);
    enemyAI.SetPatrolTarget(Vec3(15.0f, 1.0f, 10.0f));
    
    // Enemy movement
    auto& enemyMovement = manager.AddComponent<MovementComponent>(enemy);
    enemyMovement.SetMoveSpeed(Vec3(4.0f, 0.0f, 4.0f));
    
    // Enemy collision
    auto& enemyCollider = manager.AddComponent<ColliderComponent>(enemy);
    enemyCollider.SetShape(ColliderComponent::Shape::Capsule);
    enemyCollider.SetSize(Vec3(0.4f, 2.0f, 0.4f));
    
    // Enemy damage
    auto& enemyDamage = manager.AddComponent<DamageComponent>(enemy, 15.0f);
    
    // Tag enemy
    manager.AddComponent<TagComponent<struct EnemyTag>>(enemy);
    
    // Tell AI system about player
    if (auto* aiSystem = manager.GetSystem<SimpleAISystem>()) {
        aiSystem->SetPlayerEntity(player);
    }
    
    std::cout << "Created player (ID: " << player.GetID() << ") and enemy (ID: " 
              << enemy.GetID() << ")" << std::endl;
}

// ============================================================================
// EXAMPLE 6: Projectile System
// ============================================================================

void Example_SpawnProjectile(EntityManager& manager, Entity shooter, 
                            const Vec3& shooterPos, const Vec3& direction) {
    Entity projectile = manager.CreateEntity();
    
    // Transform
    auto& transform = manager.AddComponent<TransformComponent>(projectile);
    transform.SetPosition(shooterPos);
    
    // Velocity (projectile flies in direction)
    auto& velocity = manager.AddComponent<VelocityComponent>(projectile);
    velocity.SetVelocity(direction * 20.0f);  // 20 m/s
    
    // Collision
    auto& collider = manager.AddComponent<ColliderComponent>(projectile);
    collider.SetShape(ColliderComponent::Shape::Sphere);
    collider.SetSize(Vec3(0.2f, 0.2f, 0.2f));
    collider.SetIsTrigger(true);
    
    // Damage
    auto& damage = manager.AddComponent<DamageComponent>(projectile, 25.0f);
    
    // Auto-destroy after 10 seconds
    manager.AddComponent<LifetimeComponent>(projectile, 10.0f);
    
    std::cout << "Spawned projectile from " << shooter.GetID() << std::endl;
}

// ============================================================================
// EXAMPLE 7: Event Handling with Systems
// ============================================================================

class GameEventSystem : public System {
public:
    using OnPlayerDamaged = std::function<void(Entity, float)>;
    using OnEnemyDefeated = std::function<void(Entity)>;
    
    void SetOnPlayerDamaged(OnPlayerDamaged callback) {
        m_OnPlayerDamaged = callback;
    }
    
    void SetOnEnemyDefeated(OnEnemyDefeated callback) {
        m_OnEnemyDefeated = callback;
    }

    void Update(EntityManager& manager, float deltaTime) override {
        // Check for dead enemies
        auto enemies = manager.GetEntitiesWithComponents<TagComponent<struct EnemyTag>, HealthComponent>();
        
        for (const auto& enemy : enemies) {
            auto* health = manager.GetComponent<HealthComponent>(enemy);
            
            if (health && health->IsDead()) {
                if (m_OnEnemyDefeated) {
                    m_OnEnemyDefeated(enemy);
                }
            }
        }
    }

private:
    OnPlayerDamaged m_OnPlayerDamaged;
    OnEnemyDefeated m_OnEnemyDefeated;
};

void Example_EventHandling(EntityManager& manager) {
    auto& eventSystem = manager.AddSystem<GameEventSystem>();
    
    eventSystem.SetOnEnemyDefeated([&manager](Entity enemy) {
        std::cout << "Enemy defeated! ID: " << enemy.GetID() << std::endl;
        // Could spawn loot, add score, play sound, etc.
    });
}

// ============================================================================
// EXAMPLE 8: Entity Pooling Pattern
// ============================================================================

class EntityPool {
public:
    EntityPool(EntityManager& manager, size_t poolSize)
        : m_Manager(manager), m_PoolSize(poolSize) {}

    Entity AcquireEntity() {
        if (m_AvailableEntities.empty()) {
            // Create new entity if pool is empty
            return m_Manager.CreateEntity();
        }
        
        Entity entity = m_AvailableEntities.back();
        m_AvailableEntities.pop_back();
        return entity;
    }

    void ReleaseEntity(Entity entity) {
        // Clear all components
        m_Manager.ClearComponents(entity);
        m_AvailableEntities.push_back(entity);
    }

private:
    EntityManager& m_Manager;
    size_t m_PoolSize;
    std::vector<Entity> m_AvailableEntities;
};

void Example_EntityPooling(EntityManager& manager) {
    EntityPool projectilePool(manager, 100);
    
    // Acquire a projectile from pool
    Entity projectile = projectilePool.AcquireEntity();
    
    // Use projectile
    auto& velocity = manager.AddComponent<VelocityComponent>(projectile);
    velocity.SetVelocity(Vec3(10, 0, 10));
    
    // ... later ...
    
    // Return to pool
    projectilePool.ReleaseEntity(projectile);
}

// ============================================================================
// EXAMPLE 9: System with Priorities
// ============================================================================

void Example_SystemPriorities(EntityManager& manager) {
    // Physics should update before rendering
    auto& physicsSystem = manager.AddSystem<PhysicsSystem>();
    physicsSystem.SetPriority(100);  // Runs first
    
    auto& transformSystem = manager.AddSystem<TransformSystem>();
    transformSystem.SetPriority(90);  // Runs second
    
    auto& renderSystem = manager.AddSystem<RenderSystem>();
    renderSystem.SetPriority(0);  // Runs last
}

// ============================================================================
// EXAMPLE 10: Conditional System Execution
// ============================================================================

void Example_ConditionalSystems(EntityManager& manager) {
    // Create a system
    auto* aiSystem = manager.GetSystem<SimpleAISystem>();
    
    if (aiSystem) {
        // Pause AI (system won't update)
        aiSystem->SetEnabled(false);
        
        // ... some time later ...
        
        // Resume AI
        aiSystem->SetEnabled(true);
    }
}

// ============================================================================
// EXAMPLE 11: Complete Game Loop Setup
// ============================================================================

void Example_GameLoopSetup(EntityManager& manager) {
    // Initialize systems in order of execution
    manager.AddSystem<TransformSystem>()->SetPriority(100);
    manager.AddSystem<PhysicsSystem>()->SetPriority(90);
    manager.AddSystem<MovementSystem>()->SetPriority(85);
    manager.AddSystem<SimpleAISystem>()->SetPriority(80);
    manager.AddSystem<CollisionSystem>()->SetPriority(70);
    manager.AddSystem<CombatSystem>()->SetPriority(60);
    manager.AddSystem<HealthSystem>()->SetPriority(50);
    manager.AddSystem<LifetimeSystem>()->SetPriority(40);
    manager.AddSystem<RenderSystem>()->SetPriority(0);
    
    // Create player
    Entity player = manager.CreateEntity();
    manager.AddComponent<TransformComponent>(player);
    manager.AddComponent<MeshComponent>(player);
    manager.AddComponent<HealthComponent>(player, 100.0f);
    manager.AddComponent<TagComponent<struct PlayerTag>>(player);
    
    // Create some enemies
    for (int i = 0; i < 5; ++i) {
        Entity enemy = manager.CreateEntity();
        manager.AddComponent<TransformComponent>(enemy);
        manager.AddComponent<MeshComponent>(enemy);
        manager.AddComponent<AIComponent>(enemy);
        manager.AddComponent<HealthComponent>(enemy, 50.0f);
        manager.AddComponent<TagComponent<struct EnemyTag>>(enemy);
    }
    
    std::cout << "Game setup complete. Entities: " << manager.GetEntityCount() << std::endl;
}

// ============================================================================
// Main Execution Examples
// ============================================================================

int main_example() {
    EntityManager manager;
    
    // Initialize systems
    manager.AddSystem<TransformSystem>();
    manager.AddSystem<PhysicsSystem>();
    manager.AddSystem<RenderSystem>();
    
    // Run examples
    Example_CreateEntity(manager);
    Example_CustomComponent(manager);
    Example_GameEntities(manager);
    
    // Game loop
    float deltaTime = 0.016f;  // ~60 FPS
    for (int frame = 0; frame < 100; ++frame) {
        // Update all systems
        manager.Update(deltaTime);
        
        // Query and log
        if (frame % 30 == 0) {
            auto entities = manager.GetEntitiesWithComponents<TransformComponent>();
            std::cout << "Frame " << frame << ": " << entities.size() << " entities" << std::endl;
        }
    }
    
    // Cleanup
    manager.Clear();
    
    return 0;
}

// ============================================================================
// End of Examples
// ============================================================================

/*
 * To use these examples:
 * 
 * 1. Copy the function bodies into your code
 * 2. Replace EntityManager references with m_EntityManager->
 * 3. Adjust include paths and type IDs as needed
 * 4. Compile and test
 * 
 * Key patterns to remember:
 * - Always query entities with the components you need
 * - Use GetComponent() to access components
 * - Systems should be independent and reusable
 * - Use tags to categorize entities
 * - Set system priorities for correct execution order
 */
