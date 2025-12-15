#include "ECS/GameSystems.h"
#include "ECS/EntityManager.h"
#include <cmath>

// === MovementSystem ===

void MovementSystem::Update(EntityManager& manager, float deltaTime) {
    auto entities = manager.GetEntitiesWithComponents<MovementComponent, TransformComponent>();

    for (const auto& entity : entities) {
        auto* movement = manager.GetComponent<MovementComponent>(entity);
        auto* transform = manager.GetComponent<TransformComponent>(entity);

        if (!movement || !transform) continue;

        Vec3 direction = movement->GetCurrentDirection();
        Vec3 moveSpeed = movement->GetMoveSpeed();
        
        // Apply movement
        Vec3 displacement = direction * moveSpeed * deltaTime;
        transform->Translate(displacement);
    }
}

// === SimpleAISystem ===

void SimpleAISystem::Update(EntityManager& manager, float deltaTime) {
    auto entities = manager.GetEntitiesWithComponents<AIComponent, TransformComponent>();

    // Get player position if available
    Vec3 playerPos(0.0f);
    bool hasPlayer = false;
    if (m_PlayerEntity.IsValid()) {
        auto* playerTransform = manager.GetComponent<TransformComponent>(m_PlayerEntity);
        if (playerTransform) {
            playerPos = playerTransform->GetPosition();
            hasPlayer = true;
        }
    }

    for (const auto& entity : entities) {
        auto* ai = manager.GetComponent<AIComponent>(entity);
        auto* transform = manager.GetComponent<TransformComponent>(entity);
        auto* movement = manager.GetComponent<MovementComponent>(entity);

        if (!ai || !transform) continue;

        Vec3 currentPos = transform->GetPosition();
        Vec3 moveDir(0.0f);

        // Simple state machine
        switch (ai->GetState()) {
            case AIComponent::AIState::Idle:
                moveDir = Vec3(0.0f);
                // Check if player is in range
                if (hasPlayer) {
                    float distToPlayer = (playerPos - currentPos).Length();
                    if (distToPlayer < ai->GetDetectionRange()) {
                        ai->SetState(AIComponent::AIState::Chasing);
                    }
                }
                break;

            case AIComponent::AIState::Chasing:
                if (hasPlayer) {
                    float distToPlayer = (playerPos - currentPos).Length();
                    if (distToPlayer < ai->GetDetectionRange()) {
                        // Move toward player
                        moveDir = (playerPos - currentPos).Normalized();
                    } else {
                        // Player escaped
                        ai->SetState(AIComponent::AIState::Idle);
                    }
                } else {
                    ai->SetState(AIComponent::AIState::Idle);
                }
                break;

            case AIComponent::AIState::Patrolling: {
                Vec3 patrolTarget = ai->GetPatrolTarget();
                float distToTarget = (patrolTarget - currentPos).Length();
                
                if (distToTarget > 0.1f) {
                    moveDir = (patrolTarget - currentPos).Normalized();
                } else {
                    // Reached patrol point, wait
                    moveDir = Vec3(0.0f);
                }

                // Check if player is in range while patrolling
                if (hasPlayer) {
                    float distToPlayer = (playerPos - currentPos).Length();
                    if (distToPlayer < ai->GetDetectionRange()) {
                        ai->SetState(AIComponent::AIState::Chasing);
                    }
                }
                break;
            }

            default:
                break;
        }

        // Apply movement direction
        if (movement) {
            movement->SetCurrentDirection(moveDir);
        }
    }
}

// === HealthSystem ===

void HealthSystem::Update(EntityManager& manager, float deltaTime) {
    auto entities = manager.GetEntitiesWithComponents<HealthComponent>();

    for (const auto& entity : entities) {
        auto* health = manager.GetComponent<HealthComponent>(entity);

        if (!health) continue;

        if (health->IsDead()) {
            if (m_OnDeath) {
                m_OnDeath(entity);
            }
            manager.DestroyEntity(entity);
        }
    }
}

// === CombatSystem ===

void CombatSystem::Update(EntityManager& manager, float deltaTime) {
    auto damagers = manager.GetEntitiesWithComponents<DamageComponent, ColliderComponent, TransformComponent>();
    auto targets = manager.GetEntitiesWithComponents<HealthComponent, ColliderComponent, TransformComponent>();

    static float combatTimer = 0.0f;
    combatTimer += deltaTime;

    // Check collisions between damagers and targets
    for (const auto& damageEntity : damagers) {
        auto* damageComp = manager.GetComponent<DamageComponent>(damageEntity);
        auto* damageCollider = manager.GetComponent<ColliderComponent>(damageEntity);
        auto* damageTransform = manager.GetComponent<TransformComponent>(damageEntity);

        if (!damageComp || !damageCollider || !damageTransform) continue;

        Vec3 damagePos = damageTransform->GetPosition();
        Vec3 damageSize = damageCollider->GetSize();

        for (const auto& healthEntity : targets) {
            auto* health = manager.GetComponent<HealthComponent>(healthEntity);
            auto* targetCollider = manager.GetComponent<ColliderComponent>(healthEntity);
            auto* targetTransform = manager.GetComponent<TransformComponent>(healthEntity);

            if (!health || !targetCollider || !targetTransform) continue;

            Vec3 targetPos = targetTransform->GetPosition();
            Vec3 targetSize = targetCollider->GetSize();

            // AABB collision check
            bool colliding = 
                std::abs(damagePos.x - targetPos.x) < (damageSize.x + targetSize.x) * 0.5f &&
                std::abs(damagePos.y - targetPos.y) < (damageSize.y + targetSize.y) * 0.5f &&
                std::abs(damagePos.z - targetPos.z) < (damageSize.z + targetSize.z) * 0.5f;

            if (colliding && damageComp->CanDamage(combatTimer)) {
                health->TakeDamage(damageComp->GetDamage());
                damageComp->UpdateLastDamageTime(combatTimer);
            }
        }
    }
}
