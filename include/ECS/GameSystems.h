#pragma once

#include "ECS/System.h"
#include "ECS/Components.h"
#include "Math/Vec3.h"
#include <functional>

/**
 * @brief Movement component for smooth velocity-based movement.
 */
class MovementComponent : public ComponentBase<MovementComponent> {
public:
    MovementComponent(const Vec3& moveSpeed = Vec3(5.0f, 5.0f, 5.0f))
        : m_MoveSpeed(moveSpeed), m_CurrentDirection(0.0f) {}

    void SetMoveSpeed(const Vec3& speed) { m_MoveSpeed = speed; }
    Vec3 GetMoveSpeed() const { return m_MoveSpeed; }

    void SetCurrentDirection(const Vec3& dir) { m_CurrentDirection = dir; }
    Vec3 GetCurrentDirection() const { return m_CurrentDirection; }

private:
    Vec3 m_MoveSpeed;
    Vec3 m_CurrentDirection;
};

/**
 * @brief AI component for basic AI behavior.
 */
class AIComponent : public ComponentBase<AIComponent> {
public:
    enum class AIState {
        Idle,
        Patrolling,
        Chasing,
        Attacking,
        Fleeing
    };

    AIComponent() : m_State(AIState::Idle), m_DetectionRange(10.0f), 
                   m_PatrolTarget(0.0f), m_PatrolWaitTime(0.0f) {}

    void SetState(AIState state) { m_State = state; }
    AIState GetState() const { return m_State; }

    void SetDetectionRange(float range) { m_DetectionRange = range; }
    float GetDetectionRange() const { return m_DetectionRange; }

    void SetPatrolTarget(const Vec3& target) { m_PatrolTarget = target; }
    Vec3 GetPatrolTarget() const { return m_PatrolTarget; }

    void SetPatrolWaitTime(float time) { m_PatrolWaitTime = time; }
    float GetPatrolWaitTime() const { return m_PatrolWaitTime; }

private:
    AIState m_State;
    float m_DetectionRange;
    Vec3 m_PatrolTarget;
    float m_PatrolWaitTime;
};

/**
 * @brief Health component for damageable entities.
 */
class HealthComponent : public ComponentBase<HealthComponent> {
public:
    HealthComponent(float maxHealth = 100.0f)
        : m_MaxHealth(maxHealth), m_CurrentHealth(maxHealth), m_IsDead(false) {}

    void SetMaxHealth(float health) { m_MaxHealth = health; m_CurrentHealth = health; }
    float GetMaxHealth() const { return m_MaxHealth; }

    void SetHealth(float health) { m_CurrentHealth = std::clamp(health, 0.0f, m_MaxHealth); }
    float GetHealth() const { return m_CurrentHealth; }

    void TakeDamage(float damage) { 
        m_CurrentHealth = std::max(0.0f, m_CurrentHealth - damage);
        if (m_CurrentHealth <= 0.0f) m_IsDead = true;
    }

    void Heal(float amount) { 
        m_CurrentHealth = std::min(m_MaxHealth, m_CurrentHealth + amount); 
    }

    float GetHealthPercent() const { return m_CurrentHealth / m_MaxHealth; }
    bool IsDead() const { return m_IsDead; }

private:
    float m_MaxHealth;
    float m_CurrentHealth;
    bool m_IsDead;
};

/**
 * @brief Damage component for hitting other entities.
 */
class DamageComponent : public ComponentBase<DamageComponent> {
public:
    DamageComponent(float damage = 10.0f)
        : m_Damage(damage), m_LastDamageTime(0.0f), m_DamageCooldown(0.5f) {}

    void SetDamage(float damage) { m_Damage = damage; }
    float GetDamage() const { return m_Damage; }

    void SetDamageCooldown(float cooldown) { m_DamageCooldown = cooldown; }
    float GetDamageCooldown() const { return m_DamageCooldown; }

    bool CanDamage(float currentTime) const {
        return (currentTime - m_LastDamageTime) >= m_DamageCooldown;
    }

    void UpdateLastDamageTime(float currentTime) { m_LastDamageTime = currentTime; }

private:
    float m_Damage;
    float m_LastDamageTime;
    float m_DamageCooldown;
};

/**
 * @brief Basic movement system using MovementComponent.
 */
class MovementSystem : public System {
public:
    void Update(EntityManager& manager, float deltaTime) override;
};

/**
 * @brief Simple AI system for patrol and chase behavior.
 */
class SimpleAISystem : public System {
public:
    void SetPlayerEntity(Entity player) { m_PlayerEntity = player; }

    void Update(EntityManager& manager, float deltaTime) override;

private:
    Entity m_PlayerEntity;
};

/**
 * @brief Health/death system - removes dead entities or calls callbacks.
 */
class HealthSystem : public System {
public:
    using OnDeathCallback = std::function<void(Entity)>;

    void SetOnEntityDeath(OnDeathCallback callback) { m_OnDeath = callback; }

    void Update(EntityManager& manager, float deltaTime) override;

private:
    OnDeathCallback m_OnDeath;
};

/**
 * @brief Combat system - handles damage application between entities.
 */
class CombatSystem : public System {
public:
    void Update(EntityManager& manager, float deltaTime) override;
};
