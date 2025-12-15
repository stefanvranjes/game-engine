#pragma once

#include "Entity.h"
#include <vector>
#include <memory>

class EntityManager;

/**
 * @brief Base class for all ECS systems.
 * 
 * Systems contain the logic that operates on entities with specific components.
 * Each system focuses on one aspect of game logic (rendering, physics, AI, etc).
 */
class System {
public:
    virtual ~System() = default;

    /**
     * @brief Called once when the system is added to the manager.
     */
    virtual void OnInitialize() {}

    /**
     * @brief Called once per frame to update the system.
     * @param manager Reference to the EntityManager to query entities and components
     * @param deltaTime Time elapsed since last frame in seconds
     */
    virtual void Update(EntityManager& manager, float deltaTime) {}

    /**
     * @brief Called once when the system is removed from the manager.
     */
    virtual void OnShutdown() {}

    /**
     * @brief Called when a new entity is created.
     * @param entity The newly created entity
     */
    virtual void OnEntityCreated(Entity entity) {}

    /**
     * @brief Called when an entity is destroyed.
     * @param entity The entity being destroyed
     */
    virtual void OnEntityDestroyed(Entity entity) {}

    /**
     * @brief Called when a component is added to an entity.
     * @param entity The entity that received the component
     * @param typeIndex The type of component that was added
     */
    virtual void OnComponentAdded(Entity entity, std::type_index typeIndex) {}

    /**
     * @brief Called when a component is removed from an entity.
     * @param entity The entity that lost the component
     * @param typeIndex The type of component that was removed
     */
    virtual void OnComponentRemoved(Entity entity, std::type_index typeIndex) {}

    /**
     * @brief Set the execution priority of this system.
     * Higher priority systems execute first each frame.
     * @param priority Priority value (default 0)
     */
    void SetPriority(int priority) { m_Priority = priority; }
    int GetPriority() const { return m_Priority; }

    /**
     * @brief Enable/disable this system.
     * Disabled systems skip their Update() call.
     * @param enabled Whether the system should be active
     */
    void SetEnabled(bool enabled) { m_Enabled = enabled; }
    bool IsEnabled() const { return m_Enabled; }

protected:
    int m_Priority = 0;
    bool m_Enabled = true;
};
