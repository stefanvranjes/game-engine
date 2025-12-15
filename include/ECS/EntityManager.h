#pragma once

#include "Entity.h"
#include "Component.h"
#include "System.h"
#include <vector>
#include <unordered_map>
#include <memory>
#include <typeindex>
#include <queue>
#include <algorithm>
#include <cassert>

/**
 * @brief Central manager for the Entity-Component-System architecture.
 * 
 * The EntityManager:
 * - Creates and destroys entities
 * - Manages component attachment/detachment
 * - Manages and updates systems
 * - Provides queries for entities with specific components
 * 
 * Thread-safety: Not thread-safe. Call from a single game thread.
 */
class EntityManager {
public:
    EntityManager();
    ~EntityManager();

    // === Entity Management ===

    /**
     * @brief Create a new entity.
     * @return The newly created entity
     */
    Entity CreateEntity();

    /**
     * @brief Destroy an entity and all its components.
     * @param entity The entity to destroy
     */
    void DestroyEntity(Entity entity);

    /**
     * @brief Check if an entity is valid and alive.
     * @param entity The entity to check
     * @return True if the entity exists
     */
    bool IsEntityValid(Entity entity) const;

    /**
     * @brief Get all active entities.
     * @return Vector of all valid entities
     */
    const std::vector<Entity>& GetAllEntities() const { return m_Entities; }

    // === Component Management ===

    /**
     * @brief Add a component to an entity.
     * 
     * @tparam T The component type
     * @tparam Args Constructor argument types
     * @param entity The entity to add the component to
     * @param args Arguments to pass to the component constructor
     * @return Reference to the newly added component
     */
    template<typename T, typename... Args>
    T& AddComponent(Entity entity, Args&&... args) {
        static_assert(std::is_base_of_v<Component, T>, "T must derive from Component");
        assert(IsEntityValid(entity) && "Entity must be valid");

        auto& componentMap = m_EntityComponents[entity.GetID()];
        auto typeIndex = GetComponentTypeIndex<T>();

        // Check if component already exists
        assert(componentMap.find(typeIndex) == componentMap.end() && 
               "Component of this type already exists on entity");

        // Create and store the component
        auto component = std::make_shared<T>(std::forward<Args>(args)...);
        componentMap[typeIndex] = component;

        // Call OnEnable
        component->OnEnable();

        // Notify systems
        for (auto& system : m_Systems) {
            system->OnComponentAdded(entity, typeIndex);
        }

        return *component;
    }

    /**
     * @brief Get a component from an entity.
     * 
     * @tparam T The component type to retrieve
     * @param entity The entity to get the component from
     * @return Pointer to the component, or nullptr if not found
     */
    template<typename T>
    T* GetComponent(Entity entity) const {
        static_assert(std::is_base_of_v<Component, T>, "T must derive from Component");

        auto it = m_EntityComponents.find(entity.GetID());
        if (it == m_EntityComponents.end()) return nullptr;

        auto compIt = it->second.find(GetComponentTypeIndex<T>());
        if (compIt == it->second.end()) return nullptr;

        return dynamic_cast<T*>(compIt->second.get());
    }

    /**
     * @brief Check if an entity has a component of a specific type.
     * 
     * @tparam T The component type to check for
     * @param entity The entity to check
     * @return True if the entity has the component
     */
    template<typename T>
    bool HasComponent(Entity entity) const {
        return GetComponent<T>(entity) != nullptr;
    }

    /**
     * @brief Remove a component from an entity.
     * 
     * @tparam T The component type to remove
     * @param entity The entity to remove the component from
     */
    template<typename T>
    void RemoveComponent(Entity entity) {
        static_assert(std::is_base_of_v<Component, T>, "T must derive from Component");

        auto it = m_EntityComponents.find(entity.GetID());
        if (it == m_EntityComponents.end()) return;

        auto typeIndex = GetComponentTypeIndex<T>();
        auto compIt = it->second.find(typeIndex);
        if (compIt == it->second.end()) return;

        // Call OnDisable before removal
        compIt->second->OnDisable();

        // Notify systems
        for (auto& system : m_Systems) {
            system->OnComponentRemoved(entity, typeIndex);
        }

        it->second.erase(compIt);
    }

    /**
     * @brief Remove all components from an entity.
     * @param entity The entity to clear
     */
    void ClearComponents(Entity entity);

    // === System Management ===

    /**
     * @brief Add a system to the manager.
     * 
     * @tparam T The system type
     * @tparam Args Constructor argument types
     * @param args Arguments to pass to the system constructor
     * @return Reference to the newly added system
     */
    template<typename T, typename... Args>
    T& AddSystem(Args&&... args) {
        static_assert(std::is_base_of_v<System, T>, "T must derive from System");

        auto system = std::make_shared<T>(std::forward<Args>(args)...);
        system->OnInitialize();
        m_Systems.push_back(system);

        // Sort systems by priority (higher priority = earlier execution)
        std::sort(m_Systems.begin(), m_Systems.end(),
                  [](const auto& a, const auto& b) {
                      return a->GetPriority() > b->GetPriority();
                  });

        return *system;
    }

    /**
     * @brief Get a system by type.
     * 
     * @tparam T The system type to retrieve
     * @return Pointer to the system, or nullptr if not found
     */
    template<typename T>
    T* GetSystem() const {
        static_assert(std::is_base_of_v<System, T>, "T must derive from System");

        for (const auto& system : m_Systems) {
            auto typed = dynamic_cast<T*>(system.get());
            if (typed) return typed;
        }
        return nullptr;
    }

    /**
     * @brief Remove a system by type.
     * 
     * @tparam T The system type to remove
     */
    template<typename T>
    void RemoveSystem() {
        static_assert(std::is_base_of_v<System, T>, "T must derive from System");

        auto it = std::find_if(m_Systems.begin(), m_Systems.end(),
                               [](const auto& system) {
                                   return dynamic_cast<T*>(system.get()) != nullptr;
                               });

        if (it != m_Systems.end()) {
            (*it)->OnShutdown();
            m_Systems.erase(it);
        }
    }

    /**
     * @brief Update all systems.
     * @param deltaTime Time elapsed since last frame in seconds
     */
    void Update(float deltaTime);

    // === Entity Queries ===

    /**
     * @brief Get all entities that have specific components.
     * 
     * @tparam Components The component types to query for
     * @return Vector of entities that have all specified components
     */
    template<typename... Components>
    std::vector<Entity> GetEntitiesWithComponents() const {
        std::vector<Entity> result;

        if constexpr (sizeof...(Components) == 0) {
            // No components specified, return all entities
            return m_Entities;
        }

        // Collect type indices
        std::vector<std::type_index> requiredTypes{
            GetComponentTypeIndex<Components>()...
        };

        // Check each entity
        for (const auto& entity : m_Entities) {
            auto it = m_EntityComponents.find(entity.GetID());
            if (it == m_EntityComponents.end()) continue;

            bool hasAll = true;
            for (const auto& typeIdx : requiredTypes) {
                if (it->second.find(typeIdx) == it->second.end()) {
                    hasAll = false;
                    break;
                }
            }

            if (hasAll) {
                result.push_back(entity);
            }
        }

        return result;
    }

    /**
     * @brief Get the number of active entities.
     * @return Number of entities
     */
    size_t GetEntityCount() const { return m_Entities.size(); }

    /**
     * @brief Clear all entities and systems.
     */
    void Clear();

private:
    using ComponentMap = std::unordered_map<std::type_index, std::shared_ptr<Component>>;

    Entity::ID m_NextEntityID = 0;
    std::vector<Entity> m_Entities;
    std::unordered_map<Entity::ID, ComponentMap> m_EntityComponents;
    std::vector<std::shared_ptr<System>> m_Systems;
};
