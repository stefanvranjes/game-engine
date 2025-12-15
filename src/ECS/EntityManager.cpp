#include "ECS/EntityManager.h"
#include <algorithm>

EntityManager::EntityManager() 
    : m_NextEntityID(0) {
}

EntityManager::~EntityManager() {
    Clear();
}

Entity EntityManager::CreateEntity() {
    Entity entity(m_NextEntityID++);
    m_Entities.push_back(entity);
    
    // Notify all systems of new entity
    for (auto& system : m_Systems) {
        system->OnEntityCreated(entity);
    }
    
    return entity;
}

void EntityManager::DestroyEntity(Entity entity) {
    // Find and remove the entity
    auto it = std::find(m_Entities.begin(), m_Entities.end(), entity);
    if (it == m_Entities.end()) return;

    // Notify systems before destruction
    for (auto& system : m_Systems) {
        system->OnEntityDestroyed(entity);
    }

    // Remove all components
    ClearComponents(entity);

    // Remove entity from list
    m_Entities.erase(it);

    // Remove from component map
    m_EntityComponents.erase(entity.GetID());
}

bool EntityManager::IsEntityValid(Entity entity) const {
    return std::find(m_Entities.begin(), m_Entities.end(), entity) != m_Entities.end();
}

void EntityManager::ClearComponents(Entity entity) {
    auto it = m_EntityComponents.find(entity.GetID());
    if (it == m_EntityComponents.end()) return;

    // Call OnDisable on all components and notify systems
    for (auto& [typeIdx, component] : it->second) {
        component->OnDisable();
        for (auto& system : m_Systems) {
            system->OnComponentRemoved(entity, typeIdx);
        }
    }

    it->second.clear();
}

void EntityManager::Update(float deltaTime) {
    // Update all enabled systems in priority order
    for (auto& system : m_Systems) {
        if (system->IsEnabled()) {
            system->Update(*this, deltaTime);
        }
    }

    // Optional: Update components directly
    // This allows components to have update logic without a dedicated system
    for (const auto& entity : m_Entities) {
        auto it = m_EntityComponents.find(entity.GetID());
        if (it != m_EntityComponents.end()) {
            for (auto& [typeIdx, component] : it->second) {
                component->Update(deltaTime);
            }
        }
    }
}

void EntityManager::Clear() {
    // Shutdown all systems
    for (auto& system : m_Systems) {
        system->OnShutdown();
    }
    m_Systems.clear();

    // Destroy all entities
    std::vector<Entity> entitiesToDestroy = m_Entities;
    for (const auto& entity : entitiesToDestroy) {
        DestroyEntity(entity);
    }

    m_EntityComponents.clear();
    m_NextEntityID = 0;
}
