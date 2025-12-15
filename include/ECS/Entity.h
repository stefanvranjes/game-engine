#pragma once

#include <cstdint>
#include <functional>

/**
 * @brief Entity represents a unique game object in the ECS.
 * 
 * Each entity is identified by a unique ID. Entities themselves don't contain
 * data; instead, they're containers for components. The ECS Manager tracks
 * which components are attached to each entity.
 */
class Entity {
public:
    using ID = uint32_t;
    static constexpr ID INVALID_ID = ~0u;

    Entity() : m_ID(INVALID_ID) {}
    explicit Entity(ID id) : m_ID(id) {}

    // Comparison operators for use in containers
    bool operator==(const Entity& other) const { return m_ID == other.m_ID; }
    bool operator!=(const Entity& other) const { return m_ID != other.m_ID; }
    bool operator<(const Entity& other) const { return m_ID < other.m_ID; }

    // Check validity
    bool IsValid() const { return m_ID != INVALID_ID; }

    // Get the ID
    ID GetID() const { return m_ID; }

private:
    ID m_ID;

    friend class EntityManager;
    friend struct std::hash<Entity>;
};

// Hash function for Entity to use in unordered containers
namespace std {
    template<>
    struct hash<Entity> {
        std::size_t operator()(const Entity& e) const {
            return std::hash<uint32_t>()(e.m_ID);
        }
    };
}
