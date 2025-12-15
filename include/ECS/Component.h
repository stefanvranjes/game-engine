#pragma once

#include <cstdint>
#include <typeinfo>
#include <typeindex>
#include <memory>

/**
 * @brief Base class for all components in the ECS.
 * 
 * Components are data-only objects that store state for entities.
 * Derived component classes should contain only data, not logic.
 */
class Component {
public:
    virtual ~Component() = default;

    /**
     * @brief Called when the component is added to an entity.
     */
    virtual void OnEnable() {}

    /**
     * @brief Called when the component is removed from an entity.
     */
    virtual void OnDisable() {}

    /**
     * @brief Optional update callback. Systems can call this or implement custom update logic.
     * @param deltaTime Time elapsed since last frame in seconds
     */
    virtual void Update(float deltaTime) {}

    /**
     * @brief Get the type index of this component for fast type comparisons.
     * @return std::type_index representing the actual component type
     */
    virtual std::type_index GetTypeIndex() const = 0;
};

/**
 * @brief Template helper to get the type index for a component type.
 */
template<typename T>
inline std::type_index GetComponentTypeIndex() {
    return std::type_index(typeid(T));
}

/**
 * @brief CRTP helper to simplify component type identification.
 * 
 * Derive your components from Component via this class for automatic
 * type index implementation:
 * 
 * class MyComponent : public ComponentBase<MyComponent> { ... };
 */
template<typename Derived>
class ComponentBase : public Component {
public:
    std::type_index GetTypeIndex() const override {
        return std::type_index(typeid(Derived));
    }
};
