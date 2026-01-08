#pragma once

#include "Math/Vec3.h"
#include "Math/Quat.h"
#include <memory>

// Forward declarations
class IPhysicsRigidBody;
class IPhysicsCharacterController;
class IPhysicsShape;

/**
 * @brief Raycast hit result structure
 */
struct RaycastHit {
    Vec3 point;         // World space point of impact
    Vec3 normal;        // Surface normal at impact
    float distance;     // Distance from ray origin to hit point
    void* userData;     // Backend-specific rigid body pointer
};

/**
 * @brief Abstract interface for physics backend implementations
 * 
 * This interface allows the engine to support multiple physics engines
 * (Bullet3D, PhysX, etc.) through a common API.
 */
class IPhysicsBackend {
public:
    virtual ~IPhysicsBackend() = default;

    /**
     * @brief Initialize the physics world
     * @param gravity Gravity vector (e.g., {0, -9.81f, 0})
     */
    virtual void Initialize(const Vec3& gravity) = 0;

    /**
     * @brief Shut down the physics world and clean up resources
     */
    virtual void Shutdown() = 0;

    /**
     * @brief Update physics simulation for a single frame
     * @param deltaTime Time step in seconds
     * @param subSteps Number of sub-steps for fixed timestep
     */
    virtual void Update(float deltaTime, int subSteps = 1) = 0;

    /**
     * @brief Set gravity
     * @param gravity New gravity vector
     */
    virtual void SetGravity(const Vec3& gravity) = 0;

    /**
     * @brief Get current gravity
     * @return Current gravity vector
     */
    virtual Vec3 GetGravity() const = 0;

    /**
     * @brief Perform a raycast
     * @param from Start position
     * @param to End position
     * @param hit Output hit information
     * @param filter Optional collision filter mask
     * @return True if a hit was detected
     */
    virtual bool Raycast(const Vec3& from, const Vec3& to, RaycastHit& hit, uint32_t filter = ~0u) = 0;

    /**
     * @brief Get the number of rigid bodies in the world
     * @return Number of rigid bodies
     */
    virtual int GetNumRigidBodies() const = 0;

    /**
     * @brief Enable or disable debug drawing
     * @param enabled If true, collision shapes will be visualized
     */
    virtual void SetDebugDrawEnabled(bool enabled) = 0;

    /**
     * @brief Check if debug drawing is enabled
     * @return True if debug drawing is enabled
     */
    virtual bool IsDebugDrawEnabled() const = 0;

    /**
     * @brief Get backend name (e.g., "Bullet3D", "PhysX")
     * @return Backend name string
     */
    virtual const char* GetBackendName() const = 0;

    /**
     * @brief Get backend-specific world pointer (for advanced users)
     * @return Opaque pointer to backend world
     */
    virtual void* GetNativeWorld() = 0;
};
