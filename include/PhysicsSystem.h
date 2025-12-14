#pragma once

#include "Math/Vec3.h"
#include "Math/Quat.h"
#include <memory>
#include <vector>
#include <functional>

// Forward declarations
class btDynamicsWorld;
class btBroadphaseInterface;
class btCollisionDispatcher;
class btConstraintSolver;
class btDefaultCollisionConfiguration;
class btRigidBody;
class RigidBody;
class KinematicController;

/**
 * @brief Physics simulation system using Bullet3D
 * 
 * Manages the physics world, rigid bodies, and gravity.
 * Handles collision detection and response automatically.
 * 
 * Usage:
 *   PhysicsSystem physics;
 *   physics.Initialize({0, -9.81f, 0}); // gravity
 *   
 *   // Add rigid bodies through RigidBody component
 *   // Call Update(deltaTime) each frame
 *   physics.Update(deltaTime);
 */
class PhysicsSystem {
public:
    PhysicsSystem();
    ~PhysicsSystem();

    // No copying
    PhysicsSystem(const PhysicsSystem&) = delete;
    PhysicsSystem& operator=(const PhysicsSystem&) = delete;

    /**
     * @brief Initialize the physics world
     * @param gravity Gravity vector (typically {0, -9.81f, 0} for standard Earth gravity)
     */
    void Initialize(const Vec3& gravity = Vec3(0, -9.81f, 0));

    /**
     * @brief Shut down the physics world and clean up resources
     */
    void Shutdown();

    /**
     * @brief Update physics simulation for a single frame
     * @param deltaTime Time step in seconds
     * @param subSteps Number of sub-steps for fixed timestep (default: 1)
     */
    void Update(float deltaTime, int subSteps = 1);

    /**
     * @brief Get the underlying Bullet3D dynamics world
     */
    btDynamicsWorld* GetDynamicsWorld() const { return m_DynamicsWorld; }

    /**
     * @brief Set gravity
     */
    void SetGravity(const Vec3& gravity);

    /**
     * @brief Get current gravity
     */
    Vec3 GetGravity() const;

    /**
     * @brief Register a rigid body with the physics world
     * @internal Called automatically by RigidBody constructor
     */
    void RegisterRigidBody(RigidBody* rigidBody, btRigidBody* btBody);

    /**
     * @brief Unregister a rigid body from the physics world
     * @internal Called automatically by RigidBody destructor
     */
    void UnregisterRigidBody(btRigidBody* btBody);

    /**
     * @brief Register a kinematic controller
     * @internal Called automatically by KinematicController constructor
     */
    void RegisterKinematicController(KinematicController* controller);

    /**
     * @brief Unregister a kinematic controller
     * @internal Called automatically by KinematicController destructor
     */
    void UnregisterKinematicController(KinematicController* controller);

    /**
     * @brief Get the number of rigid bodies in the world
     */
    int GetNumRigidBodies() const;

    /**
     * @brief Enable or disable debug drawing
     * @param enabled If true, collision shapes will be visualized
     */
    void SetDebugDrawEnabled(bool enabled) { m_DebugDrawEnabled = enabled; }

    /**
     * @brief Check if debug drawing is enabled
     */
    bool IsDebugDrawEnabled() const { return m_DebugDrawEnabled; }

    /**
     * @brief Raycast from a point in a direction
     * @param from Start position
     * @param to End position
     * @param outHit Output hit information (hit point, normal, distance)
     * @param filter Optional collision filter mask
     * @return True if a hit was detected
     */
    bool Raycast(const Vec3& from, const Vec3& to, struct RaycastHit& outHit, uint32_t filter = ~0u);

    /**
     * @brief Get all rigid bodies in the world
     */
    const std::vector<RigidBody*>& GetRigidBodies() const { return m_RigidBodies; }

    /**
     * @brief Get singleton instance
     */
    static PhysicsSystem& Get();

private:
    btDynamicsWorld* m_DynamicsWorld;
    btBroadphaseInterface* m_Broadphase;
    btCollisionDispatcher* m_Dispatcher;
    btConstraintSolver* m_ConstraintSolver;
    btDefaultCollisionConfiguration* m_CollisionConfiguration;

    std::vector<RigidBody*> m_RigidBodies;
    std::vector<KinematicController*> m_KinematicControllers;

    bool m_DebugDrawEnabled;
    bool m_Initialized;

    // Update kinematic controllers each frame
    void UpdateKinematicControllers(float deltaTime);
};

/**
 * @brief Result of a raycast operation
 */
struct RaycastHit {
    Vec3 point;         // World space point of impact
    Vec3 normal;        // Surface normal at impact
    float distance;     // Distance from ray origin to hit point
    btRigidBody* body;  // The rigid body that was hit
};
