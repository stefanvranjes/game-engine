#pragma once

#ifdef USE_PHYSX

#include <vector>
#include <memory>
#include <PxPhysicsAPI.h>

class PhysXBackend;
class PhysXRigidBody;

/**
 * @brief Wrapper around physx::PxAggregate
 * 
 * Aggregates group actors together into a single broad-phase entry.
 * This is efficient for ragdolls, debris, and other clusters of actors.
 */
class PhysXAggregate {
public:
    PhysXAggregate(PhysXBackend* backend, int maxActors, bool selfCollisions = true);
    ~PhysXAggregate();

    /**
     * @brief Adds a rigid body to the aggregate.
     * 
     * If the body is currently in a scene, it will be removed from the scene
     * and added to the aggregate (which effectively adds it back to the scene
     * if the aggregate is in the scene).
     */
    bool AddBody(std::shared_ptr<PhysXRigidBody> body);

    /**
     * @brief Removes a rigid body from the aggregate.
     * 
     * The body remains in the scene but as an individual actor.
     */
    bool RemoveBody(std::shared_ptr<PhysXRigidBody> body);

    /**
     * @brief Adds the aggregate to the physics scene.
     */
    void AddToScene();

    /**
     * @brief Removes the aggregate from the physics scene.
     */
    void RemoveFromScene();

    physx::PxAggregate* GetNativeAggregate() const { return m_Aggregate; }
    int GetMaxActors() const { return m_MaxActors; }
    bool IsSelfCollisionEnabled() const { return m_SelfCollisions; }
    int GetNumActors() const;

private:
    PhysXBackend* m_Backend;
    physx::PxAggregate* m_Aggregate;
    int m_MaxActors;
    bool m_SelfCollisions;
    bool m_InScene;

    // Keep track of added bodies to prevent destruction
    std::vector<std::shared_ptr<PhysXRigidBody>> m_Bodies;
};

#endif // USE_PHYSX
