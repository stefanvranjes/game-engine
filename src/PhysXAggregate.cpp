#include "PhysXAggregate.h"

#ifdef USE_PHYSX

#include "PhysXBackend.h"
#include "PhysXRigidBody.h"
#include <iostream>
#include <algorithm>

using namespace physx;

PhysXAggregate::PhysXAggregate(PhysXBackend* backend, int maxActors, bool selfCollisions)
    : m_Backend(backend)
    , m_Aggregate(nullptr)
    , m_MaxActors(maxActors)
    , m_SelfCollisions(selfCollisions)
    , m_InScene(false)
{
    if (m_Backend && m_Backend->GetPhysics()) {
        m_Aggregate = m_Backend->GetPhysics()->createAggregate(maxActors, selfCollisions);
    } else {
        std::cerr << "PhysXAggregate: Backend or Physics SDK invalid!" << std::endl;
    }
}

PhysXAggregate::~PhysXAggregate() {
    RemoveFromScene();
    
    // Release aggregate
    if (m_Aggregate) {
        m_Aggregate->release();
        m_Aggregate = nullptr;
    }
    
    m_Bodies.clear();
}

bool PhysXAggregate::AddBody(std::shared_ptr<PhysXRigidBody> body) {
    if (!m_Aggregate || !body) {
        return false;
    }

    if (m_Bodies.size() >= m_MaxActors) {
        std::cerr << "PhysXAggregate: Maximum actors reached (" << m_MaxActors << ")" << std::endl;
        return false;
    }

    // Check if already added
    auto it = std::find(m_Bodies.begin(), m_Bodies.end(), body);
    if (it != m_Bodies.end()) {
        return false;
    }

    PxRigidActor* actor = static_cast<PxRigidActor*>(body->GetNativeBody());
    if (!actor) {
        return false;
    }

    // Critical: If actor is already in a scene, we must remove it first!
    PxScene* scene = actor->getScene();
    if (scene) {
        scene->removeActor(*actor);
    }

    // Add to aggregate
    if (!m_Aggregate->addActor(*actor)) {
        std::cerr << "PhysXAggregate: Failed to add actor to aggregate!" << std::endl;
        // Restore to scene if it was in one?
        // Ideally we assume m_Backend->GetScene() is the target scene
        if (scene) {
            scene->addActor(*actor);
        }
        return false;
    }

    m_Bodies.push_back(body);
    return true;
}

bool PhysXAggregate::RemoveBody(std::shared_ptr<PhysXRigidBody> body) {
    if (!m_Aggregate || !body) {
        return false;
    }

    auto it = std::find(m_Bodies.begin(), m_Bodies.end(), body);
    if (it == m_Bodies.end()) {
        return false;
    }

    PxRigidActor* actor = static_cast<PxRigidActor*>(body->GetNativeBody());
    if (actor) {
        m_Aggregate->removeActor(*actor);
        
        // If the aggregate is in the scene, removing the actor removes it from the scene too?
        // PhysX docs say: "When an actor is removed from an aggregate, it is removed from the scene as well."
        // So we might want to add it back to the scene as a standalone actor if it was active.
        if (m_InScene && m_Backend->GetScene()) {
             m_Backend->GetScene()->addActor(*actor);
        }
    }

    m_Bodies.erase(it);
    return true;
}

void PhysXAggregate::AddToScene() {
    if (m_InScene || !m_Aggregate || !m_Backend->GetScene()) {
        return;
    }

    m_Backend->GetScene()->addAggregate(*m_Aggregate);
    m_InScene = true;
}

void PhysXAggregate::RemoveFromScene() {
    if (!m_InScene || !m_Aggregate || !m_Backend->GetScene()) {
        return;
    }

    m_Backend->GetScene()->removeAggregate(*m_Aggregate);
    m_InScene = false;
}

int PhysXAggregate::GetNumActors() const {
    if (!m_Aggregate) return 0;
    return m_Aggregate->getNbActors();
}

#endif // USE_PHYSX
