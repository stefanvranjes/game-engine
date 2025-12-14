#include "PhysicsSystem.h"
#include "RigidBody.h"
#include "KinematicController.h"
#include <btBulletDynamicsCommon.h>
#include <iostream>

// Static instance
static PhysicsSystem* g_PhysicsSystem = nullptr;

PhysicsSystem::PhysicsSystem()
    : m_DynamicsWorld(nullptr),
      m_Broadphase(nullptr),
      m_Dispatcher(nullptr),
      m_ConstraintSolver(nullptr),
      m_CollisionConfiguration(nullptr),
      m_DebugDrawEnabled(false),
      m_Initialized(false) {
}

PhysicsSystem::~PhysicsSystem() {
    Shutdown();
}

void PhysicsSystem::Initialize(const Vec3& gravity) {
    if (m_Initialized) {
        return; // Already initialized
    }

    // Collision configuration
    m_CollisionConfiguration = new btDefaultCollisionConfiguration();
    
    // Collision dispatcher
    m_Dispatcher = new btCollisionDispatcher(m_CollisionConfiguration);
    
    // Broadphase (spatial partitioning)
    m_Broadphase = new btDbvtBroadphase();
    
    // Constraint solver
    m_ConstraintSolver = new btSequentialImpulseConstraintSolver();
    
    // Create dynamics world
    m_DynamicsWorld = new btDiscreteDynamicsWorld(
        m_Dispatcher,
        m_Broadphase,
        m_ConstraintSolver,
        m_CollisionConfiguration
    );
    
    SetGravity(gravity);
    
    m_Initialized = true;
    
    std::cout << "[PhysicsSystem] Initialized with gravity: " << gravity.x << ", " << gravity.y << ", " << gravity.z << std::endl;
}

void PhysicsSystem::Shutdown() {
    if (!m_Initialized) {
        return;
    }

    // Remove all rigid bodies
    for (auto rigidBody : m_RigidBodies) {
        if (rigidBody && rigidBody->m_BtRigidBody) {
            m_DynamicsWorld->removeRigidBody(rigidBody->m_BtRigidBody);
        }
    }
    m_RigidBodies.clear();

    // Remove all kinematic controllers
    for (auto controller : m_KinematicControllers) {
        if (controller && controller->m_GhostObject) {
            m_DynamicsWorld->removeCollisionObject(controller->m_GhostObject);
        }
    }
    m_KinematicControllers.clear();

    // Clean up Bullet objects
    if (m_DynamicsWorld) {
        delete m_DynamicsWorld;
        m_DynamicsWorld = nullptr;
    }
    
    if (m_ConstraintSolver) {
        delete m_ConstraintSolver;
        m_ConstraintSolver = nullptr;
    }
    
    if (m_Dispatcher) {
        delete m_Dispatcher;
        m_Dispatcher = nullptr;
    }
    
    if (m_Broadphase) {
        delete m_Broadphase;
        m_Broadphase = nullptr;
    }
    
    if (m_CollisionConfiguration) {
        delete m_CollisionConfiguration;
        m_CollisionConfiguration = nullptr;
    }

    m_Initialized = false;
    std::cout << "[PhysicsSystem] Shutdown complete" << std::endl;
}

void PhysicsSystem::Update(float deltaTime, int subSteps) {
    if (!m_Initialized || !m_DynamicsWorld) {
        return;
    }

    // Update kinematic controllers before physics step
    UpdateKinematicControllers(deltaTime);

    // Step the simulation
    m_DynamicsWorld->stepSimulation(deltaTime, subSteps, 1.0f / 60.0f);
}

void PhysicsSystem::UpdateKinematicControllers(float deltaTime) {
    for (auto controller : m_KinematicControllers) {
        if (controller) {
            controller->Update(deltaTime);
        }
    }
}

void PhysicsSystem::SetGravity(const Vec3& gravity) {
    if (!m_DynamicsWorld) {
        return;
    }

    m_DynamicsWorld->setGravity(btVector3(gravity.x, gravity.y, gravity.z));
}

Vec3 PhysicsSystem::GetGravity() const {
    if (!m_DynamicsWorld) {
        return Vec3(0, 0, 0);
    }

    btVector3 gravity = m_DynamicsWorld->getGravity();
    return Vec3(gravity.x(), gravity.y(), gravity.z());
}

void PhysicsSystem::RegisterRigidBody(RigidBody* rigidBody, btRigidBody* btBody) {
    if (!m_DynamicsWorld || !btBody) {
        return;
    }

    m_DynamicsWorld->addRigidBody(btBody);
    m_RigidBodies.push_back(rigidBody);
}

void PhysicsSystem::UnregisterRigidBody(btRigidBody* btBody) {
    if (!m_DynamicsWorld || !btBody) {
        return;
    }

    m_DynamicsWorld->removeRigidBody(btBody);
    
    // Find and remove from our tracking list
    for (auto it = m_RigidBodies.begin(); it != m_RigidBodies.end(); ++it) {
        if (*it && (*it)->m_BtRigidBody == btBody) {
            m_RigidBodies.erase(it);
            break;
        }
    }
}

void PhysicsSystem::RegisterKinematicController(KinematicController* controller) {
    if (!m_DynamicsWorld || !controller || !controller->m_GhostObject) {
        return;
    }

    m_DynamicsWorld->addCollisionObject(
        controller->m_GhostObject,
        btBroadphaseProxy::CharacterFilter,
        btBroadphaseProxy::AllFilter ^ btBroadphaseProxy::CharacterFilter
    );
    m_KinematicControllers.push_back(controller);
}

void PhysicsSystem::UnregisterKinematicController(KinematicController* controller) {
    if (!m_DynamicsWorld || !controller || !controller->m_GhostObject) {
        return;
    }

    m_DynamicsWorld->removeCollisionObject(controller->m_GhostObject);
    
    // Find and remove from our tracking list
    for (auto it = m_KinematicControllers.begin(); it != m_KinematicControllers.end(); ++it) {
        if (*it == controller) {
            m_KinematicControllers.erase(it);
            break;
        }
    }
}

int PhysicsSystem::GetNumRigidBodies() const {
    return m_DynamicsWorld ? m_DynamicsWorld->getNumCollisionObjects() : 0;
}

bool PhysicsSystem::Raycast(const Vec3& from, const Vec3& to, RaycastHit& outHit, uint32_t filter) {
    if (!m_DynamicsWorld) {
        return false;
    }

    btVector3 btFrom(from.x, from.y, from.z);
    btVector3 btTo(to.x, to.y, to.z);

    btCollisionWorld::ClosestRayResultCallback rayCallback(btFrom, btTo);
    rayCallback.m_collisionFilterMask = filter;

    m_DynamicsWorld->rayTest(btFrom, btTo, rayCallback);

    if (rayCallback.hasHit()) {
        btVector3 hitPoint = rayCallback.m_hitPointWorld;
        btVector3 hitNormal = rayCallback.m_hitNormalWorld;

        outHit.point = Vec3(hitPoint.x(), hitPoint.y(), hitPoint.z());
        outHit.normal = Vec3(hitNormal.x(), hitNormal.y(), hitNormal.z());
        outHit.distance = (outHit.point - from).Length();
        outHit.body = btRigidBody::upcast(rayCallback.m_collisionObject);

        return true;
    }

    return false;
}

PhysicsSystem& PhysicsSystem::Get() {
    if (!g_PhysicsSystem) {
        g_PhysicsSystem = new PhysicsSystem();
    }
    return *g_PhysicsSystem;
}
