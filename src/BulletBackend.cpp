#include "BulletBackend.h"

#ifdef USE_BULLET

#include "IPhysicsRigidBody.h"
#include "IPhysicsCharacterController.h"
#include <btBulletDynamicsCommon.h>
#include <iostream>

BulletBackend::BulletBackend()
    : m_DynamicsWorld(nullptr)
    , m_Broadphase(nullptr)
    , m_Dispatcher(nullptr)
    , m_ConstraintSolver(nullptr)
    , m_CollisionConfiguration(nullptr)
    , m_Initialized(false)
    , m_DebugDrawEnabled(false)
{
}

BulletBackend::~BulletBackend() {
    if (m_Initialized) {
        Shutdown();
    }
}

void BulletBackend::Initialize(const Vec3& gravity) {
    if (m_Initialized) {
        std::cerr << "BulletBackend already initialized!" << std::endl;
        return;
    }

    std::cout << "Initializing Bullet3D physics..." << std::endl;

    // Create collision configuration
    m_CollisionConfiguration = new btDefaultCollisionConfiguration();

    // Create dispatcher
    m_Dispatcher = new btCollisionDispatcher(m_CollisionConfiguration);

    // Create broadphase
    m_Broadphase = new btDbvtBroadphase();

    // Create constraint solver
    m_ConstraintSolver = new btSequentialImpulseConstraintSolver();

    // Create dynamics world
    m_DynamicsWorld = new btDiscreteDynamicsWorld(
        m_Dispatcher,
        m_Broadphase,
        m_ConstraintSolver,
        m_CollisionConfiguration
    );

    // Set gravity
    m_DynamicsWorld->setGravity(btVector3(gravity.x, gravity.y, gravity.z));

    m_Initialized = true;
    std::cout << "Bullet3D initialized successfully!" << std::endl;
}

void BulletBackend::Shutdown() {
    if (!m_Initialized) {
        return;
    }

    std::cout << "Shutting down Bullet3D..." << std::endl;

    // Clear registered components
    m_RigidBodies.clear();
    m_CharacterControllers.clear();

    // Delete dynamics world
    if (m_DynamicsWorld) {
        delete m_DynamicsWorld;
        m_DynamicsWorld = nullptr;
    }

    // Delete constraint solver
    if (m_ConstraintSolver) {
        delete m_ConstraintSolver;
        m_ConstraintSolver = nullptr;
    }

    // Delete broadphase
    if (m_Broadphase) {
        delete m_Broadphase;
        m_Broadphase = nullptr;
    }

    // Delete dispatcher
    if (m_Dispatcher) {
        delete m_Dispatcher;
        m_Dispatcher = nullptr;
    }

    // Delete collision configuration
    if (m_CollisionConfiguration) {
        delete m_CollisionConfiguration;
        m_CollisionConfiguration = nullptr;
    }

    m_Initialized = false;
    std::cout << "Bullet3D shutdown complete" << std::endl;
}

void BulletBackend::Update(float deltaTime, int subSteps) {
    if (!m_Initialized || !m_DynamicsWorld) {
        return;
    }

    // Update character controllers first
    UpdateCharacterControllers(deltaTime);

    // Step simulation
    m_DynamicsWorld->stepSimulation(deltaTime, subSteps);
}

void BulletBackend::UpdateCharacterControllers(float deltaTime) {
    for (auto* controller : m_CharacterControllers) {
        if (controller) {
            controller->Update(deltaTime, GetGravity());
        }
    }
}

void BulletBackend::SetGravity(const Vec3& gravity) {
    if (m_DynamicsWorld) {
        m_DynamicsWorld->setGravity(btVector3(gravity.x, gravity.y, gravity.z));
    }
}

Vec3 BulletBackend::GetGravity() const {
    if (m_DynamicsWorld) {
        btVector3 gravity = m_DynamicsWorld->getGravity();
        return Vec3(gravity.x(), gravity.y(), gravity.z());
    }
    return Vec3(0, -9.81f, 0);
}

bool BulletBackend::Raycast(const Vec3& from, const Vec3& to, RaycastHit& hit, uint32_t filter) {
    if (!m_DynamicsWorld) {
        return false;
    }

    btVector3 btFrom(from.x, from.y, from.z);
    btVector3 btTo(to.x, to.y, to.z);

    btCollisionWorld::ClosestRayResultCallback rayCallback(btFrom, btTo);
    rayCallback.m_collisionFilterMask = filter;

    m_DynamicsWorld->rayTest(btFrom, btTo, rayCallback);

    if (rayCallback.hasHit()) {
        hit.point = Vec3(
            rayCallback.m_hitPointWorld.x(),
            rayCallback.m_hitPointWorld.y(),
            rayCallback.m_hitPointWorld.z()
        );
        hit.normal = Vec3(
            rayCallback.m_hitNormalWorld.x(),
            rayCallback.m_hitNormalWorld.y(),
            rayCallback.m_hitNormalWorld.z()
        );
        hit.distance = (from - hit.point).Length();
        hit.userData = rayCallback.m_collisionObject;
        return true;
    }

    return false;
}

int BulletBackend::GetNumRigidBodies() const {
    return static_cast<int>(m_RigidBodies.size());
}

void BulletBackend::SetDebugDrawEnabled(bool enabled) {
    m_DebugDrawEnabled = enabled;
}

bool BulletBackend::IsDebugDrawEnabled() const {
    return m_DebugDrawEnabled;
}

void* BulletBackend::GetNativeWorld() {
    return m_DynamicsWorld;
}

void BulletBackend::RegisterRigidBody(IPhysicsRigidBody* body) {
    if (body) {
        m_RigidBodies.push_back(body);
    }
}

void BulletBackend::UnregisterRigidBody(IPhysicsRigidBody* body) {
    auto it = std::find(m_RigidBodies.begin(), m_RigidBodies.end(), body);
    if (it != m_RigidBodies.end()) {
        m_RigidBodies.erase(it);
    }
}

void BulletBackend::RegisterCharacterController(IPhysicsCharacterController* controller) {
    if (controller) {
        m_CharacterControllers.push_back(controller);
    }
}

void BulletBackend::UnregisterCharacterController(IPhysicsCharacterController* controller) {
    auto it = std::find(m_CharacterControllers.begin(), m_CharacterControllers.end(), controller);
    if (it != m_CharacterControllers.end()) {
        m_CharacterControllers.erase(it);
    }
}

#endif // USE_BULLET
