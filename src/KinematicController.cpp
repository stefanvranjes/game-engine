#include "KinematicController.h"
#include "PhysicsSystem.h"
#include <btBulletDynamicsCommon.h>
#include <BulletDynamics/Character/btKinematicCharacterController.h>

KinematicController::KinematicController()
    : m_Controller(nullptr),
      m_GhostObject(nullptr),
      m_WalkDirection(0, 0, 0),
      m_VerticalVelocity(0),
      m_MaxWalkSpeed(10.0f),
      m_FallSpeed(55.0f),
      m_Mass(1.0f) {
}

KinematicController::~KinematicController() {
    if (m_Controller) {
        PhysicsSystem::Get().UnregisterKinematicController(this);
        delete m_Controller;
        m_Controller = nullptr;
    }
    
    if (m_GhostObject) {
        delete m_GhostObject;
        m_GhostObject = nullptr;
    }
}

void KinematicController::Initialize(const PhysicsCollisionShape& shape, float mass, float stepHeight) {
    if (m_Controller) {
        return; // Already initialized
    }

    m_Mass = mass;
    m_Shape = shape;

    // Create ghost object for collision detection
    m_GhostObject = new btPairCachingGhostObject();
    m_GhostObject->setCollisionShape(shape.GetShape());
    m_GhostObject->setCollisionFlags(btCollisionObject::CF_CHARACTER_OBJECT);

    // Create kinematic character controller
    m_Controller = new btKinematicCharacterController(m_GhostObject, shape.GetShape(), stepHeight);
    
    // Set gravity effect
    m_Controller->setUseGhostSweepTest(true);
    m_Controller->setGravity(PhysicsSystem::Get().GetGravity());

    // Register with physics system
    PhysicsSystem::Get().RegisterKinematicController(this);
}

void KinematicController::Update(float deltaTime, const Vec3& gravity) {
    if (!m_Controller || !m_GhostObject) {
        return;
    }

    // Set gravity on controller
    m_Controller->setGravity(gravity);

    // Apply walk direction
    if (m_WalkDirection.Length() > 0.001f) {
        btVector3 walkDir(m_WalkDirection.x, 0, m_WalkDirection.z);
        m_Controller->setWalkDirection(walkDir);
    } else {
        m_Controller->setWalkDirection(btVector3(0, 0, 0));
    }

    // Apply falling/jumping
    if (!IsGrounded()) {
        m_VerticalVelocity -= (gravity.Length() * deltaTime);
        m_VerticalVelocity = std::max(m_VerticalVelocity, -m_FallSpeed);
    } else {
        if (m_VerticalVelocity < 0) {
            m_VerticalVelocity = 0;
        }
    }

    // Update controller with vertical velocity
    btVector3 verticalVelocity(0, m_VerticalVelocity, 0);
    m_Controller->setLinearVelocity(verticalVelocity);

    // Step simulation for this frame
    m_Controller->updateAction(nullptr, deltaTime);
}

void KinematicController::SetWalkDirection(const Vec3& direction) {
    m_WalkDirection = direction;
}

void KinematicController::Jump(const Vec3& impulse) {
    if (IsGrounded()) {
        m_VerticalVelocity = impulse.y;
    }
}

bool KinematicController::IsGrounded() const {
    if (!m_Controller) return false;
    return m_Controller->canJump();
}

Vec3 KinematicController::GetPosition() const {
    if (!m_GhostObject) return Vec3(0, 0, 0);
    
    btVector3 pos = m_GhostObject->getWorldTransform().getOrigin();
    return Vec3(pos.x(), pos.y(), pos.z());
}

void KinematicController::SetPosition(const Vec3& position) {
    if (!m_GhostObject) return;
    
    btTransform transform = m_GhostObject->getWorldTransform();
    transform.setOrigin(btVector3(position.x, position.y, position.z));
    m_GhostObject->setWorldTransform(transform);
}

void KinematicController::SetStepHeight(float height) {
    if (!m_Controller) return;
    
    m_Controller->setStepHeight(height);
}

float KinematicController::GetStepHeight() const {
    if (!m_Controller) return 0.35f;
    
    return m_Controller->getStepHeight();
}

void KinematicController::SetActive(bool active) {
    if (!m_GhostObject) return;
    
    if (active) {
        m_GhostObject->activate(true);
    } else {
        m_GhostObject->setActivationState(ISLAND_SLEEPING);
    }
}

bool KinematicController::IsActive() const {
    if (!m_GhostObject) return false;
    
    return m_GhostObject->isActive();
}

void KinematicController::SyncTransformFromPhysics(Vec3& outPosition) const {
    outPosition = GetPosition();
}

void KinematicController::SyncTransformToPhysics(const Vec3& position) {
    SetPosition(position);
}

void KinematicController::SetCollisionFilterGroup(uint16_t group) {
    if (!m_GhostObject) return;
    
    // Note: This requires re-adding to world in real implementation
    // Simplified for now
}

void KinematicController::SetCollisionFilterMask(uint16_t mask) {
    if (!m_GhostObject) return;
    
    // Note: This requires re-adding to world in real implementation
    // Simplified for now
}
