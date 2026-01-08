#include "KinematicController.h"
#include "PhysicsSystem.h"
#include <btBulletDynamicsCommon.h>
#include <BulletCollision/CollisionDispatch/btGhostObject.h>
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
    m_Controller->setGravity(btVector3(gravity.x, gravity.y, gravity.z));

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
// ==================== ADVANCED MOVEMENT QUERIES ====================

bool KinematicController::IsWallAhead(float maxDistance, Vec3* outWallNormal) const {
    if (!m_Controller) {
        return false;
    }

    Vec3 pos = GetPosition();
    Vec3 forward = m_WalkDirection;
    if (forward.Length() < 0.001f) {
        forward = Vec3(0, 0, -1); // Default forward direction
    }
    forward.Normalize();

    Vec3 checkPos = pos + forward * maxDistance;
    
    // Cast a ray forward to detect walls
    RaycastHit hit;
    if (PhysicsSystem::Get().Raycast(pos, checkPos, hit)) {
        if (outWallNormal) {
            *outWallNormal = hit.normal;
        }
        return true;
    }

    return false;
}

bool KinematicController::IsOnSlope(float* outSlopeAngle) const {
    if (!m_Controller) {
        return false;
    }

    Vec3 groundNormal;
    if (GetGroundNormal(groundNormal, 0.5f)) {
        // Calculate angle between up vector and ground normal
        Vec3 upVector(0, 1, 0);
        float dotProduct = groundNormal.Dot(upVector);
        dotProduct = std::max(-1.0f, std::min(1.0f, dotProduct)); // Clamp for acos
        
        float angle = std::acos(dotProduct);
        if (outSlopeAngle) {
            *outSlopeAngle = angle;
        }

        // Consider it a slope if angle is between 5 and 85 degrees
        const float minSlopeAngle = 0.087266f; // ~5 degrees
        const float maxSlopeAngle = 1.48353f;  // ~85 degrees
        return angle > minSlopeAngle && angle < maxSlopeAngle;
    }

    return false;
}

bool KinematicController::GetGroundNormal(Vec3& outNormal, float maxDistance) const {
    if (!m_Controller) {
        return false;
    }

    Vec3 pos = GetPosition();
    Vec3 below = pos + Vec3(0, -maxDistance, 0);

    RaycastHit hit;
    if (PhysicsSystem::Get().Raycast(pos, below, hit)) {
        outNormal = hit.normal;
        return true;
    }

    outNormal = Vec3(0, 1, 0); // Default to up
    return false;
}

bool KinematicController::WillMoveCollide(const Vec3& moveDirection, Vec3* outBlockingDirection) const {
    if (!m_Controller) {
        return false;
    }

    Vec3 startPos = GetPosition();
    Vec3 targetPos = startPos + moveDirection;

    RaycastHit hit;
    if (PhysicsSystem::Get().Raycast(startPos, targetPos, hit)) {
        if (outBlockingDirection) {
            *outBlockingDirection = hit.normal;
        }
        return true;
    }

    return false;
}

float KinematicController::GetDistanceToCollision(const Vec3& direction) const {
    if (!m_Controller || direction.Length() < 0.001f) {
        return std::numeric_limits<float>::max();
    }

    Vec3 pos = GetPosition();
    Vec3 normalizedDir = direction;
    normalizedDir.Normalize();
    
    // Cast ray a very long distance to find collision
    const float maxDistance = 1000.0f;
    Vec3 farPos = pos + normalizedDir * maxDistance;

    RaycastHit hit;
    if (PhysicsSystem::Get().Raycast(pos, farPos, hit)) {
        return hit.distance;
    }

    return maxDistance;
}

Vec3 KinematicController::GetVelocity() const {
    if (!m_Controller) {
        return Vec3(0, 0, 0);
    }

    // Current velocity is walk direction plus vertical component
    Vec3 velocity = m_WalkDirection;
    velocity.y = m_VerticalVelocity;
    return velocity;
}

float KinematicController::GetMoveSpeed() const {
    if (!m_Controller) {
        return 0.0f;
    }

    Vec3 horizontalVelocity = m_WalkDirection;
    horizontalVelocity.y = 0; // Remove vertical component
    return horizontalVelocity.Length();
}