#include "PhysXCharacterController.h"

#ifdef USE_PHYSX

#include "PhysXBackend.h"
#include "IPhysicsShape.h"
#include <PxPhysicsAPI.h>
#include <characterkinematic/PxControllerManager.h>
#include <characterkinematic/PxCapsuleController.h>

using namespace physx;

PhysXCharacterController::PhysXCharacterController(PhysXBackend* backend)
    : m_Backend(backend)
    , m_Controller(nullptr)
    , m_ControllerManager(nullptr)
    , m_WalkDirection(0, 0, 0)
    , m_VerticalVelocity(0.0f)
    , m_MaxWalkSpeed(5.0f)
    , m_FallSpeed(20.0f)
    , m_StepHeight(0.35f)
    , m_IsGrounded(false)
{
}

PhysXCharacterController::~PhysXCharacterController() {
    if (m_Controller) {
        m_Backend->UnregisterCharacterController(this);
        m_Controller->release();
        m_Controller = nullptr;
    }
    
    if (m_ControllerManager) {
        m_ControllerManager->release();
        m_ControllerManager = nullptr;
    }
}

void PhysXCharacterController::Initialize(std::shared_ptr<IPhysicsShape> shape, float mass, float stepHeight) {
    m_StepHeight = stepHeight;
    
    PxScene* scene = m_Backend->GetScene();
    if (!scene) {
        return;
    }

    // Create controller manager
    m_ControllerManager = PxCreateControllerManager(*scene);
    if (!m_ControllerManager) {
        return;
    }

    // Create capsule controller descriptor
    PxCapsuleControllerDesc desc;
    desc.height = 1.8f;  // Default height
    desc.radius = 0.3f;  // Default radius
    desc.stepOffset = stepHeight;
    desc.slopeLimit = 0.707f; // 45 degrees
    desc.contactOffset = 0.1f;
    desc.position = PxExtendedVec3(0, 0, 0);
    desc.material = m_Backend->GetDefaultMaterial();

    // Create controller
    m_Controller = m_ControllerManager->createController(desc);
    if (!m_Controller) {
        return;
    }

    // Register with backend
    m_Backend->RegisterCharacterController(this);
}

void PhysXCharacterController::Update(float deltaTime, const Vec3& gravity) {
    if (!m_Controller) {
        return;
    }

    // Apply gravity
    m_VerticalVelocity += gravity.y * deltaTime;
    
    // Clamp fall speed
    if (m_VerticalVelocity < -m_FallSpeed) {
        m_VerticalVelocity = -m_FallSpeed;
    }

    // Calculate movement
    Vec3 displacement = m_WalkDirection * deltaTime;
    displacement.y = m_VerticalVelocity * deltaTime;

    // Move controller
    PxControllerFilters filters;
    PxControllerCollisionFlags collisionFlags = m_Controller->move(
        PxVec3(displacement.x, displacement.y, displacement.z),
        0.001f, // min distance
        deltaTime,
        filters
    );

    // Check if grounded
    m_IsGrounded = (collisionFlags & PxControllerCollisionFlag::eCOLLISION_DOWN) != 0;
    
    if (m_IsGrounded) {
        m_VerticalVelocity = 0.0f;
    }
}

void PhysXCharacterController::SetWalkDirection(const Vec3& direction) {
    m_WalkDirection = direction;
    
    // Clamp to max speed
    float speed = m_WalkDirection.Length();
    if (speed > m_MaxWalkSpeed) {
        m_WalkDirection = m_WalkDirection * (m_MaxWalkSpeed / speed);
    }
}

Vec3 PhysXCharacterController::GetWalkDirection() const {
    return m_WalkDirection;
}

void PhysXCharacterController::Jump(const Vec3& impulse) {
    if (m_IsGrounded) {
        m_VerticalVelocity = impulse.y;
    }
}

bool PhysXCharacterController::IsGrounded() const {
    return m_IsGrounded;
}

Vec3 PhysXCharacterController::GetPosition() const {
    if (m_Controller) {
        PxExtendedVec3 pos = m_Controller->getPosition();
        return Vec3(static_cast<float>(pos.x), static_cast<float>(pos.y), static_cast<float>(pos.z));
    }
    return Vec3(0, 0, 0);
}

void PhysXCharacterController::SetPosition(const Vec3& position) {
    if (m_Controller) {
        m_Controller->setPosition(PxExtendedVec3(position.x, position.y, position.z));
    }
}

float PhysXCharacterController::GetVerticalVelocity() const {
    return m_VerticalVelocity;
}

void PhysXCharacterController::SetVerticalVelocity(float velocity) {
    m_VerticalVelocity = velocity;
}

void PhysXCharacterController::SetMaxWalkSpeed(float speed) {
    m_MaxWalkSpeed = speed;
}

float PhysXCharacterController::GetMaxWalkSpeed() const {
    return m_MaxWalkSpeed;
}

void PhysXCharacterController::SetFallSpeed(float speed) {
    m_FallSpeed = speed;
}

float PhysXCharacterController::GetFallSpeed() const {
    return m_FallSpeed;
}

void PhysXCharacterController::SetStepHeight(float height) {
    m_StepHeight = height;
    if (m_Controller) {
        m_Controller->setStepOffset(height);
    }
}

float PhysXCharacterController::GetStepHeight() const {
    return m_StepHeight;
}

void* PhysXCharacterController::GetNativeController() {
    return m_Controller;
}

#endif // USE_PHYSX
