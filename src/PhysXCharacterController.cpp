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
    , m_UpDirection(0, 1, 0)
    , m_VerticalVelocity(0.0f)
    , m_MaxWalkSpeed(5.0f)
    , m_FallSpeed(20.0f)
    , m_StepHeight(0.35f)
    , m_SlopeLimit(0.707f)
    , m_ContactOffset(0.1f)
    , m_PushForce(5.0f)
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
    desc.stepOffset = m_StepHeight;
    desc.slopeLimit = m_SlopeLimit;
    desc.contactOffset = m_ContactOffset;
    desc.upDirection = PxVec3(m_UpDirection.x, m_UpDirection.y, m_UpDirection.z);
    desc.reportCallback = this; // Register hit report callback
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

void PhysXCharacterController::SetSlopeLimit(float slopeLimit) {
    m_SlopeLimit = slopeLimit;
    if (m_Controller) {
        m_Controller->setSlopeLimit(m_SlopeLimit);
    }
}

float PhysXCharacterController::GetSlopeLimit() const {
    return m_SlopeLimit;
}

void PhysXCharacterController::SetContactOffset(float offset) {
    m_ContactOffset = offset;
    if (m_Controller) {
        m_Controller->setContactOffset(m_ContactOffset);
    }
}

float PhysXCharacterController::GetContactOffset() const {
    return m_ContactOffset;
}

void PhysXCharacterController::Resize(float height) {
    if (m_Controller) {
        m_Controller->resize(height);
    }
}

void PhysXCharacterController::SetUpDirection(const Vec3& up) {
    m_UpDirection = up;
    if (m_Controller) {
        m_Controller->setUpDirection(PxVec3(up.x, up.y, up.z));
    }
}

Vec3 PhysXCharacterController::GetUpDirection() const {
    return m_UpDirection;
}

void PhysXCharacterController::SetPushForce(float force) {
    m_PushForce = force;
}

float PhysXCharacterController::GetPushForce() const {
    return m_PushForce;
}

void PhysXCharacterController::onShapeHit(const physx::PxControllerShapeHit& hit) {
    // Apply force to dynamic rigid bodies
    PxRigidDynamic* actor = hit.shape->getActor()->is<PxRigidDynamic>();
    if (actor && !(actor->getRigidBodyFlags() & PxRigidBodyFlag::eKINEMATIC)) {
        if (actor->getMassSpaceInertiaTensor() != PxVec3(0.0f)) { // Check if not static/kinematic effectively
            const PxVec3 pushDir = hit.worldNormal * -1.0f; // Push in direction of character movement (roughly)
            // Or better: push along the hit normal inverted (which points out of obstacle towards char?)
            // Actually, we want to push the object away from the character.
            // hit.dir is the direction of controller movement.
            // But for a push, we usually want to use the controller's velocity or the hit normal.
            
            // Let's use the hit direction (controller movement direction) projected onto the object
            // Or simply the direction from character to object?
            
            // Standard approach: push along the controller's motion
            // const PxVec3 pushDir = PxVec3(hit.dir.x, 0, hit.dir.z).getNormalized();
            
            // Apply impulse
            // We need to apply force at the contact point
            PxVec3 force = hit.dir * m_PushForce;
            PxRigidBodyExt::addForceAtPos(*actor, force, hit.worldPos, PxForceMode::eIMPULSE);
        }
    }
}

void PhysXCharacterController::onControllerHit(const physx::PxControllersHit& hit) {
    // Handle character-character collisions if needed
}

void PhysXCharacterController::onObstacleHit(const physx::PxControllerObstacleHit& hit) {
    // Handle obstacle collisions (not using PxObstacleContext currently)
}

#endif // USE_PHYSX
