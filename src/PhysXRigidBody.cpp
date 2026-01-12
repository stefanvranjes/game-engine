#include "PhysXRigidBody.h"

#ifdef USE_PHYSX

#include "PhysXBackend.h"
#include "IPhysicsShape.h"
#include <PxPhysicsAPI.h>

using namespace physx;

PhysXRigidBody::PhysXRigidBody(PhysXBackend* backend)
    : m_Backend(backend)
    , m_Actor(nullptr)
    , m_DynamicActor(nullptr)
    , m_BodyType(BodyType::Dynamic)
    , m_Mass(1.0f)
    , m_GravityEnabled(true)
{
}

PhysXRigidBody::~PhysXRigidBody() {
    if (m_Actor) {
        m_Backend->UnregisterRigidBody(this);
        m_Actor->release();
        m_Actor = nullptr;
        m_DynamicActor = nullptr;
    }
}

void PhysXRigidBody::Initialize(BodyType type, float mass, std::shared_ptr<IPhysicsShape> shape) {
    m_BodyType = type;
    m_Mass = mass;

    PxPhysics* physics = m_Backend->GetPhysics();
    PxScene* scene = m_Backend->GetScene();

    if (!physics || !scene) {
        return;
    }

    // Get PhysX shape from interface
    PxShape* pxShape = static_cast<PxShape*>(shape->GetNativeShape());
    if (!pxShape) {
        return;
    }

    PxTransform transform(PxVec3(0, 0, 0)); // Initial position

    if (type == BodyType::Static) {
        // Create static actor
        m_Actor = physics->createRigidStatic(transform);
        m_Actor->attachShape(*pxShape);
    } else {
        // Create dynamic actor
        PxRigidDynamic* dynamicActor = physics->createRigidDynamic(transform);
        m_Actor = dynamicActor;
        m_DynamicActor = dynamicActor;

        dynamicActor->attachShape(*pxShape);

        if (type == BodyType::Dynamic) {
            // Set mass properties
            PxRigidBodyExt::updateMassAndInertia(*dynamicActor, mass);
        } else if (type == BodyType::Kinematic) {
            // Kinematic bodies are controlled by code
            dynamicActor->setRigidBodyFlag(PxRigidBodyFlag::eKINEMATIC, true);
        }
    }

    // Add to scene
    m_Actor->userData = this;
    scene->addActor(*m_Actor);

    // Register with backend
    m_Backend->RegisterRigidBody(this);
}

void PhysXRigidBody::SetLinearVelocity(const Vec3& velocity) {
    if (m_DynamicActor) {
        m_DynamicActor->setLinearVelocity(PxVec3(velocity.x, velocity.y, velocity.z));
    }
}

Vec3 PhysXRigidBody::GetLinearVelocity() const {
    if (m_DynamicActor) {
        PxVec3 vel = m_DynamicActor->getLinearVelocity();
        return Vec3(vel.x, vel.y, vel.z);
    }
    return Vec3(0, 0, 0);
}

void PhysXRigidBody::SetAngularVelocity(const Vec3& velocity) {
    if (m_DynamicActor) {
        m_DynamicActor->setAngularVelocity(PxVec3(velocity.x, velocity.y, velocity.z));
    }
}

Vec3 PhysXRigidBody::GetAngularVelocity() const {
    if (m_DynamicActor) {
        PxVec3 vel = m_DynamicActor->getAngularVelocity();
        return Vec3(vel.x, vel.y, vel.z);
    }
    return Vec3(0, 0, 0);
}

void PhysXRigidBody::ApplyForce(const Vec3& force) {
    if (m_DynamicActor) {
        m_DynamicActor->addForce(PxVec3(force.x, force.y, force.z));
    }
}

void PhysXRigidBody::ApplyImpulse(const Vec3& impulse) {
    if (m_DynamicActor) {
        m_DynamicActor->addForce(PxVec3(impulse.x, impulse.y, impulse.z), PxForceMode::eIMPULSE);
    }
}

void PhysXRigidBody::ApplyForceAtPoint(const Vec3& force, const Vec3& point) {
    if (m_DynamicActor) {
        PxRigidBodyExt::addForceAtPos(*m_DynamicActor, 
            PxVec3(force.x, force.y, force.z), 
            PxVec3(point.x, point.y, point.z));
    }
}

void PhysXRigidBody::ApplyImpulseAtPoint(const Vec3& impulse, const Vec3& point) {
    if (m_DynamicActor) {
        PxRigidBodyExt::addForceAtPos(*m_DynamicActor, 
            PxVec3(impulse.x, impulse.y, impulse.z), 
            PxVec3(point.x, point.y, point.z),
            PxForceMode::eIMPULSE);
    }
}

void PhysXRigidBody::ApplyTorque(const Vec3& torque) {
    if (m_DynamicActor) {
        m_DynamicActor->addTorque(PxVec3(torque.x, torque.y, torque.z));
    }
}

void PhysXRigidBody::SetMass(float mass) {
    m_Mass = mass;
    if (m_DynamicActor) {
        PxRigidBodyExt::updateMassAndInertia(*m_DynamicActor, mass);
    }
}

float PhysXRigidBody::GetMass() const {
    return m_Mass;
}

void PhysXRigidBody::SetFriction(float friction) {
    if (m_Actor) {
        PxShape* shape = nullptr;
        m_Actor->getShapes(&shape, 1);
        if (shape) {
            PxMaterial* material = nullptr;
            shape->getMaterials(&material, 1);
            if (material) {
                material->setStaticFriction(friction);
                material->setDynamicFriction(friction);
            }
        }
    }
}

float PhysXRigidBody::GetFriction() const {
    if (m_Actor) {
        PxShape* shape = nullptr;
        m_Actor->getShapes(&shape, 1);
        if (shape) {
            PxMaterial* material = nullptr;
            shape->getMaterials(&material, 1);
            if (material) {
                return material->getStaticFriction();
            }
        }
    }
    return 0.5f;
}

void PhysXRigidBody::SetRestitution(float restitution) {
    if (m_Actor) {
        PxShape* shape = nullptr;
        m_Actor->getShapes(&shape, 1);
        if (shape) {
            PxMaterial* material = nullptr;
            shape->getMaterials(&material, 1);
            if (material) {
                material->setRestitution(restitution);
            }
        }
    }
}

float PhysXRigidBody::GetRestitution() const {
    if (m_Actor) {
        PxShape* shape = nullptr;
        m_Actor->getShapes(&shape, 1);
        if (shape) {
            PxMaterial* material = nullptr;
            shape->getMaterials(&material, 1);
            if (material) {
                return material->getRestitution();
            }
        }
    }
    return 0.6f;
}

void PhysXRigidBody::SetLinearDamping(float damping) {
    if (m_DynamicActor) {
        m_DynamicActor->setLinearDamping(damping);
    }
}

void PhysXRigidBody::SetAngularDamping(float damping) {
    if (m_DynamicActor) {
        m_DynamicActor->setAngularDamping(damping);
    }
}

void PhysXRigidBody::SetGravityEnabled(bool enabled) {
    m_GravityEnabled = enabled;
    if (m_DynamicActor) {
        m_DynamicActor->setActorFlag(PxActorFlag::eDISABLE_GRAVITY, !enabled);
    }
}

bool PhysXRigidBody::IsGravityEnabled() const {
    return m_GravityEnabled;
}

bool PhysXRigidBody::IsActive() const {
    if (m_DynamicActor) {
        return !m_DynamicActor->isSleeping();
    }
    return false;
}

void PhysXRigidBody::SetActive(bool active) {
    if (m_DynamicActor) {
        if (active) {
            m_DynamicActor->wakeUp();
        } else {
            m_DynamicActor->putToSleep();
        }
    }
}

void PhysXRigidBody::SyncTransformFromPhysics(Vec3& outPosition, Quat& outRotation) {
    if (m_Actor) {
        PxTransform transform = m_Actor->getGlobalPose();
        outPosition = Vec3(transform.p.x, transform.p.y, transform.p.z);
        outRotation = Quat(transform.q.w, transform.q.x, transform.q.y, transform.q.z);
    }
}

void PhysXRigidBody::SyncTransformToPhysics(const Vec3& position, const Quat& rotation) {
    if (m_Actor) {
        PxTransform transform(
            PxVec3(position.x, position.y, position.z),
            PxQuat(rotation.x, rotation.y, rotation.z, rotation.w)
        );
        m_Actor->setGlobalPose(transform);
    }
}

void* PhysXRigidBody::GetNativeBody() {
    return m_Actor;
}

void PhysXRigidBody::SetOnCollisionCallback(OnCollisionCallback callback) {
    m_CollisionCallback = callback;
}

void PhysXRigidBody::HandleCollision(const CollisionInfo& info) {
    if (m_CollisionCallback) {
        m_CollisionCallback(info);
    }
}

#endif // USE_PHYSX
