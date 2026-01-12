#include "PhysXRigidBody.h"
#include "PhysXFluidVolume.h" // Added for fluid support

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

void PhysXRigidBody::SetOnTriggerEnterCallback(OnTriggerCallback callback) {
    m_TriggerEnterCallback = callback;
}

void PhysXRigidBody::SetOnTriggerExitCallback(OnTriggerCallback callback) {
    m_TriggerExitCallback = callback;
}

void PhysXRigidBody::SetTriggerVelocityThreshold(float threshold) {
    m_TriggerVelocityThreshold = threshold;
}

float PhysXRigidBody::GetTriggerVelocityThreshold() const {
    return m_TriggerVelocityThreshold;
}

void PhysXRigidBody::HandleTriggerEnter(const TriggerInfo& info) {
    // Check velocity threshold
    if (info.relativeVelocity.Length() < m_TriggerVelocityThreshold) {
        return;
    }

    if (info.otherBody) {
        m_TriggerEntryTimes[info.otherBody] = m_Backend->GetSimulationTime();
        
        // Fluid check
        PhysXFluidVolume* fluid = m_Backend->GetFluidVolume(info.otherBody);
        if (fluid) {
            AddFluid(fluid);
        }
    }
    
    if (m_TriggerEnterCallback) {
        m_TriggerEnterCallback(info);
    }
}

void PhysXRigidBody::HandleTriggerExit(const TriggerInfo& info) {
    TriggerInfo mutableInfo = info;
    bool wasTracking = false;
    
    if (info.otherBody) {
        auto it = m_TriggerEntryTimes.find(info.otherBody);
        if (it != m_TriggerEntryTimes.end()) {
            wasTracking = true;
            double entryTime = it->second;
            double exitTime = m_Backend->GetSimulationTime();
            mutableInfo.dwellTime = static_cast<float>(exitTime - entryTime);
            m_TriggerEntryTimes.erase(it);
        }
        
        // Fluid check
        PhysXFluidVolume* fluid = m_Backend->GetFluidVolume(info.otherBody);
        if (fluid) {
            RemoveFluid(fluid);
        }
    }
    
    // Only dispatch exit if we were tracking this object (meaning it passed the entry threshold)
    // Or if there is no threshold set (backward compatibility/safety, though checking map is robust logic)
    // Actually, if threshold is set, we MUST check map. If threshold is 0, map should contain it.
    // Logic: If map entry existed, dispatch.
    
    if (wasTracking) {
        if (m_TriggerExitCallback) {
            m_TriggerExitCallback(mutableInfo);
        }
    }
}

float PhysXRigidBody::GetTriggerDwellTime(IPhysicsRigidBody* otherBody) const {
    if (!otherBody) return 0.0f;
    
    auto it = m_TriggerEntryTimes.find(otherBody);
    if (it != m_TriggerEntryTimes.end()) {
        double entryTime = it->second;
        double currentTime = m_Backend->GetSimulationTime();
        return static_cast<float>(currentTime - entryTime);
    }
    return 0.0f;
}

void PhysXRigidBody::SetCCDEnabled(bool enabled) {
    if (m_DynamicActor) {
        m_DynamicActor->setRigidBodyFlag(PxRigidBodyFlag::eENABLE_CCD, enabled);
    }
}

bool PhysXRigidBody::IsCCDEnabled() const {
    if (m_DynamicActor) {
        return (m_DynamicActor->getRigidBodyFlags() & PxRigidBodyFlag::eENABLE_CCD);
    }
    return false;
}

void PhysXRigidBody::AddFluid(PhysXFluidVolume* fluid) {
    if (!fluid) return;
    auto it = std::find(m_ActiveFluids.begin(), m_ActiveFluids.end(), fluid);
    if (it == m_ActiveFluids.end()) {
        m_ActiveFluids.push_back(fluid);
    }
}

void PhysXRigidBody::RemoveFluid(PhysXFluidVolume* fluid) {
    if (!fluid) return;
    auto it = std::find(m_ActiveFluids.begin(), m_ActiveFluids.end(), fluid);
    if (it != m_ActiveFluids.end()) {
        m_ActiveFluids.erase(it);
    }
}

void PhysXRigidBody::UpdateBuoyancy(float deltaTime) {
    if (m_ActiveFluids.empty() || m_BodyType != BodyType::Dynamic) return;

    for (auto* fluid : m_ActiveFluids) {
        Vec3 buoyancyCenter;
        Vec3 flowVelocity;
        
        float submergedFraction = fluid->CalculateSubmergedVolume(this, buoyancyCenter, flowVelocity);
        
        if (submergedFraction > 0.0f) {
            PxRigidActor* actor = m_Actor;
            PxBounds3 bounds = actor->getWorldBounds();
            PxVec3 dims = bounds.getDimensions();
            float volume = dims.x * dims.y * dims.z;
            
            // Allow override if density/mass implies specific volume?
            // For now use AABB volume approximation.
            // If m_Mass is set, we could infer volume if we knew object density, but we don't.
            
            float displacedVolume = volume * submergedFraction;
            Vec3 gravity = m_Backend->GetGravity(); 
            
            // Buoyancy Force = -Gravity * Density * DisplacedVolume
            // PhysX Gravity is usually negative (down), so -Gravity is Up.
            Vec3 buoyancyForce = -gravity * fluid->Density * displacedVolume; // Archimedes

            // Apply Buoyancy
            ApplyForceAtPoint(buoyancyForce, buoyancyCenter);
            
            // Drag
            Vec3 vel = GetLinearVelocity(); // Approximation: use body COM velocity
            Vec3 relativeVel = vel - flowVelocity;
            float speed = relativeVel.Length();
            
            if (speed > 1e-4f) {
                // Linear Drag: F = -0.5 * rho * v^2 * Cd * Area
                // Simplified: F = -v * k * fraction
                // Let's use simplified linear drag for game stability
                // DragCoefficient usually combines 0.5 * Cd * Area * Rho...
                // But fluid->LinearDrag is just a "Drag Factor".
                // F = -RelativeVel * fluid->LinearDrag * displacedVolume * speed?
                // Or just standard damping approach.
                // fluid->LinearDrag is user conceptual drag.
                // Let's model it as: F = -RelativeVel * fluid->LinearDrag * speed * submergedFraction
                
                Vec3 dragForce = -relativeVel.Normalized() * (speed * speed * fluid->LinearDrag * submergedFraction * 10.0f); 
                // Using V^2 drag. Multiplier 10.0f to make default 0.5 noticeable.
                ApplyForce(dragForce);
            }
            
            // Angular Drag
            Vec3 angVel = GetAngularVelocity();
            float angSpeed = angVel.Length();
            if (angSpeed > 1e-4f) {
                Vec3 angDragTorque = -angVel.Normalized() * (angSpeed * angSpeed * fluid->AngularDrag * submergedFraction * 10.0f);
                ApplyTorque(angDragTorque);
            }
        }
    }
}

#endif // USE_PHYSX
