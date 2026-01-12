#include "PhysXArticulationLink.h"
#include "PhysXArticulation.h"
#include "PhysXShape.h"
#include "PhysXBackend.h"

#ifdef USE_PHYSX
#include <PxPhysicsAPI.h>
#include <iostream>

using namespace physx;

PhysXArticulationLink::PhysXArticulationLink(PxArticulationLink* link, PhysXArticulation* articulation)
    : m_Link(link)
    , m_Articulation(articulation)
    , m_InboundJoint(nullptr)
    , m_BodyType(BodyType::Dynamic)
{
    if (m_Link) {
        // Set user data to this wrapper
        m_Link->userData = this;

        // Create joint wrapper if inbound joint exists
        PxArticulationJointReducedCoordinate* joint = m_Link->getInboundJoint();
        if (joint) {
            m_InboundJoint = new PhysXArticulationJoint(joint);
        }
    }
}

PhysXArticulationLink::~PhysXArticulationLink() {
    if (m_InboundJoint) {
        delete m_InboundJoint;
    }
    // Link is destroyed when articulation is released or via removeLink
    // We do NOT release m_Link manually here as it's managed by articulation
}

void PhysXArticulationLink::Initialize(BodyType type, float mass, std::shared_ptr<IPhysicsShape> shape) {
    if (!m_Link) return;

    m_BodyType = type; // Links are always dynamic in reduced coordinate, but we store type for consistency

    // Attach shape
    if (shape) {
        std::shared_ptr<PhysXShape> pxShape = std::dynamic_pointer_cast<PhysXShape>(shape);
        if (pxShape) {
            // Need to clone shape or add it
            // Typically generic shape wrapper holds PxMaterial and geometry info
            // For now assume single shape per link for simplicity in this method
             PxShape* nativeShape = pxShape->GetNativeShape();
             if (nativeShape) {
                 m_Link->attachShape(*nativeShape);
             }
        }
    }

    // Set mass properties
    // PxArticulationLink calculates mass from shapes usually, or we can override
    // Using updateMassPropertiesFromShapes is safest
    PxRigidBodyExt::updateMassAndInertia(*m_Link, mass > 0 ? mass : 1.0f);
    
    if (mass > 0) {
        m_Link->setMass(mass);
    }
}

PhysXArticulationJoint* PhysXArticulationLink::GetInboundJoint() const {
    return m_InboundJoint;
}

std::vector<PhysXArticulationLink*> PhysXArticulationLink::GetChildren() const {
    if (!m_Articulation) return {};
    
    // This is expensive O(N) or need caching.
    // For now we can query the articulation links and find those whose parent is this.
    // Optimally PhysXArticulation maintains the tree.
    // But PxArticulationLink has getChildren() helper in extensions? No.
    
    std::vector<PhysXArticulationLink*> children;
    PxU32 childCount = m_Link->getNbChildren();
    if (childCount > 0) {
        std::vector<PxArticulationLink*> pxChildren(childCount);
        m_Link->getChildren(pxChildren.data(), childCount);
        
        for (auto* child : pxChildren) {
             if (child && child->userData) {
                 children.push_back(static_cast<PhysXArticulationLink*>(child->userData));
             }
        }
    }
    return children;
}

// IPhysicsRigidBody Implementation
void PhysXArticulationLink::SetLinearVelocity(const Vec3& velocity) {
    if (m_Link) m_Link->setLinearVelocity(PxVec3(velocity.x, velocity.y, velocity.z));
}

Vec3 PhysXArticulationLink::GetLinearVelocity() const {
    if (!m_Link) return Vec3(0);
    PxVec3 v = m_Link->getLinearVelocity();
    return Vec3(v.x, v.y, v.z);
}

void PhysXArticulationLink::SetAngularVelocity(const Vec3& velocity) {
    if (m_Link) m_Link->setAngularVelocity(PxVec3(velocity.x, velocity.y, velocity.z));
}

Vec3 PhysXArticulationLink::GetAngularVelocity() const {
    if (!m_Link) return Vec3(0);
    PxVec3 v = m_Link->getAngularVelocity();
    return Vec3(v.x, v.y, v.z);
}

void PhysXArticulationLink::ApplyForce(const Vec3& force) {
    if (m_Link) PxRigidBodyExt::addForceAtPos(*m_Link, PxVec3(force.x, force.y, force.z), m_Link->getGlobalPose().p, PxForceMode::eFORCE);
}

void PhysXArticulationLink::ApplyImpulse(const Vec3& impulse) {
    if (m_Link) PxRigidBodyExt::addForceAtPos(*m_Link, PxVec3(impulse.x, impulse.y, impulse.z), m_Link->getGlobalPose().p, PxForceMode::eIMPULSE);
}

void PhysXArticulationLink::ApplyForceAtPoint(const Vec3& force, const Vec3& point) {
    if (m_Link) PxRigidBodyExt::addForceAtPos(*m_Link, PxVec3(force.x, force.y, force.z), PxVec3(point.x, point.y, point.z), PxForceMode::eFORCE);
}

void PhysXArticulationLink::ApplyImpulseAtPoint(const Vec3& impulse, const Vec3& point) {
    if (m_Link) PxRigidBodyExt::addForceAtPos(*m_Link, PxVec3(impulse.x, impulse.y, impulse.z), PxVec3(point.x, point.y, point.z), PxForceMode::eIMPULSE);
}

void PhysXArticulationLink::ApplyTorque(const Vec3& torque) {
    if (m_Link) m_Link->addTorque(PxVec3(torque.x, torque.y, torque.z));
}

void PhysXArticulationLink::SetMass(float mass) {
    if (m_Link) {
        m_Link->setMass(mass);
        // Should probably recompute inertia
    }
}

float PhysXArticulationLink::GetMass() const {
    return m_Link ? m_Link->getMass() : 0.0f;
}

void PhysXArticulationLink::SetFriction(float friction) {
    // Friction is per-material/shape.
    // Helper to set on all shapes
    if (m_Link) {
        PxU32 nbShapes = m_Link->getNbShapes();
        std::vector<PxShape*> shapes(nbShapes);
        m_Link->getShapes(shapes.data(), nbShapes);
        for(auto* shape : shapes) {
            PxMaterial* mats[1];
            shape->getMaterials(mats, 1);
            if(mats[0]) {
                mats[0]->setStaticFriction(friction);
                mats[0]->setDynamicFriction(friction);
            }
        }
    }
}

float PhysXArticulationLink::GetFriction() const {
    // Just return first shape's friction
     if (m_Link && m_Link->getNbShapes() > 0) {
        PxShape* shape;
        m_Link->getShapes(&shape, 1);
        PxMaterial* mat;
        shape->getMaterials(&mat, 1);
        return mat->getStaticFriction();
    }
    return 0.5f;
}

void PhysXArticulationLink::SetRestitution(float restitution) {
     if (m_Link) {
        PxU32 nbShapes = m_Link->getNbShapes();
        std::vector<PxShape*> shapes(nbShapes);
        m_Link->getShapes(shapes.data(), nbShapes);
        for(auto* shape : shapes) {
            PxMaterial* mats[1];
            shape->getMaterials(mats, 1);
            if(mats[0]) mats[0]->setRestitution(restitution);
        }
    }
}

float PhysXArticulationLink::GetRestitution() const {
    if (m_Link && m_Link->getNbShapes() > 0) {
        PxShape* shape;
        m_Link->getShapes(&shape, 1);
        PxMaterial* mat;
        shape->getMaterials(&mat, 1);
        return mat->getRestitution();
    }
    return 0.5f;
}

void PhysXArticulationLink::SetLinearDamping(float damping) {
    if (m_Link) m_Link->setLinearDamping(damping);
}

void PhysXArticulationLink::SetAngularDamping(float damping) {
    if (m_Link) m_Link->setAngularDamping(damping);
}

void PhysXArticulationLink::SetGravityEnabled(bool enabled) {
    if (m_Link) m_Link->setActorFlag(PxActorFlag::eDISABLE_GRAVITY, !enabled);
}

bool PhysXArticulationLink::IsGravityEnabled() const {
    return m_Link && !(m_Link->getActorFlags() & PxActorFlag::eDISABLE_GRAVITY);
}

bool PhysXArticulationLink::IsActive() const {
    // Articulation link activity is tied to articulation, but can be put to sleep?
    return m_Link && !m_Link->isSleeping();
}

void PhysXArticulationLink::SetActive(bool active) {
    if (!active && m_Link) m_Link->putToSleep();
    else if (active && m_Link) m_Link->wakeUp();
}

void PhysXArticulationLink::SyncTransformFromPhysics(Vec3& outPosition, Quat& outRotation) {
    if (m_Link) {
        PxTransform t = m_Link->getGlobalPose();
        outPosition = Vec3(t.p.x, t.p.y, t.p.z);
        outRotation = Quat(t.q.x, t.q.y, t.q.z, t.q.w);
    }
}

void PhysXArticulationLink::SyncTransformToPhysics(const Vec3& position, const Quat& rotation) {
    if (m_Link) {
        // Can we move a link directly? Usually we move the root or use drives.
        // Moving a child link directly might break the articulation constraints or just teleport it.
        // Assuming this is used for initialization or teleport:
        m_Link->setGlobalPose(PxTransform(PxVec3(position.x, position.y, position.z), PxQuat(rotation.x, rotation.y, rotation.z, rotation.w)));
    }
}

void PhysXArticulationLink::SetOnCollisionCallback(OnCollisionCallback callback) {
    m_CollisionCallback = callback;
}

void* PhysXArticulationLink::GetNativeBody() {
    return m_Link;
}

void PhysXArticulationLink::HandleCollision(const CollisionInfo& info) {
    if (m_CollisionCallback) {
        m_CollisionCallback(info);
    }
}

#endif // USE_PHYSX
