#include "RigidBody.h"
#include "PhysicsSystem.h"
#include <btBulletDynamicsCommon.h>

// Custom motion state to keep Bullet in sync with our transforms
class RigidBodyMotionState : public btMotionState {
public:
    RigidBodyMotionState() : m_Position(0, 0, 0), m_Rotation(0, 0, 0, 1) {}
    
    void getWorldTransform(btTransform& worldTrans) const override {
        worldTrans.setOrigin(btVector3(m_Position.x, m_Position.y, m_Position.z));
        worldTrans.setRotation(btQuaternion(m_Rotation.x, m_Rotation.y, m_Rotation.z, m_Rotation.w));
    }
    
    void setWorldTransform(const btTransform& worldTrans) override {
        btVector3 pos = worldTrans.getOrigin();
        btQuaternion rot = worldTrans.getRotation();
        m_Position = Vec3(pos.x(), pos.y(), pos.z());
        m_Rotation = Quat(rot.x(), rot.y(), rot.z(), rot.w());
    }
    
    Vec3 m_Position;
    Quat m_Rotation;
};

RigidBody::RigidBody()
    : m_BtRigidBody(nullptr),
      m_MotionState(nullptr),
      m_BodyType(BodyType::Static),
      m_Mass(1.0f),
      m_GravityEnabled(true) {
}

RigidBody::~RigidBody() {
    if (m_BtRigidBody) {
        PhysicsSystem::Get().UnregisterRigidBody(m_BtRigidBody);
        delete m_BtRigidBody;
    }
    
    if (m_MotionState) {
        delete m_MotionState;
    }
}

void RigidBody::Initialize(BodyType type, float mass, const PhysicsCollisionShape& shape) {
    if (m_BtRigidBody) {
        return; // Already initialized
    }

    m_BodyType = type;
    m_Mass = mass;
    m_Shape = shape;

    // Create motion state
    m_MotionState = new RigidBodyMotionState();

    // Calculate inertia
    btVector3 inertia(0, 0, 0);
    if (type != BodyType::Static) {
        shape.GetShape()->calculateLocalInertia(mass, inertia);
    }

    // Create rigid body
    btRigidBody::btRigidBodyConstructionInfo rbInfo(
        (type == BodyType::Static) ? 0.0f : mass,
        m_MotionState,
        shape.GetShape(),
        inertia
    );

    m_BtRigidBody = new btRigidBody(rbInfo);

    // Set collision flags
    if (type == BodyType::Kinematic) {
        m_BtRigidBody->setCollisionFlags(
            m_BtRigidBody->getCollisionFlags() | btRigidBody::CF_KINEMATIC_OBJECT
        );
        m_BtRigidBody->setActivationState(DISABLE_DEACTIVATION);
    }

    // Register with physics system
    PhysicsSystem::Get().RegisterRigidBody(this, m_BtRigidBody);
}

void RigidBody::SetLinearVelocity(const Vec3& velocity) {
    if (!m_BtRigidBody) return;
    m_BtRigidBody->setLinearVelocity(btVector3(velocity.x, velocity.y, velocity.z));
}

Vec3 RigidBody::GetLinearVelocity() const {
    if (!m_BtRigidBody) return Vec3(0, 0, 0);
    
    btVector3 vel = m_BtRigidBody->getLinearVelocity();
    return Vec3(vel.x(), vel.y(), vel.z());
}

void RigidBody::SetAngularVelocity(const Vec3& velocity) {
    if (!m_BtRigidBody) return;
    m_BtRigidBody->setAngularVelocity(btVector3(velocity.x, velocity.y, velocity.z));
}

Vec3 RigidBody::GetAngularVelocity() const {
    if (!m_BtRigidBody) return Vec3(0, 0, 0);
    
    btVector3 vel = m_BtRigidBody->getAngularVelocity();
    return Vec3(vel.x(), vel.y(), vel.z());
}

void RigidBody::ApplyForce(const Vec3& force) {
    if (!m_BtRigidBody) return;
    m_BtRigidBody->applyCentralForce(btVector3(force.x, force.y, force.z));
}

void RigidBody::ApplyForceAtPoint(const Vec3& force, const Vec3& point) {
    if (!m_BtRigidBody) return;
    btVector3 btForce(force.x, force.y, force.z);
    btVector3 btPoint(point.x, point.y, point.z);
    btVector3 btCenterOfMass = m_BtRigidBody->getCenterOfMassPosition();
    btVector3 relPos = btPoint - btCenterOfMass;
    m_BtRigidBody->applyForce(btForce, relPos);
}

void RigidBody::ApplyImpulse(const Vec3& impulse) {
    if (!m_BtRigidBody) return;
    m_BtRigidBody->applyCentralImpulse(btVector3(impulse.x, impulse.y, impulse.z));
}

void RigidBody::ApplyImpulseAtPoint(const Vec3& impulse, const Vec3& point) {
    if (!m_BtRigidBody) return;
    btVector3 btImpulse(impulse.x, impulse.y, impulse.z);
    btVector3 btPoint(point.x, point.y, point.z);
    btVector3 btCenterOfMass = m_BtRigidBody->getCenterOfMassPosition();
    btVector3 relPos = btPoint - btCenterOfMass;
    m_BtRigidBody->applyImpulse(btImpulse, relPos);
}

void RigidBody::SetMass(float mass) {
    if (!m_BtRigidBody) return;
    
    m_Mass = mass;
    
    btVector3 inertia(0, 0, 0);
    if (m_BodyType != BodyType::Static && mass != 0) {
        m_BtRigidBody->getCollisionShape()->calculateLocalInertia(mass, inertia);
    }
    
    m_BtRigidBody->setMassProps(mass, inertia);
}

float RigidBody::GetMass() const {
    if (!m_BtRigidBody) return 0.0f;
    
    float invMass = m_BtRigidBody->getInvMass();
    return invMass > 0 ? 1.0f / invMass : 0.0f;
}

void RigidBody::SetLinearDamping(float damping) {
    if (!m_BtRigidBody) return;
    
    float angularDamping = m_BtRigidBody->getAngularDamping();
    m_BtRigidBody->setDamping(damping, angularDamping);
}

void RigidBody::SetAngularDamping(float damping) {
    if (!m_BtRigidBody) return;
    
    float linearDamping = m_BtRigidBody->getLinearDamping();
    m_BtRigidBody->setDamping(linearDamping, damping);
}

void RigidBody::SetGravityEnabled(bool enabled) {
    if (!m_BtRigidBody) return;
    
    m_GravityEnabled = enabled;
    
    if (enabled) {
        m_BtRigidBody->setGravity(PhysicsSystem::Get().GetGravity());
    } else {
        m_BtRigidBody->setGravity(btVector3(0, 0, 0));
    }
}

void RigidBody::SetCollisionFilterGroup(uint16_t group) {
    if (!m_BtRigidBody) return;
    // This would require removing and re-adding to physics world
    // Simplified for now
}

void RigidBody::SetCollisionFilterMask(uint16_t mask) {
    if (!m_BtRigidBody) return;
    // This would require removing and re-adding to physics world
    // Simplified for now
}

void RigidBody::SyncTransformFromPhysics(Vec3& outPosition, Quat& outRotation) const {
    if (!m_BtRigidBody) return;
    
    btTransform transform = m_BtRigidBody->getWorldTransform();
    
    btVector3 pos = transform.getOrigin();
    outPosition = Vec3(pos.x(), pos.y(), pos.z());
    
    btQuaternion rot = transform.getRotation();
    outRotation = Quat(rot.x(), rot.y(), rot.z(), rot.w());
}

void RigidBody::SyncTransformToPhysics(const Vec3& position, const Quat& rotation) {
    if (!m_BtRigidBody) return;
    
    btTransform transform;
    transform.setOrigin(btVector3(position.x, position.y, position.z));
    transform.setRotation(btQuaternion(rotation.x, rotation.y, rotation.z, rotation.w));
    
    m_BtRigidBody->setWorldTransform(transform);
    
    if (m_MotionState) {
        m_MotionState->setWorldTransform(transform);
    }
}

void RigidBody::ResetVelocities() {
    if (!m_BtRigidBody) return;
    
    m_BtRigidBody->setLinearVelocity(btVector3(0, 0, 0));
    m_BtRigidBody->setAngularVelocity(btVector3(0, 0, 0));
}

void RigidBody::SetActive(bool active) {
    if (!m_BtRigidBody) return;
    
    if (active) {
        m_BtRigidBody->activate(true);
    } else {
        m_BtRigidBody->setActivationState(ISLAND_SLEEPING);
    }
}

bool RigidBody::IsActive() const {
    if (!m_BtRigidBody) return false;
    
    return m_BtRigidBody->isActive();
}

float RigidBody::GetFriction() const {
    if (!m_BtRigidBody) return 0.0f;
    return m_BtRigidBody->getFriction();
}

void RigidBody::SetFriction(float friction) {
    if (!m_BtRigidBody) return;
    m_BtRigidBody->setFriction(friction);
}

float RigidBody::GetRestitution() const {
    if (!m_BtRigidBody) return 0.0f;
    return m_BtRigidBody->getRestitution();
}

void RigidBody::SetRestitution(float restitution) {
    if (!m_BtRigidBody) return;
    m_BtRigidBody->setRestitution(restitution);
}
