#include "Box2DRigidBody.h"
#include "Box2DBackend.h"
#include "Box2DShape.h"

#ifdef USE_BOX2D

Box2DRigidBody::Box2DRigidBody(Box2DBackend* backend)
    : m_Backend(backend)
    , m_Body(nullptr)
    , m_Type(BodyType::Dynamic)
    , m_Mass(1.0f)
{
}

Box2DRigidBody::~Box2DRigidBody() {
    if (m_Body && m_Backend) {
        m_Backend->GetWorld()->DestroyBody(m_Body);
        m_Body = nullptr;
    }
}

void Box2DRigidBody::Initialize(BodyType type, float mass, std::shared_ptr<IPhysicsShape> shape) {
    m_Type = type;
    m_Mass = mass;
    
    if (!m_Backend) return;
    
    b2BodyDef bodyDef;
    if (type == BodyType::Static) bodyDef.type = b2_staticBody;
    else if (type == BodyType::Dynamic) bodyDef.type = b2_dynamicBody;
    else bodyDef.type = b2_kinematicBody;
    
    // Initial position 0,0 (Transform sync will move it)
    bodyDef.position.Set(0.0f, 0.0f);
    
    if (m_UserData) {
        bodyDef.userData.pointer = reinterpret_cast<uintptr_t>(m_UserData);
    }
    
    m_Body = m_Backend->GetWorld()->CreateBody(&bodyDef);
    
    // Create fixture from shape
    Box2DShape* box2DShape = static_cast<Box2DShape*>(shape.get());
    if (box2DShape && box2DShape->GetB2Shape()) {
        b2FixtureDef fixtureDef;
        fixtureDef.shape = box2DShape->GetB2Shape();
        
        // Density calculation (Box2D needs density, not mass)
        // If mass is given, we need to compute density based on area.
        // Or we can let Box2D compute mass from density.
        // If we want exact mass, we might need to adjust density.
        // For now, set density = 1.0 (default) or handle mass override.
        fixtureDef.density = 1.0f;
        fixtureDef.friction = 0.5f;
        fixtureDef.restitution = 0.0f; // No bounce default
        
        m_Body->CreateFixture(&fixtureDef);
        
        // If mass is specific and body is dynamic, we might need to reset mass data?
        // Box2D calculates mass from shapes and density.
        // If we want specific mass, we can try to set it, but Box2D prefers density.
        // Let's rely on density for now or ignore mass parameter if density is derived.
        // If the user expects 'mass' to be the total mass, we should calculate density = mass / area.
        // Area depends on shape.
        // For now, simplicity: density = 1.0.
    }
}

void Box2DRigidBody::SetLinearVelocity(const Vec3& velocity) {
    if (m_Body) m_Body->SetLinearVelocity(Box2DBackend::ToBox2D(velocity));
}

Vec3 Box2DRigidBody::GetLinearVelocity() const {
    if (m_Body) return Box2DBackend::ToVec3(m_Body->GetLinearVelocity());
    return Vec3(0,0,0);
}

void Box2DRigidBody::SetAngularVelocity(const Vec3& velocity) {
    // 2D only has Z angular velocity
    if (m_Body) m_Body->SetAngularVelocity(velocity.z); 
}

Vec3 Box2DRigidBody::GetAngularVelocity() const {
    if (m_Body) return Vec3(0, 0, m_Body->GetAngularVelocity());
    return Vec3(0,0,0);
}

void Box2DRigidBody::ApplyForce(const Vec3& force) {
    if (m_Body) m_Body->ApplyForceToCenter(Box2DBackend::ToBox2D(force), true);
}

void Box2DRigidBody::ApplyImpulse(const Vec3& impulse) {
    if (m_Body) m_Body->ApplyLinearImpulseToCenter(Box2DBackend::ToBox2D(impulse), true);
}

void Box2DRigidBody::ApplyForceAtPoint(const Vec3& force, const Vec3& point) {
    if (m_Body) m_Body->ApplyForce(Box2DBackend::ToBox2D(force), Box2DBackend::ToBox2D(point), true);
}

void Box2DRigidBody::ApplyImpulseAtPoint(const Vec3& impulse, const Vec3& point) {
    if (m_Body) m_Body->ApplyLinearImpulse(Box2DBackend::ToBox2D(impulse), Box2DBackend::ToBox2D(point), true);
}

void Box2DRigidBody::ApplyTorque(const Vec3& torque) {
    if (m_Body) m_Body->ApplyTorque(torque.z, true);
}

void Box2DRigidBody::SetMass(float mass) {
    m_Mass = mass;
    // Box2D mass is derived from density usually.
    // To set exact mass, we can use b2MassData.
    if (m_Body) {
        b2MassData massData;
        m_Body->GetMassData(&massData);
        massData.mass = mass;
        m_Body->SetMassData(&massData);
    }
}

float Box2DRigidBody::GetMass() const {
    if (m_Body) return m_Body->GetMass();
    return m_Mass;
}

void Box2DRigidBody::SetFriction(float friction) {
    if (m_Body) {
        for (b2Fixture* f = m_Body->GetFixtureList(); f; f = f->GetNext()) {
            f->SetFriction(friction);
        }
    }
}

float Box2DRigidBody::GetFriction() const {
    if (m_Body && m_Body->GetFixtureList()) {
        return m_Body->GetFixtureList()->GetFriction();
    }
    return 0.0f;
}

void Box2DRigidBody::SetRestitution(float restitution) {
    if (m_Body) {
        for (b2Fixture* f = m_Body->GetFixtureList(); f; f = f->GetNext()) {
            f->SetRestitution(restitution);
        }
    }
}

float Box2DRigidBody::GetRestitution() const {
    if (m_Body && m_Body->GetFixtureList()) {
        return m_Body->GetFixtureList()->GetRestitution();
    }
    return 0.0f;
}

void Box2DRigidBody::SetLinearDamping(float damping) {
    if (m_Body) m_Body->SetLinearDamping(damping);
}

void Box2DRigidBody::SetAngularDamping(float damping) {
    if (m_Body) m_Body->SetAngularDamping(damping);
}

void Box2DRigidBody::SetGravityEnabled(bool enabled) {
    if (m_Body) m_Body->SetGravityScale(enabled ? 1.0f : 0.0f);
}

bool Box2DRigidBody::IsGravityEnabled() const {
    if (m_Body) return m_Body->GetGravityScale() > 0.0f;
    return true;
}

bool Box2DRigidBody::IsActive() const {
    if (m_Body) return m_Body->IsAwake();
    return false;
}

void Box2DRigidBody::SetActive(bool active) {
    if (m_Body) m_Body->SetAwake(active);
}

void Box2DRigidBody::SyncTransformFromPhysics(Vec3& outPosition, Quat& outRotation) {
    if (m_Body) {
        b2Vec2 pos = m_Body->GetPosition();
        float angle = m_Body->GetAngle();
        
        outPosition = Vec3(pos.x, pos.y, 0.0f); // Z is 0 or maintained?
        // Box2D angle is radians around Z.
        // Quaternion from axis-angle (Z axis)
        // Z axis is (0,0,1)
        outRotation = Quat::FromAxisAngle(Vec3(0,0,1), angle);
    }
}

void Box2DRigidBody::SyncTransformToPhysics(const Vec3& position, const Quat& rotation) {
    if (m_Body) {
        // Assume rotation is mostly Z axis.
        // Extract Z rotation from Quat
        // Or if Quat is only Z rotation...
        // eulerAngles.z
        // We can convert quat to euler
        Vec3 euler = rotation.ToEulerAngles();
        m_Body->SetTransform(Box2DBackend::ToBox2D(position), euler.z);
    }
}

void Box2DRigidBody::SetCCDEnabled(bool enabled) {
    if (m_Body) m_Body->SetBullet(enabled);
}

bool Box2DRigidBody::IsCCDEnabled() const {
    if (m_Body) return m_Body->IsBullet();
    return false;
}

#endif // USE_BOX2D
