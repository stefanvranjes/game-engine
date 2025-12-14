#include "PhysicsCollisionShape.h"
#include <btBulletDynamicsCommon.h>

PhysicsCollisionShape::PhysicsCollisionShape()
    : m_Shape(nullptr), m_Type(PhysicsShapeType::Box) {
}

PhysicsCollisionShape::~PhysicsCollisionShape() {
    if (m_Shape) {
        delete m_Shape;
        m_Shape = nullptr;
    }
}

PhysicsCollisionShape::PhysicsCollisionShape(PhysicsCollisionShape&& other) noexcept
    : m_Shape(other.m_Shape), m_Type(other.m_Type) {
    other.m_Shape = nullptr;
}

PhysicsCollisionShape& PhysicsCollisionShape::operator=(PhysicsCollisionShape&& other) noexcept {
    if (this != &other) {
        if (m_Shape) {
            delete m_Shape;
        }
        m_Shape = other.m_Shape;
        m_Type = other.m_Type;
        other.m_Shape = nullptr;
    }
    return *this;
}

PhysicsCollisionShape::PhysicsCollisionShape(btCollisionShape* shape, PhysicsShapeType type)
    : m_Shape(shape), m_Type(type) {
}

PhysicsCollisionShape PhysicsCollisionShape::CreateBox(const Vec3& halfExtents) {
    btVector3 btHalfExtents(halfExtents.x, halfExtents.y, halfExtents.z);
    btBoxShape* boxShape = new btBoxShape(btHalfExtents);
    return PhysicsCollisionShape(boxShape, PhysicsShapeType::Box);
}

PhysicsCollisionShape PhysicsCollisionShape::CreateSphere(float radius) {
    btSphereShape* sphereShape = new btSphereShape(radius);
    return PhysicsCollisionShape(sphereShape, PhysicsShapeType::Sphere);
}

PhysicsCollisionShape PhysicsCollisionShape::CreateCapsule(float radius, float height) {
    btCapsuleShape* capsuleShape = new btCapsuleShape(radius, height);
    return PhysicsCollisionShape(capsuleShape, PhysicsShapeType::Capsule);
}

PhysicsCollisionShape PhysicsCollisionShape::CreateCylinder(float radius, float height) {
    btVector3 halfExtents(radius, height / 2.0f, radius);
    btCylinderShape* cylinderShape = new btCylinderShape(halfExtents);
    return PhysicsCollisionShape(cylinderShape, PhysicsShapeType::Cylinder);
}

PhysicsCollisionShape PhysicsCollisionShape::CreateCompound() {
    btCompoundShape* compoundShape = new btCompoundShape();
    return PhysicsCollisionShape(compoundShape, PhysicsShapeType::Compound);
}

void PhysicsCollisionShape::AddChildShape(const PhysicsCollisionShape& childShape, const Vec3& offset, const Vec3& rotation) {
    if (m_Type != PhysicsShapeType::Compound) {
        return; // Can only add children to compound shapes
    }

    btCompoundShape* compound = static_cast<btCompoundShape*>(m_Shape);
    
    // Create transform
    btTransform transform;
    transform.setOrigin(btVector3(offset.x, offset.y, offset.z));
    
    // Convert Euler angles to quaternion
    btQuaternion quat;
    quat.setEulerZYX(rotation.z, rotation.y, rotation.x);
    transform.setRotation(quat);
    
    compound->addChildShape(transform, childShape.GetShape());
}

Vec3 PhysicsCollisionShape::GetLocalScaling() const {
    if (!m_Shape) return Vec3(1, 1, 1);
    
    btVector3 scaling = m_Shape->getLocalScaling();
    return Vec3(scaling.x(), scaling.y(), scaling.z());
}

void PhysicsCollisionShape::SetLocalScaling(const Vec3& scale) {
    if (!m_Shape) return;
    
    m_Shape->setLocalScaling(btVector3(scale.x, scale.y, scale.z));
}

float PhysicsCollisionShape::GetMargin() const {
    if (!m_Shape) return 0.04f;
    
    return m_Shape->getMargin();
}

void PhysicsCollisionShape::SetMargin(float margin) {
    if (!m_Shape) return;
    
    m_Shape->setMargin(margin);
}
