#include "PhysXShape.h"

#ifdef USE_PHYSX

#include "PhysXBackend.h"
#include <PxPhysicsAPI.h>

using namespace physx;

PhysXShape::PhysXShape(PhysXBackend* backend, PhysicsShapeType type)
    : m_Backend(backend)
    , m_Shape(nullptr)
    , m_Type(type)
    , m_LocalScaling(1, 1, 1)
{
}

PhysXShape::~PhysXShape() {
    if (m_Shape) {
        m_Shape->release();
        m_Shape = nullptr;
    }
}

Vec3 PhysXShape::GetLocalScaling() const {
    return m_LocalScaling;
}

void PhysXShape::SetLocalScaling(const Vec3& scale) {
    m_LocalScaling = scale;
    // PhysX handles scaling differently per geometry type
    // This would need to be implemented per-shape-type
}

float PhysXShape::GetMargin() const {
    if (m_Shape) {
        return m_Shape->getContactOffset();
    }
    return 0.04f;
}

void PhysXShape::SetMargin(float margin) {
    if (m_Shape) {
        m_Shape->setContactOffset(margin);
    }
}

void* PhysXShape::GetNativeShape() {
    return m_Shape;
}

void PhysXShape::AddChildShape(std::shared_ptr<IPhysicsShape> child, const Vec3& position, const Vec3& rotation) {
    // PhysX doesn't have compound shapes in the same way as Bullet
    // Multiple shapes can be attached to a single actor instead
    m_ChildShapes.push_back(child);
}

std::shared_ptr<PhysXShape> PhysXShape::CreateBox(PhysXBackend* backend, const Vec3& halfExtents) {
    auto shape = std::make_shared<PhysXShape>(backend, PhysicsShapeType::Box);
    
    PxPhysics* physics = backend->GetPhysics();
    PxMaterial* material = backend->GetDefaultMaterial();
    
    if (physics && material) {
        PxBoxGeometry geometry(halfExtents.x, halfExtents.y, halfExtents.z);
        shape->m_Shape = physics->createShape(geometry, *material);
    }
    
    return shape;
}

std::shared_ptr<PhysXShape> PhysXShape::CreateSphere(PhysXBackend* backend, float radius) {
    auto shape = std::make_shared<PhysXShape>(backend, PhysicsShapeType::Sphere);
    
    PxPhysics* physics = backend->GetPhysics();
    PxMaterial* material = backend->GetDefaultMaterial();
    
    if (physics && material) {
        PxSphereGeometry geometry(radius);
        shape->m_Shape = physics->createShape(geometry, *material);
    }
    
    return shape;
}

std::shared_ptr<PhysXShape> PhysXShape::CreateCapsule(PhysXBackend* backend, float radius, float height) {
    auto shape = std::make_shared<PhysXShape>(backend, PhysicsShapeType::Capsule);
    
    PxPhysics* physics = backend->GetPhysics();
    PxMaterial* material = backend->GetDefaultMaterial();
    
    if (physics && material) {
        // PhysX capsule is defined by radius and half-height
        PxCapsuleGeometry geometry(radius, height * 0.5f);
        shape->m_Shape = physics->createShape(geometry, *material);
    }
    
    return shape;
}

std::shared_ptr<PhysXShape> PhysXShape::CreateCylinder(PhysXBackend* backend, float radius, float height) {
    // PhysX doesn't have a native cylinder, use convex mesh or approximate with capsule
    // For now, we'll use a capsule as approximation
    return CreateCapsule(backend, radius, height);
}

std::shared_ptr<PhysXShape> PhysXShape::CreateCompound(PhysXBackend* backend) {
    auto shape = std::make_shared<PhysXShape>(backend, PhysicsShapeType::Compound);
    // Compound shapes in PhysX are handled by attaching multiple shapes to one actor
    return shape;
}

#endif // USE_PHYSX
