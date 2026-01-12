#include "PhysXShape.h"

#ifdef USE_PHYSX

#include "PhysXBackend.h"
#include <PxPhysicsAPI.h>
#include <algorithm>

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
    if (!m_Shape) {
        m_LocalScaling = scale;
        return;
    }

    // Calculate ratio relative to current scale
    // (Avoiding divide by zero)
    Vec3 ratio(1.0f, 1.0f, 1.0f);
    if (std::abs(m_LocalScaling.x) > 1e-6f) ratio.x = scale.x / m_LocalScaling.x;
    if (std::abs(m_LocalScaling.y) > 1e-6f) ratio.y = scale.y / m_LocalScaling.y;
    if (std::abs(m_LocalScaling.z) > 1e-6f) ratio.z = scale.z / m_LocalScaling.z;

    PxGeometryHolder geom = m_Shape->getGeometry();
    
    switch (geom.getType()) {
        case PxGeometryType::eBOX: {
            PxBoxGeometry box = geom.box();
            box.halfExtents.x *= ratio.x;
            box.halfExtents.y *= ratio.y;
            box.halfExtents.z *= ratio.z;
            m_Shape->setGeometry(box);
            break;
        }
        case PxGeometryType::eSPHERE: {
            PxSphereGeometry sphere = geom.sphere();
            // Uniform scaling for sphere (use max)
            float maxScale = std::max(ratio.x, std::max(ratio.y, ratio.z));
            sphere.radius *= maxScale;
            m_Shape->setGeometry(sphere);
            break;
        }
        case PxGeometryType::eCAPSULE: {
            PxCapsuleGeometry capsule = geom.capsule();
            // Capsule extends along X axis by default in PhysX. 
            // We assume scaling maps: X -> Height (HalfHeight), Y/Z -> Radius
            float radiusScale = std::max(ratio.y, ratio.z);
            capsule.radius *= radiusScale;
            capsule.halfHeight *= ratio.x;
            m_Shape->setGeometry(capsule);
            break;
        }
        default:
            // Other shapes (Convex, Mesh) require more complex handling (re-baking or scale parameter if supported)
            // PxConvexMeshGeometry/PxTriangleMeshGeometry have a scale parameter.
            // But modifying it requires accessing the mesh scale structure.
            break;
    }

    m_LocalScaling = scale;
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

void PhysXShape::SetTrigger(bool isTrigger) {
    if (m_Shape) {
        if (isTrigger) {
            m_Shape->setFlag(PxShapeFlag::eSIMULATION_SHAPE, false);
            m_Shape->setFlag(PxShapeFlag::eTRIGGER_SHAPE, true);
        } else {
            m_Shape->setFlag(PxShapeFlag::eTRIGGER_SHAPE, false);
            m_Shape->setFlag(PxShapeFlag::eSIMULATION_SHAPE, true);
        }
    }
}

bool PhysXShape::IsTrigger() const {
    if (m_Shape) {
        return (m_Shape->getFlags() & PxShapeFlag::eTRIGGER_SHAPE);
    }
    return false;
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
