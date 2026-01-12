#include "Box2DShape.h"

#ifdef USE_BOX2D

Box2DShape::Box2DShape(PhysicsShapeType type)
    : m_Type(type)
    , m_Scale(1,1,1)
    , m_IsTrigger(false)
{
}

std::shared_ptr<Box2DShape> Box2DShape::CreateBox(const Vec3& halfExtents) {
    auto shape = std::make_shared<Box2DShape>(PhysicsShapeType::Box);
    shape->m_BoxShape.SetAsBox(halfExtents.x, halfExtents.y);
    shape->m_Shape = &shape->m_BoxShape;
    return shape;
}

std::shared_ptr<Box2DShape> Box2DShape::CreateSphere(float radius) {
    auto shape = std::make_shared<Box2DShape>(PhysicsShapeType::Sphere);
    shape->m_CircleShape.m_radius = radius;
    shape->m_Shape = &shape->m_CircleShape;
    return shape;
}

std::shared_ptr<Box2DShape> Box2DShape::CreateCapsule(float radius, float height) {
    // Box2D doesn't have a capsule. We approximate with a box for now, or circle.
    // If height is significant, use box. Dimensions: radius*2 width, height tall?
    // "Capsule" usually allows rolling.
    // Let's use a Box for simplicity for now as per plan.
    // Half extents... radius is usually thickness.
    // Dimensions x = radius, y = height/2 + radius?
    // Let's assume vertical capsule.
    return CreateBox(Vec3(radius, height * 0.5f + radius, 0));
}

#endif // USE_BOX2D
