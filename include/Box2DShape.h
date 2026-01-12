#pragma once

#include "IPhysicsShape.h"
#include <box2d/box2d.h>
#include <memory>
#include <vector>

#ifdef USE_BOX2D

class Box2DBackend;

class Box2DShape : public IPhysicsShape {
public:
    Box2DShape(PhysicsShapeType type);
    ~Box2DShape() override = default;

    PhysicsShapeType GetType() const override { return m_Type; }
    
    Vec3 GetLocalScaling() const override { return m_Scale; }
    void SetLocalScaling(const Vec3& scale) override { m_Scale = scale; }
    
    // Box2D doesn't use margin in the same way, but we keep interface
    float GetMargin() const override { return 0.0f; }
    void SetMargin(float margin) override {} 
    
    void SetTrigger(bool isTrigger) override { m_IsTrigger = isTrigger; }
    bool IsTrigger() const override { return m_IsTrigger; }
    
    void* GetNativeShape() override { return &m_ShapeDef; } // Return pointer to internal structure?
    
    void AddChildShape(std::shared_ptr<IPhysicsShape> child, const Vec3& position, const Vec3& rotation) override {}

    // Factory methods
    static std::shared_ptr<Box2DShape> CreateBox(const Vec3& halfExtents);
    static std::shared_ptr<Box2DShape> CreateSphere(float radius);
    static std::shared_ptr<Box2DShape> CreateCapsule(float radius, float height);
    
    // Internal access for RigidBody
    const b2Shape* GetB2Shape() const { return m_Shape; }
    b2Shape* GetB2Shape() { return m_Shape; }

private:
    PhysicsShapeType m_Type;
    Vec3 m_Scale;
    bool m_IsTrigger;
    
    b2Shape* m_Shape = nullptr; // Raw pointer managed by this class
    
    // Storage for specific shapes
    b2PolygonShape m_BoxShape;
    b2CircleShape m_CircleShape;
    // We point m_Shape to one of these
    
    void* m_ShapeDef = nullptr; // Placeholder if needed
};

#endif // USE_BOX2D
