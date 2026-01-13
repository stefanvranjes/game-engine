#pragma once

#include "IPhysicsBackend.h"
#include "IPhysicsJoint.h"
#include <vector>
#include <memory>
#include <box2d/box2d.h>

class Renderer; // Forward decl

#ifdef USE_BOX2D

/**
 * @brief Box2D implementation of physics backend
 */
class Box2DBackend : public IPhysicsBackend {
public:
    enum class Plane2D {
        XY,
        XZ
    };

    Box2DBackend();
    ~Box2DBackend() override;

    // IPhysicsBackend implementation
    void Initialize(const Vec3& gravity) override;
    void Shutdown() override;
    void Update(float deltaTime, int subSteps = 1) override;
    
    void SetGravity(const Vec3& gravity) override;
    Vec3 GetGravity() const override;
    
    bool Raycast(const Vec3& from, const Vec3& to, PhysicsRaycastHit& hit, uint32_t filter = ~0u) override;
    
    int GetNumRigidBodies() const override;
    
    void SetDebugDrawEnabled(bool enabled) override;
    bool IsDebugDrawEnabled() const override;
    void DebugDraw(class Renderer* renderer) override;
    
    const char* GetBackendName() const override { return "Box2D"; }
    
    void* GetNativeWorld() override;

    void ApplyImpulse(void* userData, const Vec3& impulse, const Vec3& point) override;

    int OverlapSphere(const Vec3& center, float radius, std::vector<void*>& results, uint32_t filter = ~0u) override;
    int OverlapBox(const Vec3& center, const Vec3& halfExtents, const Quat& rotation, std::vector<void*>& results, uint32_t filter = ~0u) override;
    int OverlapCapsule(const Vec3& center, float radius, float halfHeight, const Quat& rotation, std::vector<void*>& results, uint32_t filter = ~0u) override;

    IPhysicsJoint* CreateJoint(const JointDef& def) override;
    void DestroyJoint(IPhysicsJoint* joint) override;

    std::shared_ptr<class IPhysicsCharacterController> CreateCharacterController();
    void DestroyCharacterController(class IPhysicsCharacterController* controller);

    // Box2D specific
    void GetActiveRigidBodies(std::vector<class IPhysicsRigidBody*>& outBodies);
    b2World* GetWorld() { return m_World; }
    
    void SetPlane(Plane2D plane);
    Plane2D GetPlane() const { return m_Plane; }

    // Helpers (Converted to members to respect plane settings)
    b2Vec2 ToBox2D(const Vec3& v) const;
    Vec3 ToVec3(const b2Vec2& v) const;

private:
    b2World* m_World;
    Plane2D m_Plane = Plane2D::XY; // Default to XY (Side-scroller)
    int32 m_VelocityIterations;
    int32 m_PositionIterations;
    bool m_Initialized;
    bool m_DebugDrawEnabled;
    
    std::vector<b2Body*> m_Bodies; // Tracking bodies if needed, or just iterate world
    
    class Box2DDebugDraw* m_DebugDraw;
    class Box2DContactListener* m_ContactListener;
    
    std::vector<IPhysicsJoint*> m_Joints;
    std::vector<class IPhysicsCharacterController*> m_Controllers;
};

#endif // USE_BOX2D
