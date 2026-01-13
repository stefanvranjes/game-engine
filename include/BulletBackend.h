#pragma once

#include "IPhysicsBackend.h"
#include "Math/Vec3.h"
#include "Math/Quat.h"
#include <memory>
#include <vector>

#ifdef USE_BULLET

// Bullet forward declarations
class btDynamicsWorld;
class btBroadphaseInterface;
class btCollisionDispatcher;
class btConstraintSolver;
class btDefaultCollisionConfiguration;

class IPhysicsRigidBody;
class IPhysicsCharacterController;

/**
 * @brief Bullet3D implementation of physics backend
 * 
 * Wraps existing Bullet3D physics code in the abstraction layer
 */
class BulletBackend : public IPhysicsBackend {
public:
    BulletBackend();
    ~BulletBackend() override;

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
    const char* GetBackendName() const override { return "Bullet3D"; }
    void* GetNativeWorld() override;

    // Bullet-specific methods
    btDynamicsWorld* GetDynamicsWorld() const { return m_DynamicsWorld; }

    // Registration methods
    void RegisterRigidBody(IPhysicsRigidBody* body);
    void UnregisterRigidBody(IPhysicsRigidBody* body);
    void RegisterCharacterController(IPhysicsCharacterController* controller);
    void UnregisterCharacterController(IPhysicsCharacterController* controller);

private:
    btDynamicsWorld* m_DynamicsWorld;
    btBroadphaseInterface* m_Broadphase;
    btCollisionDispatcher* m_Dispatcher;
    btConstraintSolver* m_ConstraintSolver;
    btDefaultCollisionConfiguration* m_CollisionConfiguration;

    std::vector<IPhysicsRigidBody*> m_RigidBodies;
    std::vector<IPhysicsCharacterController*> m_CharacterControllers;

    bool m_Initialized;
    bool m_DebugDrawEnabled;

    void UpdateCharacterControllers(float deltaTime);
};

#endif // USE_BULLET
