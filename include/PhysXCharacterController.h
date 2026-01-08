#pragma once

#include "IPhysicsCharacterController.h"
#include "Math/Vec3.h"
#include <memory>

#ifdef USE_PHYSX

namespace physx {
    class PxController;
    class PxControllerManager;
}

class PhysXBackend;

/**
 * @brief PhysX implementation of character controller
 */
class PhysXCharacterController : public IPhysicsCharacterController {
public:
    PhysXCharacterController(PhysXBackend* backend);
    ~PhysXCharacterController() override;

    // IPhysicsCharacterController implementation
    void Initialize(std::shared_ptr<IPhysicsShape> shape, float mass, float stepHeight) override;
    void Update(float deltaTime, const Vec3& gravity) override;
    void SetWalkDirection(const Vec3& direction) override;
    Vec3 GetWalkDirection() const override;
    void Jump(const Vec3& impulse) override;
    bool IsGrounded() const override;
    Vec3 GetPosition() const override;
    void SetPosition(const Vec3& position) override;
    float GetVerticalVelocity() const override;
    void SetVerticalVelocity(float velocity) override;
    void SetMaxWalkSpeed(float speed) override;
    float GetMaxWalkSpeed() const override;
    void SetFallSpeed(float speed) override;
    float GetFallSpeed() const override;
    void SetStepHeight(float height) override;
    float GetStepHeight() const override;
    void* GetNativeController() override;

private:
    PhysXBackend* m_Backend;
    physx::PxController* m_Controller;
    physx::PxControllerManager* m_ControllerManager;
    
    Vec3 m_WalkDirection;
    float m_VerticalVelocity;
    float m_MaxWalkSpeed;
    float m_FallSpeed;
    float m_StepHeight;
    bool m_IsGrounded;
};

#endif // USE_PHYSX
