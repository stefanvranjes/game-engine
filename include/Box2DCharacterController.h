#pragma once

#include "IPhysicsCharacterController.h"
#include <box2d/box2d.h>

#ifdef USE_BOX2D

class Box2DBackend;

class Box2DCharacterController : public IPhysicsCharacterController {
public:
    Box2DCharacterController(Box2DBackend* backend);
    virtual ~Box2DCharacterController();

    void Initialize(std::shared_ptr<IPhysicsShape> shape, float mass, float stepHeight) override;
    void Update(float deltaTime, const Vec3& gravity) override;
    
    void SetWalkDirection(const Vec3& direction) override;
    Vec3 GetWalkDirection() const override { return m_WalkDirection; }
    
    void Jump(const Vec3& impulse) override;
    bool IsGrounded() const override;
    
    Vec3 GetPosition() const override;
    void SetPosition(const Vec3& position) override;
    
    float GetVerticalVelocity() const override;
    void SetVerticalVelocity(float velocity) override;
    
    void SetMaxWalkSpeed(float speed) override { m_MaxWalkSpeed = speed; }
    float GetMaxWalkSpeed() const override { return m_MaxWalkSpeed; }
    
    void SetFallSpeed(float speed) override { m_FallSpeed = speed; }
    float GetFallSpeed() const override { return m_FallSpeed; }
    
    void SetStepHeight(float height) override { m_StepHeight = height; }
    float GetStepHeight() const override { return m_StepHeight; }
    
    void* GetNativeController() override { return m_Body; }
    
    // Implement remaining pure virtuals
    void SetSlopeLimit(float slopeLimit) override { m_SlopeLimit = slopeLimit; }
    float GetSlopeLimit() const override { return m_SlopeLimit; }
    
    void SetContactOffset(float offset) override { m_ContactOffset = offset; }
    float GetContactOffset() const override { return m_ContactOffset; }
    
    void Resize(float height) override;
    
    void SetUpDirection(const Vec3& up) override { m_UpDirection = up; }
    Vec3 GetUpDirection() const override { return m_UpDirection; }
    
    void SetPushForce(float force) override { m_PushForce = force; }
    float GetPushForce() const override { return m_PushForce; }

private:
    Box2DBackend* m_Backend;
    b2Body* m_Body = nullptr;
    
    Vec3 m_WalkDirection;
    float m_MaxWalkSpeed = 5.0f;
    float m_FallSpeed = 20.0f;
    float m_StepHeight = 0.3f;
    float m_SlopeLimit = 45.0f; // Degrees
    float m_ContactOffset = 0.01f;
    Vec3 m_UpDirection = Vec3(0, 1, 0);
    float m_PushForce = 1.0f;
    
    bool m_IsGrounded = false;
    
    void CheckGrounded();
};

#endif // USE_BOX2D
