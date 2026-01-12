#pragma once

#include "Math/Vec3.h"
#include "IPhysicsRigidBody.h"

enum class JointType {
    Revolute,   // Hinge
    Prismatic,  // Slider
    Distance,   // Fixed or spring distance
    Fixed       // Weld
};

struct JointDef {
    JointType type;
    IPhysicsRigidBody* bodyA = nullptr;
    IPhysicsRigidBody* bodyB = nullptr;
    Vec3 anchorA; // Local or World? Usually World for initialization convenience
    Vec3 anchorB; // Usually World, backend converts to local
    bool collideConnected = false;
    
    // Revolute specific
    bool enableLimit = false;
    float lowerAngle = 0.0f; // Radians
    float upperAngle = 0.0f; // Radians
    bool enableMotor = false;
    float motorSpeed = 0.0f;
    float maxMotorTorque = 0.0f;
    
    // Distance/Prismatic
    float length = 0.0f;
    float damping = 0.0f;
    float stiffness = 0.0f;
};

class IPhysicsJoint {
public:
    virtual ~IPhysicsJoint() = default;

    virtual JointType GetType() const = 0;
    virtual IPhysicsRigidBody* GetBodyA() const = 0;
    virtual IPhysicsRigidBody* GetBodyB() const = 0;
    
    // Common
    virtual void SetCollideConnected(bool collide) = 0;
    
    // Revolute specific getters/setters (could use RTTI or virtuals, keeping it simple)
    // If backend doesn't support, these do nothing
    virtual void SetMotorEnabled(bool enabled) {}
    virtual void SetMotorSpeed(float speed) {}
    virtual void SetMaxMotorTorque(float torque) {}
    virtual void SetLimitsEnabled(bool enabled) {}
    virtual void SetLimits(float lower, float upper) {}
};
