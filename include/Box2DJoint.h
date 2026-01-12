#pragma once

#include "IPhysicsJoint.h"
#include <box2d/box2d.h>

#ifdef USE_BOX2D

class Box2DBackend;

class Box2DJoint : public IPhysicsJoint {
public:
    Box2DJoint(Box2DBackend* backend, b2Joint* joint, JointType type);
    virtual ~Box2DJoint();

    JointType GetType() const override { return m_Type; }
    IPhysicsRigidBody* GetBodyA() const override;
    IPhysicsRigidBody* GetBodyB() const override;
    void SetCollideConnected(bool collide) override;

    b2Joint* GetNativeJoint() { return m_Joint; }

protected:
    Box2DBackend* m_Backend;
    b2Joint* m_Joint;
    JointType m_Type;
};

class Box2DRevoluteJoint : public Box2DJoint {
public:
    Box2DRevoluteJoint(Box2DBackend* backend, b2RevoluteJoint* joint);
    
    void SetMotorEnabled(bool enabled) override;
    void SetMotorSpeed(float speed) override;
    void SetMaxMotorTorque(float torque) override;
    void SetLimitsEnabled(bool enabled) override;
    void SetLimits(float lower, float upper) override;
};

#endif // USE_BOX2D
