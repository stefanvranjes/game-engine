#include "Box2DJoint.h"
#include "Box2DBackend.h"
#include "Box2DRigidBody.h"

#ifdef USE_BOX2D

Box2DJoint::Box2DJoint(Box2DBackend* backend, b2Joint* joint, JointType type)
    : m_Backend(backend), m_Joint(joint), m_Type(type) {
}

Box2DJoint::~Box2DJoint() {
    // Backend manages destruction via DestroyJoint usually, or we do it here if detached.
    // NOTE: Application/User usually calls DestroyJoint on backend, which deletes this object AND native joint.
    // If native joint is still valid, we shouldn't delete it here unless we own it?
    // Box2D World owns joints. So we must call world->DestroyJoint(joint).
    // This destructor is called by DestroyJoint likely.
}

IPhysicsRigidBody* Box2DJoint::GetBodyA() const {
    if (!m_Joint) return nullptr;
    b2Body* body = m_Joint->GetBodyA();
    if (body) return reinterpret_cast<Box2DRigidBody*>(body->GetUserData().pointer);
    return nullptr;
}

IPhysicsRigidBody* Box2DJoint::GetBodyB() const {
    if (!m_Joint) return nullptr;
    b2Body* body = m_Joint->GetBodyB();
    if (body) return reinterpret_cast<Box2DRigidBody*>(body->GetUserData().pointer);
    return nullptr;
}

void Box2DJoint::SetCollideConnected(bool collide) {
    // Box2D doesn't support changing collideConnected after creation easily?
    // b2JointDef has collideConnected. b2Joint doesn't have SetCollideConnected?
    // Checked docs: b2Joint usually fixed at creation.
    // We can't change it at runtime in Box2D easily without recreating.
    // We'll leave empty or log warning.
}

// Revolute
Box2DRevoluteJoint::Box2DRevoluteJoint(Box2DBackend* backend, b2RevoluteJoint* joint)
    : Box2DJoint(backend, joint, JointType::Revolute) {
}

void Box2DRevoluteJoint::SetMotorEnabled(bool enabled) {
    static_cast<b2RevoluteJoint*>(m_Joint)->EnableMotor(enabled);
}

void Box2DRevoluteJoint::SetMotorSpeed(float speed) {
    static_cast<b2RevoluteJoint*>(m_Joint)->SetMotorSpeed(speed);
}

void Box2DRevoluteJoint::SetMaxMotorTorque(float torque) {
    static_cast<b2RevoluteJoint*>(m_Joint)->SetMaxMotorTorque(torque);
}

void Box2DRevoluteJoint::SetLimitsEnabled(bool enabled) {
    static_cast<b2RevoluteJoint*>(m_Joint)->EnableLimit(enabled);
}

void Box2DRevoluteJoint::SetLimits(float lower, float upper) {
    static_cast<b2RevoluteJoint*>(m_Joint)->SetLimits(lower, upper);
}

#endif // USE_BOX2D
