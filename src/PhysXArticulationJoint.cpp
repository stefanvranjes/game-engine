#include "PhysXArticulationJoint.h"

#ifdef USE_PHYSX
#include <PxPhysicsAPI.h>

using namespace physx;

PhysXArticulationJoint::PhysXArticulationJoint(PxArticulationJointReducedCoordinate* joint)
    : m_Joint(joint) {
}

PhysXArticulationJoint::~PhysXArticulationJoint() {
    // Joint is owned by the link/articulation in PhysX, so we don't delete it explicitly
}

void PhysXArticulationJoint::SetMotion(Axis axis, Motion motion) {
    if (!m_Joint) return;

    PxArticulationMotion::Enum pxMotion = PxArticulationMotion::eLOCKED;
    switch (motion) {
        case Motion::Locked: pxMotion = PxArticulationMotion::eLOCKED; break;
        case Motion::Limited: pxMotion = PxArticulationMotion::eLIMITED; break;
        case Motion::Free: pxMotion = PxArticulationMotion::eFREE; break;
    }

    m_Joint->setJointMotion(static_cast<PxArticulationAxis::Enum>(axis), pxMotion);
}

void PhysXArticulationJoint::SetTwistLimit(float lower, float upper) {
    if (!m_Joint) return;
    // For twist (X), limits are usually set on X axis if motion is Limited
    m_Joint->setLimitParams(PxArticulationAxis::eTWIST, PxArticulationLimit(lower, upper));
}

void PhysXArticulationJoint::SetSwingLimit(float yLimit, float zLimit) {
    if (!m_Joint) return;
    m_Joint->setLimitParams(PxArticulationAxis::eSWING1, PxArticulationLimit(-yLimit, yLimit));
    m_Joint->setLimitParams(PxArticulationAxis::eSWING2, PxArticulationLimit(-zLimit, zLimit));
}

void PhysXArticulationJoint::SetDrive(Axis axis, float stiffness, float damping, float maxForce, DriveType type) {
    if (!m_Joint) return;

    PxArticulationDrive drive;
    drive.stiffness = stiffness;
    drive.damping = damping;
    drive.maxForce = maxForce;
    
    switch (type) {
        case DriveType::None: drive.driveType = PxArticulationDriveType::eNONE; break;
        case DriveType::Target: drive.driveType = PxArticulationDriveType::eTARGET; break;
        case DriveType::Velocity: drive.driveType = PxArticulationDriveType::eVELOCITY; break;
    }

    m_Joint->setDriveParams(static_cast<PxArticulationAxis::Enum>(axis), drive);
}

void PhysXArticulationJoint::SetDriveTarget(Axis axis, float target) {
    if (!m_Joint) return;
    m_Joint->setDriveTarget(static_cast<PxArticulationAxis::Enum>(axis), target);
}

void PhysXArticulationJoint::SetDriveTarget(const Quat& target) {
    if (!m_Joint) return;
    m_Joint->setDriveTarget(PxTransform(PxVec3(0,0,0), PxQuat(target.x, target.y, target.z, target.w)));
}

void PhysXArticulationJoint::SetDriveVelocity(Axis axis, float velocity) {
    if (!m_Joint) return;
    m_Joint->setDriveVelocity(static_cast<PxArticulationAxis::Enum>(axis), velocity);
}

void PhysXArticulationJoint::SetFrictionCoefficient(float coefficient) {
    if (!m_Joint) return;
    m_Joint->setFrictionCoefficient(coefficient);
}

#endif // USE_PHYSX
