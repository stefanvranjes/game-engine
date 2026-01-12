#pragma once

#include "Math/Vec3.h"
#include "Math/Quat.h"

#ifdef USE_PHYSX
namespace physx {
    class PxArticulationJointReducedCoordinate;
}

class PhysXArticulationJoint {
public:
    enum class Motion {
        Locked,
        Limited,
        Free
    };

    enum class Axis {
        Twist = 0, // X
        Swing1 = 1, // Y
        Swing2 = 2, // Z
        X = 3,
        Y = 4,
        Z = 5
    };

    enum class DriveType {
        None,
        Target,
        Velocity
    };

    PhysXArticulationJoint(physx::PxArticulationJointReducedCoordinate* joint);
    ~PhysXArticulationJoint();

    // Motion limits
    void SetMotion(Axis axis, Motion motion);
    void SetTwistLimit(float lower, float upper);
    void SetSwingLimit(float yLimit, float zLimit);
    
    // Drive settings
    void SetDrive(Axis axis, float stiffness, float damping, float maxForce, DriveType type = DriveType::Target);
    void SetDriveTarget(Axis axis, float target);
    void SetDriveTarget(const Quat& target);
    void SetDriveVelocity(Axis axis, float velocity);

    // Friction
    void SetFrictionCoefficient(float coefficient);

    // Properties
    physx::PxArticulationJointReducedCoordinate* GetNativeJoint() const { return m_Joint; }

private:
    physx::PxArticulationJointReducedCoordinate* m_Joint;
};
#endif // USE_PHYSX
