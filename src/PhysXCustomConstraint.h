#pragma once

#include "Math/Vec3.h"
#include "Math/Quat.h"

#ifdef USE_PHYSX
#include <PxPhysicsAPI.h>

class PhysXBackend;
class IPhysicsRigidBody;

/**
 * @brief Base class for custom PhysX constraints.
 * Implement "PrepareConstraints" to define the rows of the constraint solver.
 */
class PhysXCustomConstraint : public physx::PxConstraintConnector {
public:
    PhysXCustomConstraint(PhysXBackend* backend);
    virtual ~PhysXCustomConstraint();

    void Initialize(IPhysicsRigidBody* body0, IPhysicsRigidBody* body1);
    
    // Notify PhysX that parameters have changed and solver prep needs to run again
    void MarkDirty();

    // PxConstraintConnector interface
    virtual void prepareData() override;
    virtual bool updatePvdProperties(physx::PxPhysXGpu&, physx::PxPvd*, const physx::PxConstraint*, physx::PxPvdUpdateType::Enum) const override { return true; }
    virtual void onConstraintRelease() override;
    virtual void onComShift(physx::PxU32 actor) override {}
    virtual void onOriginShift(const physx::PxVec3& shift) override {}
    virtual void* getExternalReference(physx::PxU32& typeID) override;

    // Custom Logic to be implemented by subclass
    // Populates the 1D constraint rows.
    // Return the number of rows used.
    virtual uint32_t Solve(physx::Px1DConstraint* constraints, 
                          const physx::PxVec3& body0Pos, const physx::PxQuat& body0Rot,
                          const physx::PxVec3& body1Pos, const physx::PxQuat& body1Rot,
                          float dt) = 0;

protected:
    friend physx::PxU32 CustomSolverPrep(physx::Px1DConstraint*, physx::PxVec3&, physx::PxU32, physx::PxConstraintInvMassScale&, const void*, physx::PxU32);

    PhysXBackend* m_Backend;
    physx::PxConstraint* m_Constraint;
    IPhysicsRigidBody* m_Body0;
    IPhysicsRigidBody* m_Body1;
};

#endif // USE_PHYSX
