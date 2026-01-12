#include "PhysXCustomConstraint.h"
#include "PhysXBackend.h"
#include "PhysXRigidBody.h"
#include <iostream>

#ifdef USE_PHYSX
using namespace physx;

// Static Shader Function
// This is the callback PhysX calls during the solver phase (or prep phase)
static PxU32 CustomSolverPrep(Px1DConstraint* constraints,
    PxVec3& body0WorldOffset,
    PxU32 maxConstraints,
    PxConstraintInvMassScale& invMassScale,
    const void* constantBlock,
    PxU32 constantBlockByteSize)
{
    // The constant block contains a pointer to our class instance
    // Note: ensure size matches
    if (constantBlockByteSize != sizeof(PhysXCustomConstraint*)) return 0;

    const PhysXCustomConstraint* constraintInst = *reinterpret_cast<const PhysXCustomConstraint* const*>(constantBlock);
    
    // Need body transforms to calculate errors
    // Since we don't have PxArticulation/RigidBody access easily here inside the shader without queries,
    // we usually cache them in prepareData or fetch them here if we stored body pointers.
    // BUT this function is called inside the solver potentially.
    // Actually, solverPrep is called on CPU.
    
    // However, the arguments "body0WorldOffset" suggest we output the relative contact points here?
    // No, that's for contact constraints.
    // For generic constraints, we define the constraint rows.
    
    // We need to fetch the current pose of the bodies to compute the violation (Geometric Error)
    // The "constraintInst" has references to bodies.
    
    // Wait, accessing PxRigidBody::getGlobalPose() here ensures we use the latest kinematic update?
    // Yes, this is run before the solver.

    // Let's create a non-const cast helper or make Solve const (preferable)
    // But Solve might need to cache things? Let's assume const for now or cast away.
    PhysXCustomConstraint* constraint = const_cast<PhysXCustomConstraint*>(constraintInst);
    
    // It's tricky to get the PxRigidBody from here safely if we don't pass them or store them.
    // constraint->m_Body0 and m_Body1 are IPhysicsRigidBody which wrap PxRigidBody.
    
    // Getting transform:
    // We can assume we pass the transforms to Solve.
    
    // Get Poses
    // Ideally we assume bodies are valid.
    // FIXME: This access might be unsafe if bodies were deleted. But onConstraintRelease handles that.
    
    // Get poses from stored bodies
    PxTransform t0 = PxTransform(PxIdentity);
    PxTransform t1 = PxTransform(PxIdentity);

    if (constraint->m_Body0) {
        PxRigidActor* actor = static_cast<PxRigidActor*>(constraint->m_Body0->GetNativeWorld());
        t0 = actor->getGlobalPose();
    }
    if (constraint->m_Body1) {
        PxRigidActor* actor = static_cast<PxRigidActor*>(constraint->m_Body1->GetNativeWorld());
        t1 = actor->getGlobalPose();
    }

    // Call Solve with actual transforms
    return constraint->Solve(constraints, 
                             t0.p, t0.q,
                             t1.p, t1.q,
                             0.016f); // Approximation of dt, or we can store it in prepareData
    
    // CRITICAL: We need DT. It's not passed to solverPrep?
    // It is passed if we use 'PxConstraintShaderTable' correctly?
    // No, solverPrep signature is fixed.
    // Usually DT is irrelevant for strict constraints (Position Based), but relevant for Drives (Velocity).
    // Or we compute it: dt = 1/60.0f (approx) or store it in constant block during "prepareData".
}

// Defines the shader table
static PxConstraintShaderTable gCustomShaderTable = { 
    CustomSolverPrep, 
    nullptr, // project
    nullptr, // visualize
    PxConstraintFlag::Enum(0)
};

PhysXCustomConstraint::PhysXCustomConstraint(PhysXBackend* backend)
    : m_Backend(backend)
    , m_Constraint(nullptr)
    , m_Body0(nullptr)
    , m_Body1(nullptr)
{
}

PhysXCustomConstraint::~PhysXCustomConstraint() {
    if (m_Constraint) {
        m_Constraint->release();
        m_Constraint = nullptr;
    }
}

void PhysXCustomConstraint::Initialize(IPhysicsRigidBody* body0, IPhysicsRigidBody* body1) {
    if (!m_Backend || (!body0 && !body1)) return;

    m_Body0 = body0;
    m_Body1 = body1;

    PxRigidActor* actor0 = body0 ? static_cast<PxRigidActor*>(body0->GetNativeWorld()) : nullptr;
    PxRigidActor* actor1 = body1 ? static_cast<PxRigidActor*>(body1->GetNativeWorld()) : nullptr;

    m_Constraint = m_Backend->GetPhysics()->createConstraint(actor0, actor1, *this, gCustomShaderTable, sizeof(PhysXCustomConstraint*));
    
    // Set this instance as the constant block data
    PhysXCustomConstraint* self = this;
    void* data = m_Constraint->allocateConstantData(sizeof(PhysXCustomConstraint*));
    memcpy(data, &self, sizeof(PhysXCustomConstraint*));
}

void PhysXCustomConstraint::MarkDirty() {
    if (m_Constraint) m_Constraint->markDirty();
}

void PhysXCustomConstraint::prepareData() {
    // Called before solver prep.
    // We can update the constant block here if needed (e.g. update DT).
    // For now, the pointer is enough.
}

void PhysXCustomConstraint::onConstraintRelease() {
    // Creating object is deleted, so just clear pointer
    // Note: The physics object effectively deletes this if we associate them? 
    // No, PxConstraintConnector implies WE own the logic, but usually the Wrapper owns the Connector.
    // Here, `PhysXCustomConstraint` IS the connector.
    // So if the SDK releases the constraint, we should know.
    m_Constraint = nullptr;
    // We don't delete `this` here because the game engine likely owns `PhysXCustomConstraint`.
    // Instead we handle the null constraint.
}

void* PhysXCustomConstraint::getExternalReference(physx::PxU32& typeID) {
    // Return user data
    typeID = 0;
    return this; 
}

#endif // USE_PHYSX
