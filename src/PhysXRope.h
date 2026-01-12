#pragma once

#include "PhysXArticulation.h"
#include "PhysXArticulationLink.h"
#include "Math/Vec3.h"
#include <vector>

class PhysXBackend;
class IPhysicsRigidBody;

class PhysXRope {
public:
    PhysXRope(PhysXBackend* backend);
    ~PhysXRope();

    // Create a rope between two points
    // segments: number of links
    // radius: thickness of rope
    // totalMass: total mass of rope (distributed among segments)
    // thickness: if true, uses capsule shapes, else uses very thin capsules
    void Initialize(const Vec3& start, const Vec3& end, int segments, float radius, float totalMass);

    // Attach ends to dynamic/static bodies
    // If body is nullptr, attaches to the world (static) at the current position of the rope end
    void AttachStart(IPhysicsRigidBody* body, const Vec3& localOffset = Vec3(0));
    void AttachEnd(IPhysicsRigidBody* body, const Vec3& localOffset = Vec3(0));

    // Simulation properties
    void SetStiffness(float stiffness); // Resistance to bending (0 = string, high = cable)
    void SetDamping(float damping);
    void SetFriction(float friction);

    PhysXArticulation* GetArticulation() const { return m_Articulation; }
    std::vector<PhysXArticulationLink*>& GetLinks() { return m_Links; }

private:
    PhysXBackend* m_Backend;
    PhysXArticulation* m_Articulation;
    std::vector<PhysXArticulationLink*> m_Links;
    
    // Attachments (using raw PhysX constraints)
    void* m_StartAttachment; // PxJoint*
    void* m_EndAttachment;   // PxJoint*
    
    void DestroyAttachments();
};
