#include "PhysXRope.h"
#include "PhysXBackend.h"
#include "PhysXShape.h" // Assuming this exists or needed for shapes
#include "PhysXRigidBody.h" // For attachment casting

#ifdef USE_PHYSX
#include <PxPhysicsAPI.h>
using namespace physx;
#endif

PhysXRope::PhysXRope(PhysXBackend* backend)
    : m_Backend(backend)
    , m_Articulation(nullptr)
    , m_StartAttachment(nullptr)
    , m_EndAttachment(nullptr)
{
}

PhysXRope::~PhysXRope() {
    DestroyAttachments();
    if (m_Articulation) {
        delete m_Articulation;
    }
}

void PhysXRope::DestroyAttachments() {
#ifdef USE_PHYSX
    if (m_StartAttachment) {
        static_cast<PxJoint*>(m_StartAttachment)->release();
        m_StartAttachment = nullptr;
    }
    if (m_EndAttachment) {
        static_cast<PxJoint*>(m_EndAttachment)->release();
        m_EndAttachment = nullptr;
    }
#endif
}

void PhysXRope::Initialize(const Vec3& start, const Vec3& end, int segments, float radius, float totalMass) {
    if (!m_Backend || segments < 1) return;

    m_Articulation = new PhysXArticulation(m_Backend);
    m_Articulation->Initialize(false); // Floating base

    Vec3 diff = end - start;
    float length = diff.Length();
    Vec3 dir = diff / length;
    float segmentLength = length / segments;
    float segmentMass = totalMass / segments;

    PhysXArticulationLink* parent = nullptr;
    
    // Need a generic capsule shape
    // Ideally we reuse shapes, but here we might need to create one manually or via backend
    // For now, we will let the Initialize of link create a default shape if we pass null? 
    // No, Initialize needs a shape.
    // Let's rely on PhysXBackend to provide a default material helper or create loose shape.
    
#ifdef USE_PHYSX
    PxMaterial* mat = m_Backend->GetDefaultMaterial();
    // Start position of the first link center
    // We want the rope to span from Start to End.
    // Each segment is a capsule of height 'segmentLength'.
    // PhysX capsules are long along X usually? No, standard is Y?
    // Let's assume capsules are Y-aligned locally.
    
    // Rotation to align Y axis with 'dir'
    Vec3 up(0, 1, 0);
    Quat rotation = Quat::FromToRotation(up, dir);
    
    for (int i = 0; i < segments; ++i) {
        // Calculate center position of this segment
        // Start + dir * (i * len + len/2)
        float dist = (i * segmentLength) + (segmentLength * 0.5f);
        Vec3 pos = start + dir * dist;
        
        PhysXArticulationLink* link = m_Articulation->AddLink(parent, pos, rotation);
        if (!link) continue;
        
        m_Links.push_back(link);

        // We need to access the native link to add a shape directly, 
        // OR we wrap a shape.
        // Let's create a raw PxShape and attach.
        PxShape* shape = m_Backend->GetPhysics()->createShape(PxCapsuleGeometry(radius, segmentLength * 0.5f), *mat);
        // Note: PxCapsuleGeometry takes halfHeight.
        
        // Link wrapper `Initialize` calls attachShape if we pass a generic shape wrapper.
        // But we don't have a factory for generic shapes exposed easily here in this snippet.
        // So we will access native directly.
        PxRigidBodyExt::updateMassAndInertia(*static_cast<PxArticulationLink*>(link->GetNativeBody()), segmentMass > 0 ? segmentMass : 1.0f);
        static_cast<PxArticulationLink*>(link->GetNativeBody())->attachShape(*shape);
        shape->release();
        
        // Configure Joint (if not root)
        PhysXArticulationJoint* joint = link->GetInboundJoint();
        if (joint) {
            // Anchor location
            // By default, child frame is at child center.
            // Parent frame is at parent center.
            // We want the joint to be at the boundary between capsules.
            
            // Parent anchor: +Y * (halfHeight)
            // Child anchor: -Y * (halfHeight)
            float halfLen = segmentLength * 0.5f;
            
            PxArticulationJointReducedCoordinate* pxJoint = joint->GetNativeJoint();
            pxJoint->setParentPose(PxTransform(PxVec3(0, halfLen, 0))); // Local to parent
            pxJoint->setChildPose(PxTransform(PxVec3(0, -halfLen, 0))); // Local to child
            
            // Free rotation (spherical)
            joint->SetMotion(PhysXArticulationJoint::Axis::Swing1, PhysXArticulationJoint::Motion::Free);
            joint->SetMotion(PhysXArticulationJoint::Axis::Swing2, PhysXArticulationJoint::Motion::Free);
            joint->SetMotion(PhysXArticulationJoint::Axis::Twist, PhysXArticulationJoint::Motion::Free);
        }
        
        parent = link;
    }
    
    // Add to scene
    m_Backend->GetScene()->addArticulation(*static_cast<PxArticulationReducedCoordinate*>(m_Articulation->GetNativeArticulation()));
#endif
}

void PhysXRope::AttachStart(IPhysicsRigidBody* body, const Vec3& localOffset) {
#ifdef USE_PHYSX
    if (m_Links.empty()) return;
    
    // Destroy old
    if (m_StartAttachment) {
        static_cast<PxJoint*>(m_StartAttachment)->release();
        m_StartAttachment = nullptr;
    }
    
    PhysXArticulationLink* firstLink = m_Links[0];
    PxRigidActor* actor0 = body ? static_cast<PxRigidActor*>(body->GetNativeWorld()) : nullptr; // Assuming GetNativeWorld returns Actor
    PxRigidActor* actor1 = static_cast<PxRigidActor*>(static_cast<PxArticulationLink*>(firstLink->GetNativeBody()));
    
    // Connect actor0 (body) to actor1 (rope start)
    // Anchor on rope: Top of first capsule -> -Y * halfHeight
    // Wait, in Initialize we placed center at len/2.
    // So "start" of rope is at -Y * halfHeight in local space of first link.
    
    // Get half height from geometry (approx)
    // Or just assume initialization logic:
    // PxShape...
    
    PxTransform t0 = PxTransform(PxVec3(localOffset.x, localOffset.y, localOffset.z));
    PxTransform t1 = PxTransform(PxVec3(0, -1.0f, 0)); // Placeholder, need actual half length

    // Re-calculating half length from shape would be robust, but let's assume standard setup or fetch it.
    PxShape* shape;
    if (actor1->getNbShapes() > 0) {
        actor1->getShapes(&shape, 1);
        PxCapsuleGeometry geom;
        if (shape->getCapsuleGeometry(geom)) {
            t1 = PxTransform(PxVec3(0, -geom.halfHeight, 0));
        }
    }
    
    // Use PxFixedJoint or Spherical
    // Spherical allows rope to hinge at attachment
    PxSphericalJoint* joint = PxSphericalJointCreate(*m_Backend->GetPhysics(), actor0, t0, actor1, t1);
    m_StartAttachment = joint;
#endif
}

void PhysXRope::AttachEnd(IPhysicsRigidBody* body, const Vec3& localOffset) {
#ifdef USE_PHYSX
    if (m_Links.empty()) return;
    
    // Destroy old
    if (m_EndAttachment) {
        static_cast<PxJoint*>(m_EndAttachment)->release();
        m_EndAttachment = nullptr;
    }
    
    PhysXArticulationLink* lastLink = m_Links.back();
    PxRigidActor* actor0 = body ? static_cast<PxRigidActor*>(body->GetNativeWorld()) : nullptr;
    PxRigidActor* actor1 = static_cast<PxRigidActor*>(static_cast<PxArticulationLink*>(lastLink->GetNativeBody()));
    
    PxTransform t0 = PxTransform(PxVec3(localOffset.x, localOffset.y, localOffset.z));
    PxTransform t1 = PxTransform(PxVec3(0, 1.0f, 0)); 
    
    PxShape* shape;
    if (actor1->getNbShapes() > 0) {
        actor1->getShapes(&shape, 1);
        PxCapsuleGeometry geom;
        if (shape->getCapsuleGeometry(geom)) {
            t1 = PxTransform(PxVec3(0, geom.halfHeight, 0)); // End is +Y
        }
    }
    
    PxSphericalJoint* joint = PxSphericalJointCreate(*m_Backend->GetPhysics(), actor0, t0, actor1, t1);
    m_EndAttachment = joint;
#endif
}

void PhysXRope::SetStiffness(float stiffness) {
    for (auto* link : m_Links) {
        PhysXArticulationJoint* joint = link->GetInboundJoint();
        if (joint) {
            // Apply stiffness to all axes to keep straight
            joint->SetDrive(PhysXArticulationJoint::Axis::Swing1, stiffness, stiffness * 0.1f, FLT_MAX, PhysXArticulationJoint::DriveType::Target);
            joint->SetDrive(PhysXArticulationJoint::Axis::Swing2, stiffness, stiffness * 0.1f, FLT_MAX, PhysXArticulationJoint::DriveType::Target);
            joint->SetDriveTarget(PhysXArticulationJoint::Axis::Swing1, 0.0f);
            joint->SetDriveTarget(PhysXArticulationJoint::Axis::Swing2, 0.0f);
        }
    }
}

void PhysXRope::SetDamping(float damping) {
     for (auto* link : m_Links) {
        link->SetLinearDamping(damping);
        link->SetAngularDamping(damping);
     }
}

void PhysXRope::SetFriction(float friction) {
    for (auto* link : m_Links) {
        link->SetFriction(friction);
    }
}
