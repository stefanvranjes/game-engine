#pragma once

#include "PhysXArticulation.h"
#include "PhysXArticulationLink.h"
#include "PhysXArticulationJoint.h"
#include "Bone.h"
#include "Animator.h"
#include "IPhysicsShape.h"
#include <map>
#include <string>
#include <memory>
#include <vector>

class PhysXBackend;

enum class RagdollState {
    Kinematic,  // Animated completely (physics disabled or kinematic)
    Dynamic,    // Full physics (floppy ragdoll)
    Active      // Physics driven by animation targets (active/powered ragdoll)
};

struct RagdollBoneConfig {
    std::string boneName;
    float mass = 1.0f;
    float friction = 0.5f;
    float restitution = 0.0f;
    float linearDamping = 0.5f;
    float angularDamping = 0.5f;
    
    // Joint Limits (in degrees)
    float twistLimitLow = -45.0f;
    float twistLimitHigh = 45.0f;
    float swing1Limit = 45.0f;
    float swing2Limit = 45.0f;
    
    // Active Drive Params
    float driveStiffness = 1000.0f;
    float driveDamping = 100.0f;
    float driveForceLimit = FLT_MAX;
};

#ifdef USE_PHYSX
class PhysXRagdoll {
public:
    PhysXRagdoll(PhysXBackend* backend, std::shared_ptr<Skeleton> skeleton);
    ~PhysXRagdoll();

    // Initialization
    void Initialize(const Vec3& rootPosition, const Quat& rootRotation);
    
    // Configuration
    PhysXArticulationLink* AddBone(const RagdollBoneConfig& config, std::shared_ptr<IPhysicsShape> shape);
    
    // State Management
    void SetState(RagdollState state);
    RagdollState GetState() const { return m_State; }
    
    // Runtime
    void Update(float deltaTime);
    
    // Sync Methods
    // Call this when State == Kinematic to snap physics to animation
    void MatchAnimation(Animator* animator); 
    
    // Call this when State == Active to drive physics towards animation
    void DriveToAnimation(Animator* animator);
    
    // Call this when State == Dynamic to apply physics results to the Transform/Animator
    // Note: This usually requires a way to write back to the Animator's transforms or a POST-animator pass
    void ApplyPoseToSkeleton(std::vector<Mat4>& outGlobalTransforms);

    PhysXArticulation* GetArticulation() const { return m_Articulation.get(); }
    PhysXArticulationLink* GetLink(const std::string& boneName) const;

private:
    PhysXBackend* m_Backend;
    std::shared_ptr<Skeleton> m_Skeleton;
    PhysXArticulation* m_Articulation;
    
    struct BoneEntry {
        int boneIndex;
        PhysXArticulationLink* link;
        PhysXArticulationJoint* joint; // Null for root
        RagdollBoneConfig config;
        Mat4 bindPoseInv; // Inverse of bind pose for relative calculations
    };

    std::map<std::string, BoneEntry> m_Bones;
    std::vector<BoneEntry*> m_OrderedBones; // Ordered by hierarchy ideally

    RagdollState m_State;
};
